import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import List, Tuple

import fire
import numpy as np
import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from sch.sch_llama import SpeedDataset, collate_fn
from torch.utils.data import DataLoader
from tqdm import tqdm

from llama_org import LLaMA, ModelArgs, Tokenizer, Transformer


def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(max_batch_size=64, **params)
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def chunked(iterable, n):
    """Yield successive n-sized chunks from iterable."""
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]


def is_tf(
    tokens: torch.Tensor,
    results: torch.Tensor,
    inp_lens: List[int],
    cont_lens: List[int],
    golds: List,
    cont_str_lens: List[int],
):
    to_return = []
    for chunked_t, chunked_r, chunked_il, chunked_cl, g, chunked_csl in zip(
        chunked(tokens, 4),
        chunked(results, 4),
        chunked(inp_lens, 4),
        chunked(cont_lens, 4),
        golds,
        chunked(cont_str_lens, 4),
    ):
        sums = []
        for t, r, il, cl, csl in zip(
            chunked_t,
            chunked_r,
            chunked_il,
            chunked_cl,
            chunked_csl,
        ):
            t = t[il - cl : il]
            r = r[il - cl - 1 : il - 1, :]
            # logits = torch.gather(r, 1, t.unsqueeze(-1)).sum() / csl
            logits = torch.gather(r, 1, t.unsqueeze(-1)).sum()
            sums.append(logits)
        print(sums)
        to_return.append(1 if g == np.argmax(np.array(sums)) else 0.0)
    return to_return


def main(
    ckpt_dir: str,
    tokenizer_path: str,
):
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    generator = load(ckpt_dir, tokenizer_path, local_rank, world_size)

    # with open("./test2.pickle", "rb") as fr:
    #     validset = pickle.load(fr)[:128]

    default_batch_size = 64

    assert default_batch_size % 4 == 0

    valid_datas = SpeedDataset(
        # validset,
        tokenizer_path=tokenizer_path,
        order="None",
        default_batch_size=default_batch_size,
        device=f"cuda:{local_rank}",
    )

    dataloader = DataLoader(
        valid_datas, batch_size=1, shuffle=False, collate_fn=collate_fn
    )

    tfs = []
    # ts_event = torch.cuda.Event(enable_timing=True)
    # te_event = torch.cuda.Event(enable_timing=True)
    # tf_events = []

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_event.record()
    for tokens, inp_lens, continuation_lens, golds, cont_str_lens in tqdm(dataloader):
        result = F.log_softmax(generator.model(tokens, 0), dim=-1).cpu()
        # ts_event.record()
        tfs.extend(
            is_tf(
                tokens.cpu(), result, inp_lens, continuation_lens, golds, cont_str_lens
            )
        )
        # te_event.record()
        # tf_events.append(ts_event.elapsed_time(te_event) / 1000)

    end_event.record()
    torch.cuda.synchronize()
    total_time = start_event.elapsed_time(end_event) / 1000
    print(total_time)
    # print(sum(tf_events) * 100 / total_time)

    model_name = "llama"
    if local_rank == 0:
        # 결과를 JSON 파일에 저장
        result = {
            "model_name": model_name,
            "execution_time": total_time,
            "accuracy": sum(tfs) / len(tfs),
        }
        print(result["accuracy"])
        with open(
            f"./{model_name}_speed_test_{default_batch_size}.json", "w"
        ) as json_file:
            json.dump(result, json_file, indent=4)


if __name__ == "__main__":
    main(
        ckpt_dir="../weights/original/30B",
        tokenizer_path="../weights/original/tokenizer.model",
    )
"""
torchrun --nproc_per_node 4 llama_speed.py
"""
