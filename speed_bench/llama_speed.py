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
):
    sums = []
    for i in range(len(tokens)):
        token = tokens[i, inp_lens[i] - cont_lens[i] : inp_lens[i]]
        result = results[i, inp_lens[i] - cont_lens[i] - 1 : inp_lens[i] - 1, :]
        # token = tokens[i, : inp_lens[i]]
        # result = results[i, inp_lens[i] - cont_lens[i] : inp_lens[i], :]
        logits = torch.gather(result, 1, token.unsqueeze(-1)).sum()
        # print(logits)
        sums.append(logits)
    # print(sums)
    # print(golds)
    return [1 if golds[0] == np.argmax(np.array(sums)) else 0.0]
    # to_return = []
    # # assuming tokens, results, inp_lens, and golds have the same length
    # for chunk_tokens, chunk_results, chunk_inp_lens, chunk_cont_lens, gold in zip(
    #     chunked(tokens, 4),
    #     chunked(results, 4),
    #     chunked(inp_lens, 4),
    #     chunked(continuation_lens, 4),
    #     golds,
    # ):
    #     sums = []
    #     cnt = 0
    #     max_index = 0
    #     for token, result, inp_len, cont_len in zip(
    #         chunk_tokens, chunk_results, chunk_inp_lens, chunk_cont_lens
    #     ):
    #         token = token[inp_len - cont_len : inp_len].unsqueeze(0)
    #         result = result[inp_len - cont_len : inp_len, :].unsqueeze(0)
    #         logits = torch.gather(result, 2, token.unsqueeze(-1)).squeeze(-1)
    #         sums.append(logits.sum().item() / cont_len)
    #     to_return.append(1.0 if gold == sums.index(max(sums)) else 0.0)
    # return to_return


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

    default_batch_size = 1

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
    for tokens, inp_lens, continuation_lens, golds in tqdm(dataloader):
        result = F.log_softmax(generator.model(tokens, 0), dim=-1).cpu()
        # ts_event.record()
        tfs.extend(is_tf(tokens.cpu(), result, inp_lens, continuation_lens, golds))
        # te_event.record()
        # tf_events.append(ts_event.elapsed_time(te_event) / 1000)

    end_event.record()
    torch.cuda.synchronize()
    total_time = start_event.elapsed_time(end_event) / 1000
    print(total_time)
    # print(sum(tf_events) / len(tf_events))

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
