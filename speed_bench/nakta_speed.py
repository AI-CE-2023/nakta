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
from sch.sch_nakta import SpeedDataset, collate_fn
from torch.utils.data import DataLoader
from tqdm import tqdm

from nakta_model import LLaMA, ModelArgs, Tokenizer, Transformer

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

    model_args: ModelArgs = ModelArgs(**params)
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    generator = LLaMA(model, tokenizer)
    load_time = time.time() - start_time
    print(f"Loaded in {load_time:.2f} seconds")
    return generator, load_time


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
            t = t[il - cl : il - 1]
            r = r[il - cl - 1 : il - 2, :]
            pre_logits = torch.gather(r, 1, t.unsqueeze(-1))
            logits = pre_logits.sum() / csl
            # logits = pre_logits.sum()
            sums.append(logits)
        # print(sums)
        to_append = 1 if g == torch.argmax(torch.tensor(sums)) else 0.0
        # print(to_append)
        to_return.append(to_append)
    return to_return


def main(
    ckpt_dir: str,
    tokenizer_path: str,
):
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    generator, load_time = load(ckpt_dir, tokenizer_path, local_rank, world_size)

    # with open("./test.pickle", "rb") as fr:
    #     validset = pickle.load(fr)

    default_batch_size = 16
    candidate_mtp = 6
    cache = True
    order = "descending"

    data_start = time.time()

    valid_datas = SpeedDataset(
        # validset,
        tokenizer_path=tokenizer_path,
        order=order,
        default_batch_size=default_batch_size,
        device=f"cuda:{local_rank}",
        candidate_multiple=candidate_mtp,
    )

    dataloader = DataLoader(
        valid_datas, batch_size=1, shuffle=True, collate_fn=collate_fn
    )

    data_end = time.time()
    # first_batch = next(iter(dataloader))

    # for _ in range(2):
    #     generator.bench(first_batch, cache=cache)
    tfs = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_event.record()
    for (
        ctx_tokens,
        following_tokens,
        min_ctx,
        _,
        _,
        inp_lens,
        f_lens,
        targets,
        fs_lens,
    ) in tqdm(dataloader):
        generator.model.forward(ctx_tokens, (0, 1, 4))
        result = generator.model.forward(following_tokens, (min_ctx, 1, 4))
        result = F.log_softmax(result, dim=-1)
        # result = result.cpu()
        result = (
            result.reshape(4, result.shape[0] // 4, result.shape[1], -1)
            .permute(1, 0, 2, 3)
            .contiguous()
            .view(result.shape[0], result.shape[1], -1)
        )
        # ctx_tokens = ctx_tokens.cpu()
        ctx_tokens = ctx_tokens.repeat(4, 1)
        # following_tokens = following_tokens.cpu()
        tokens = torch.cat((ctx_tokens, following_tokens), dim=1)
        tokens = (
            tokens.reshape(4, result.shape[0] // 4, -1)
            .permute(1, 0, 2)
            .contiguous()
            .view(tokens.shape[0], -1)
        )
        assert tokens.shape == result.shape[:-1]

        tfs.extend(
            is_tf(
                tokens,
                result,
                inp_lens,
                f_lens,
                targets,
                fs_lens,
            )
        )
    end_event.record()
    torch.cuda.synchronize()

    total_time = start_event.elapsed_time(end_event) / 1000
    # print(total_time)

    model_name = "nakta"

    if local_rank == 0:
        # 결과를 JSON 파일에 저장
        result = {
            "model_name": model_name,
            "inference_time": total_time,
            "preprocessing time": data_end - data_start,
            "model load time": load_time,
            "cache": cache,
            "acc norm": sum(tfs) / len(tfs),
        }
        # print(result["accuracy"])
        # with open(
        #     f"./{model_name}_speed_test_{default_batch_size*4}_{candidate_mtp}_{order}.json",
        #     "w",
        # ) as json_file:
        #     json.dump(result, json_file, indent=4)
        with open(
            f"./{model_name}_speed_test.json",
            "w",
        ) as json_file:
            json.dump(result, json_file, indent=4)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the main function with command line arguments.")
    parser.add_argument("--ckpt_dir", type=str, default="../weights/modified/30B", help="Checkpoint directory path")
    parser.add_argument("--tokenizer_path", type=str, default="../weights/original/tokenizer.model", help="Tokenizer path")

    args = parser.parse_args()
    main(args.ckpt_dir, args.tokenizer_path)

"""
torchrun --nproc_per_node 4 nakta_speed.py
"""
