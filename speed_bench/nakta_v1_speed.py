import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Tuple

import fire
import numpy as np
import torch
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from sch.sch_llama import SpeedDataset, collate_fn
from torch.utils.data import DataLoader
from tqdm import tqdm

from nakta_model_v1 import LLaMA, ModelArgs, Tokenizer, Transformer


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
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def main(
    ckpt_dir: str,
    tokenizer_path: str,
):
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    generator = load(ckpt_dir, tokenizer_path, local_rank, world_size)

    with open("./test.pickle", "rb") as fr:
        validset = pickle.load(fr)

    default_batch_size = 1

    valid_datas = SpeedDataset(
        validset,
        tokenizer_path=tokenizer_path,
        order="descending",
        default_batch_size=default_batch_size,
        device=f"cuda:{local_rank}",
    )

    dataloader = DataLoader(
        valid_datas, batch_size=1, shuffle=True, collate_fn=collate_fn
    )

    first_batch = next(iter(dataloader))

    for _ in range(2):
        generator.bench(first_batch)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_event.record()
    for batch in tqdm(dataloader):
        generator.bench(batch)
    end_event.record()
    torch.cuda.synchronize()
    total_time = start_event.elapsed_time(end_event) / 1000
    print(total_time)

    model_name = "nakta_v1"

    # 결과를 JSON 파일에 저장
    result = {
        "model_name": model_name,
        "execution_time": total_time,
    }

    with open(f"{model_name}_speed_test_{default_batch_size}.json", "w") as json_file:
        json.dump(result, json_file, indent=4)


if __name__ == "__main__":
    main(
        ckpt_dir="../weights/modified/30B_2",
        tokenizer_path="../weights/original/tokenizer.model",
    )
"""
torchrun --nproc_per_node 4 nakta_v1_speed.py
"""
