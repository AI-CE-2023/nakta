import csv
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Tuple

import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from tqdm import tqdm

from nakta_model import LLaMA, ModelArgs, Tokenizer, Transformer


def save_to_csv(
    ctx_lens, cached_speeds, n_cached_speeds, filename="speed_test_results.csv"
):
    """Save the results to a CSV file"""
    with open(filename, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(
            ["ctx_len", "cached_speed", "n_cached_speed"]
        )  # Writing headers
        for ctx, cached, non_cached in zip(ctx_lens, cached_speeds, n_cached_speeds):
            csvwriter.writerow([ctx, cached, non_cached])


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

    generator = load(
        ckpt_dir,
        tokenizer_path,
        local_rank,
        world_size,
    )

    start_ctx = 10
    end_ctx = 100
    bin_size = 10

    # batch_sizes = [16, 32, 64, 128]
    batch_sizes = [72, 80, 92]

    results = []

    for batch_size in batch_sizes:
        ctx_lens = []
        cached_speeds = []
        n_cached_speeds = []

        for ctx_len in tqdm(range(start_ctx, end_ctx + bin_size, bin_size)):
            follow_len = min(50, int(ctx_len * (2 / 3)))

            cached_speed, n_cached_speed = generator.speed_test(
                ctx_len, follow_len, batch_size
            )

            ctx_lens.append(ctx_len)
            cached_speeds.append(cached_speed)
            n_cached_speeds.append(n_cached_speed)

        results.append(
            {
                "batch_size": batch_size,
                "ctx_lens": ctx_lens,
                "cached_speeds": cached_speeds,
                "n_cached_speeds": n_cached_speeds,
            }
        )

        # Saving the data for each batch_size
        filename = f"speed_test_results_batch_{batch_size}.csv"
        save_to_csv(ctx_lens, cached_speeds, n_cached_speeds, filename=filename)

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(ctx_lens, cached_speeds, label="cached_speed", marker="o")
        plt.plot(ctx_lens, n_cached_speeds, label="n_cached_speed", marker="x")
        plt.xlabel("ctx_len")
        plt.ylabel("Speed")
        plt.title(
            f"Cached Speed vs Non-Cached Speed by Context Length (Batch Size: {batch_size})"
        )
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            f"Cached Speed vs Non-Cached Speed by Context Length (Batch Size: {batch_size}).png"
        )
        plt.close()


# ... (rest of the code unchanged)


if __name__ == "__main__":
    main(
        ckpt_dir="../../weights/modified/30B_2",
        tokenizer_path="../../weights/original/tokenizer.model",
    )
"""
torchrun --nproc_per_node 4 test_speed_by_len_batch.py
"""
