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

# test
from nakta_model4 import LLaMA, ModelArgs, Tokenizer, Transformer


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
    ctx_len: int = 60,
    follow_len: int = 40,
    batch_size: int = 64,
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
    torch.manual_seed(0)
    ctx_lens = [ctx_len] * (batch_size // 4)
    follow_lens = [follow_len] * batch_size
    ctx = torch.randint(1, 32000, (sum(ctx_lens),))
    follow = torch.randint(1, 32000, (sum(follow_lens),))
    results = generator.test_equal(ctx_lens, follow_lens, ctx, follow)
    print(results.shape)
    if local_rank == 0:
        #     torch.save(generator.model.cpu().state_dict(), "./test.pt")
        ctx = ctx.split(ctx_lens)
        follow = follow.split(follow_lens)
        ctx = [i.unsqueeze(dim=0).repeat(4, 1) for i in ctx]
        ctx = torch.cat(ctx, dim=0)
        follow = torch.stack(follow, dim=0)
        tokens = torch.cat([ctx, follow], dim=1)
        torch.save(tokens, "./test_data.pt")
        torch.save(results, "./nakta.pt")


if __name__ == "__main__":
    fire.Fire(main)

"""
CUDA_LAUNCH_BLOCKING=1 nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas --force-overwrite true -o ./model_profile/nakta4_c.nsys-rep torchrun --nproc_per_node 4 9_profile_nakta4.py --ckpt_dir ./weights/modified/30B --tokenizer_path ./weights/original/tokenizer.model
"""