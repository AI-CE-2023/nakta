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

from llama_org2 import LLaMA, ModelArgs, Tokenizer, Transformer



def load(
    # ckpt_dir: str,
    # tokenizer_path: str,
    # local_rank: int,
    # world_size: int,
) -> LLaMA:
    # start_time = time.time()
    # checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    # assert world_size == len(
    #     checkpoints
    # ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    # ckpt_path = checkpoints[local_rank]
    # print("Loading")
    # checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open("./params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(**params)
    # tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = 20
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    # torch.set_default_tensor_type(torch.FloatTensor)
    # model.load_state_dict(checkpoint, strict=False)

    # generator = LLaMA(model, tokenizer)
    # print(f"Loaded in {time.time() - start_time:.2f} seconds")
    # return generator


def main(
    # ckpt_dir: str,
    # tokenizer_path: str,
    ctx_len: int = 60,
    follow_len: int = 40,
    batch_size: int = 64,
):


    generator = load(
        # ckpt_dir,
        # tokenizer_path,
        # local_rank,
        # world_size,
    )
    time.sleep(20)
    # results = generator.prof(ctx_len, follow_len, batch_size)
    # torch.save(results, "./original.pt")


if __name__ == "__main__":
    fire.Fire(main)

"""
CUDA_LAUNCH_BLOCKING=1 nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas --force-overwrite true -o ./model_profile/original.nsys-rep torchrun --nproc_per_node 4 6_profile_original.py --ckpt_dir ./weights/original/30B --tokenizer_path ./weights/original/tokenizer.model
"""
