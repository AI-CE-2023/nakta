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
    max_seq_len: int,
    max_batch_size: int,
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

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
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
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_seq_len: int = 512,
    max_batch_size: int = 32,
):
    seq_num = max_batch_size

    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    generator = load(
        ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size
    )

    prompts = [
        """‘query’: 'Removing ice from car: Then, the man writes over the snow covering the window of a car, and a woman wearing winter clothes smiles. then'
    ‘choices’: [', the man adds wax to the windshield and cuts it.', ', a person board a ski lift, while two men supporting the head of the person wearing winter clothes snow as the we girls sled.', ', the man puts on a christmas coat, knitted with netting.', ', the man continues removing the snow on his car.']"""
        for _ in range(seq_num)
    ]

    for i in range(5):
        results = generator.accuracy(prompts)

    time_l = []
    num_trials = 5
    for _ in range(num_trials):
        results, to_append = generator.accuracy(prompts)

        print(to_append)
        time_l.append(to_append)

    print("mem-eff time is:", np.mean(time_l))
    print("mem-eff batch/time :", max_batch_size / np.mean(time_l))
    # torch.save(results, './test_custom.pt')


if __name__ == "__main__":
    fire.Fire(main)

"""
torchrun --nproc_per_node 4 1_nakta_non_gen_bench.py --ckpt_dir ./weights/original/30B --tokenizer_path ./weights/original/tokenizer.model  --max_batch_size 64
"""
