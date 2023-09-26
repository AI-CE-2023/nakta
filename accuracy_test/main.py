import argparse
import json
import logging
import os
from dataclasses import asdict, dataclass
from typing import Tuple

import torch
from fairscale.nn.model_parallel.initialize import initialize_model_parallel


@dataclass
class ModelArgs:
    ckpt_dir: str
    tokenizer_path: str
    local_rank: int = 0
    world_size: int = 1
    max_seq_len: int = 512
    max_batch_size: int = 128
    batch_size: int = 64

    def asdict(self):
        return asdict(self)


def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


from lm_eval import evaluator, tasks, utils

logging.getLogger("openai").setLevel(logging.WARNING)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument(
        "--tasks", default=None, choices=utils.MultiChoice(tasks.ALL_TASKS)
    )
    parser.add_argument("--provide_description", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=str, default=None)
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=None,
        help="Maximal batch size to try with --batch_size auto",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help="Limit the number of examples per task. "
        "If <1, limit is a percentage of the total number of examples.",
    )
    parser.add_argument("--data_sampling", type=float, default=None)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--decontamination_ngrams_path", default=None)
    parser.add_argument("--description_dict_path", default=None)
    parser.add_argument("--check_integrity", action="store_true")
    parser.add_argument("--write_out", action="store_true", default=False)
    parser.add_argument("--output_base_path", type=str, default=None)

    return parser.parse_args()


def main(local_rank, world_size):
    args = parse_args()

    assert not args.provide_description  # not implemented

    if args.limit:
        print(
            "WARNING: --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )

    if args.tasks is None:
        task_names = tasks.ALL_TASKS
    else:
        task_names = utils.pattern_match(args.tasks.split(","), tasks.ALL_TASKS)

    print(f"Selected Tasks: {task_names}")

    description_dict = {}
    if args.description_dict_path:
        with open(args.description_dict_path, "r") as f:
            description_dict = json.load(f)

    model_args = ModelArgs(
        ckpt_dir=args.ckpt_dir,
        tokenizer_path=args.tokenizer_path,
        local_rank=local_rank,
        world_size=world_size,
    )

    results = evaluator.simple_evaluate(
        model=args.model,
        model_args=model_args,
        tasks=task_names,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        device=args.device,
        no_cache=args.no_cache,
        limit=args.limit,
        description_dict=description_dict,
        decontamination_ngrams_path=args.decontamination_ngrams_path,
        check_integrity=args.check_integrity,
        write_out=args.write_out,
        output_base_path=args.output_base_path,
    )

    if local_rank == 0:
        dumped = json.dumps(results, indent=2)
        print(dumped)

        if args.output_path:
            os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
            with open(args.output_path, "w") as f:
                f.write(dumped)

        print(evaluator.make_table(results))


if __name__ == "__main__":
    local_rank, world_size = setup_model_parallel()
    main(local_rank, world_size)

"""
torchrun --nproc_per_node 4 main.py \
    --model nakta \
    --ckpt_dir ../weights/modified/30B_2 \
    --tokenizer_path ../weights/original/tokenizer.model
    --tasks hellaswag 
    --output_path ./accuracy_test_result_nakta 
"""
"""
torchrun --nproc_per_node 4 main.py \
    --model llama \
    --ckpt_dir ../weights/original/30B \
    --tokenizer_path ../weights/original/tokenizer.model \
    --tasks hellaswag 
    --output_path ./accuracy_test_result_llama 
"""
