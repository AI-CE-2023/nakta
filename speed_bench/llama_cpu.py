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
from transformers import LlamaPreTrainedModel

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
            # t = t[il - cl : il]
            # r = r[il - cl - 1 : il - 1, :]
            t = t[il - cl : il - 1]
            r = r[il - cl - 1 : il - 2, :]
            logits = torch.gather(r, 1, t.unsqueeze(-1)).sum() / csl
            # logits = torch.gather(r, 1, t.unsqueeze(-1)).sum()
            sums.append(logits)
        # print(sums)
        to_return.append(1 if g == torch.argmax(torch.tensor(sums)) else 0.0)
    return to_return


def main(
    ckpt_dir: str,
    tokenizer_path: str,
):

    model = LlamaForCausalLM.from_pretrained(ckpt_dir, use_safetensors=True)

    default_batch_size = 64
    assert default_batch_size % 4 == 0

    cache = False

    valid_datas = SpeedDataset(
        # validset,
        tokenizer_path=tokenizer_path,
        order="ascending",
        default_batch_size=default_batch_size,
        device=f"cpu",
    )

    dataloader = DataLoader(
        valid_datas, batch_size=1, shuffle=True, collate_fn=collate_fn
    )

    # first_batch = next(iter(dataloader))

    # for _ in range(2):
    #     generator.bench(first_batch, cache=cache)

    tfs = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_event.record()
    for tokens, inp_lens, continuation_lens, golds, cont_str_lens in tqdm(dataloader):
        result = model(tokens)
        result = F.log_softmax(result, dim=-1)
        tfs.extend(
            is_tf(tokens, result, inp_lens, continuation_lens, golds, cont_str_lens)
        )
    end_event.record()
    torch.cuda.synchronize()
    total_time = start_event.elapsed_time(end_event) / 1000
    print(total_time)

    model_name = "nakta_cpu"

    # 결과를 JSON 파일에 저장
    result = {
        "model_name": model_name,
        "execution_time": total_time,
        "cache": cache,
        "accuracy": sum(tfs) / len(tfs),
    }
    print(result["accuracy"])
    with open(
        f"{model_name}_speed_test_{default_batch_size}.json", "w"
    ) as json_file:
        json.dump(result, json_file, indent=4)


if __name__ == "__main__":
    main(
        ckpt_dir="/home/llama-30b",
        tokenizer_path="../weights/original/tokenizer.model",
    )

"""
python llama_cpu.py
"""