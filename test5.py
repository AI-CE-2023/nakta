import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from torch.distributed.pipeline.sync import Pipe

from nakta_model2 import LLaMA, ModelArgs, Transformer


def parallel_load(paths):
    with ThreadPoolExecutor(max_workers=len(paths)) as executor:
        return list(executor.map(torch.load, paths))


def set_global_settings():
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"


def load_model() -> LLaMA:
    s = time.time()
    ckpt_dirs = ["./weights/pp2/merged_0.pth", "./weights/pp2/merged_1.pth"]

    # checkpoints = list(sorted(Path(ckpt_dir).glob("*.pth")))
    cs = parallel_load(ckpt_dirs)
    # for i, p in enumerate(ckpt_dir):
    #     cs.append(torch.load(p, map_location="cpu"))
    with open("./params.json", "r") as f:
        params = json.loads(f.read())
    e1 = time.time()
    print(f"param load time: {e1 - s}")
    model_args: ModelArgs = ModelArgs(**params)
    model_args.vocab_size = 32000

    # emb = torch.nn.Embedding(model_args.vocab_size, model_args.)

    devices = ["cuda:0", "cuda:1"]
    transformers = []

    for idx, device in enumerate(devices):
        torch.set_default_device(device)
        model = Transformer(model_args, idx)
        model.load_state_dict(cs[idx], strict=False)
        transformers.append(model)
    torch.set_default_device("cuda:0")
    model = torch.nn.Sequential(*transformers)
    model = Pipe(model, chunks=1)
    e = time.time()
    print(f"model load time {e - s}")
    return model


def main(ctx_len: int = 70, follow_len: int = 40, batch_size: int = 16):
    set_global_settings()

    torch.distributed.rpc.init_rpc("worker", rank=0, world_size=1)
    # tokens = torch.randint(1, 32000, (batch_size, ctx_len)).cuda()
    model = load_model()
    # tokens = torch.load("./tokens.pt").cuda()
    ctx_tokens = torch.randint(1, 32000, (batch_size // 4, ctx_len))
    follow_tokens = torch.randint(1, 32000, (batch_size, follow_len))
    # torch.save(ctx_tokens, "./ctx.pt")
    # torch.save(follow_tokens, "./follow.pt")
    # print(model)
    print("load complete")
    t_list = []
    # print(tokens.shape)
    for _ in range(10):
        a = time.time()
        model(ctx_tokens, (0, 1, 4), 0)
        result = model(follow_tokens, (ctx_len, 1, 4), 0)
        b = time.time()
        print(b - a)
        t_list.append(b - a)
    print(f"{sum(t_list)/len(t_list)}")
    print(result.to_here().shape)
    # torch.save(result.to_here(), "./ref.pt")
    # print(result.shape)
    # time.sleep(20)


if __name__ == "__main__":
    main()
