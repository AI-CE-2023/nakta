import json
import os
import time
from pathlib import Path

import torch
from torch.distributed.pipeline.sync import Pipe

from nakta_model2 import LLaMA, ModelArgs, Transformer


def set_global_settings():
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"


def load_model() -> LLaMA:
    ckpt_dir = ["./weights/pp/merged_0.pth", "./weights/pp/merged_1.pth"]

    # checkpoints = list(sorted(Path(ckpt_dir).glob("*.pth")))
    cs = []
    for i, p in enumerate(ckpt_dir[:1]):
        cs.append(torch.load(p, map_location="cpu"))
    with open("./params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(**params)
    model_args.vocab_size = 32000

    # emb = torch.nn.Embedding(model_args.vocab_size, model_args.)

    devices = ["cuda:0", "cuda:1"]
    transformers = []

    for idx, device in enumerate(devices):
        torch.set_default_device(device)
        transformers.append(
            Transformer(model_args, idx).load_state_dict(cs[idx], strict=False)
        )

    # proj_devices = ["cpu"]
    # for idx, device in enumerate(devices):
    #     torch.set_default_device(device)
    #     transformers.append(
    #         torch.nn.Sequential(
    #             RMSNorm(model_args.dim, model_args.norm_eps),
    #             torch.nn.Linear(model_args.dim, model_args.vocab_size, bias=False),
    #         )
    #     )
    torch.set_default_device("cuda:0")
    model = torch.nn.Sequential(*transformers)
    model = Pipe(model, chunks=1)

    return model


def main(ctx_len: int = 200, follow_len: int = 80, batch_size: int = 4 * 1):
    set_global_settings()

    torch.distributed.rpc.init_rpc("worker", rank=0, world_size=1)
    tokens = torch.randint(1, 32000, (batch_size, ctx_len)).cuda()
    model = load_model()
    # print(model)
    print("load complete")
    t_list = []
    print(tokens.shape)
    for _ in range(10):
        a = time.time()
        result = model(tokens, (0, -1, 0), 0)
        b = time.time()
        print(b - a)
        t_list.append(b - a)
    print(f"{sum(t_list)/len(t_list)}")
    print(result.to_here().shape)
    # print(result.shape)
    # time.sleep(20)


if __name__ == "__main__":
    main()
