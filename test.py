import json
import os
import time

import torch
from torch.distributed.pipeline.sync import Pipe

from llama_org2 import LLaMA, ModelArgs, Transformer


def set_global_settings():
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"


def load_model() -> LLaMA:
    with open("./params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(**params)
    model_args.vocab_size = 320000

    devices = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
    transformers = []

    for idx, device in enumerate(devices):
        torch.set_default_device(device)
        transformers.append(Transformer(model_args, idx))

    torch.set_default_device("cuda:0")
    model = torch.nn.Sequential(*transformers)
    model = Pipe(model, chunks=8)

    return model


def main(ctx_len: int = 70, follow_len: int = 80, batch_size: int = 128 * 1):
    set_global_settings()

    torch.distributed.rpc.init_rpc("worker", rank=0, world_size=1)
    tokens = torch.randint(1, 32000, (batch_size, ctx_len)).cuda()
    model = load_model()
    # t_list = []
    # print(tokens.shape)
    # for _ in range(10):
    #     a = time.time()
    #     result = model(tokens, 0, 0)
    #     b = time.time()
    #     print(b - a)
    #     t_list.append(b - a)
    # print(f"{sum(t_list)/len(t_list)}")
    # print(result.to_here().shape)
    # print(result.shape)
    time.sleep(20)
    # If there's a plan to use the model for inference, add the inference code here.
    # For now, it simply sleeps for 20 seconds before ending the program.
    # e.g., results = generator.prof(ctx_len, follow_len, batch_size)
    # torch.save(results, "./original.pt")


if __name__ == "__main__":
    main()
