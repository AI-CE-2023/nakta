import json

import torch

from nakta_model2 import LLaMA, ModelArgs, Transformer

torch.set_default_tensor_type(torch.cuda.HalfTensor)

ckpt_dir = ["./weights/pp/merged_0.pth", "./weights/pp/merged_1.pth"]

with open("./params.json", "r") as f:
    params = json.loads(f.read())

model_args: ModelArgs = ModelArgs(**params)
model_args.vocab_size = 32000


devices = ["cuda:0", "cuda:1"]

torch.set_default_device(devices[0])

model = Transformer(model_args, 0)

torch.load(ckpt_dir[0], map_location="cpu")
