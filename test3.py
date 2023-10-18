import json
import time

import torch
from torch.distributed.pipeline.sync import Pipe

from llama_org2 import LLaMA, ModelArgs, Transformer

with open("./params.json", "r") as f:
    params = json.loads(f.read())

torch.set_default_tensor_type(torch.cuda.HalfTensor)

model_args: ModelArgs = ModelArgs(**params)
model_args.vocab_size = 320000

model = Transformer(model_args, 1)
time.sleep(10)
