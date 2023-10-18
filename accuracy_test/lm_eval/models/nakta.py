## 우리 모델 구현

import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Optional, Union

import fire
import numpy as np
import torch
import transformers
from lm_eval.base import BaseLM

from nakta_model import LLaMA, ModelArgs, Tokenizer, Transformer


def _get_dtype(dtype: Union[str, torch.dtype]) -> torch.dtype:
    """Converts `dtype` from `str` to torch.dtype when possible. Does not use an instantiated HF AutoConfig"""
    if isinstance(dtype, str) and dtype != "auto":
        # Convert `str` args torch dtype: `float16` -> `torch.float16`
        _torch_dtype = getattr(torch, dtype)
    else:
        _torch_dtype = dtype
    return _torch_dtype


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> (Transformer, Tokenizer):
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

    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return (model, tokenizer)


class Nakta_ACC(BaseLM):
    _DEFAULT_MAX_LENGTH = 2048

    def __init__(
        self,
        ckpt_dir,
        tokenizer_path,
        local_rank,
        world_size,
        max_seq_len,
        device="cuda",
        pretrained="gpt2",
        revision="main",
        low_cpu_mem_usage=None,
        subfolder=None,
        tokenizer=None,
        batch_size=1,
        max_batch_size=512,
        max_length=None,
        load_in_8bit: Optional[bool] = False,
        trust_remote_code: Optional[bool] = False,
        dtype: Optional[Union[str, torch.dtype]] = "auto",
    ):
        super().__init__()

        self.model_name = "nakta"

        self._device = local_rank

        self.model, self.tokenizer = load(
            ckpt_dir,
            tokenizer_path,
            local_rank,
            world_size,
            max_seq_len,
            max_batch_size,
        )

        self.model.eval()

        self.vocab_size = self.tokenizer.n_words

        # Validate batch_size
        assert isinstance(batch_size, (int, str))

        # setup for automatic batch size detection
        if str(batch_size).startswith("auto"):
            batch_size = batch_size.split(":")
            self.batch_size_per_gpu = batch_size[0]
            self.batch_schedule = float(batch_size[1]) if len(batch_size) > 1 else 1
        else:
            self.batch_size_per_gpu = int(batch_size)
        self.max_batch_size = max_batch_size

        self._max_length = max_length

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        if self._max_length:  # if max length manually set, return it
            return self._max_length
        seqlen_config_attrs = (
            "n_positions",
            "max_position_embeddings",
            "n_ctx",
        )
        # for attr in seqlen_config_attrs:
        #     if hasattr(self.model.config, attr):
        #         return getattr(self.model.config, attr)
        # if hasattr(self.tokenizer, "model_max_length"):
        #     if self.tokenizer.model_max_length == 1000000000000000019884624838656:
        #         return self._DEFAULT_MAX_LENGTH
        #     return self.tokenizer.model_max_length
        return self._DEFAULT_MAX_LENGTH

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, bos=False, eos=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps, ctx_info):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            return self.model(inps, ctx_info)

    def _model_generate(self, context, max_length, eos_token_id):
        generation_kwargs = {"do_sample": False, "max_length": max_length}
        if eos_token_id is not None:
            generation_kwargs["eos_token_id"] = eos_token_id
            generation_kwargs[
                "pad_token_id"
            ] = eos_token_id  # setting eos_token_id as pad token
        return self.model.generate(context, **generation_kwargs)
