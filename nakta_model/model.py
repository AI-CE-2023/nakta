# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from nakta_attn import silu_and_mul
from torch import nn
from torch.nn.functional import scaled_dot_product_attention

from .kernel.Emb import RotaryEmbedding
from .kernel.Norm import RMSNorm
from .profile import nvtx_annotate, nvtx_annotate_function

ColumnParallelLinear = nvtx_annotate(ColumnParallelLinear)
ParallelEmbedding = nvtx_annotate(ParallelEmbedding)
RowParallelLinear = nvtx_annotate(RowParallelLinear)
nn.Embedding = nvtx_annotate(nn.Embedding)
RotaryEmbedding = nvtx_annotate(RotaryEmbedding)

scaled_dot_product_attention = nvtx_annotate_function(scaled_dot_product_attention)
silu_and_mul = nvtx_annotate_function(silu_and_mul)


@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048


# triton fused layer norm 을 implement 할 필요 있을 듯
# https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html

# @nvtx_annotate_function
# def SwiGLU(x_1, x_2):
#     return F.silu(x_1) * x_2


@nvtx_annotate
class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_local_heads = args.n_heads // fs_init.get_model_parallel_world_size()
        self.head_dim = args.dim // args.n_heads

        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wv = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )
        self.rotary_emb = RotaryEmbedding(
            dim=self.head_dim, max_seq_len=512, interleaved=True
        )

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
    ):
        bsz, seqlen, _ = x.shape

        # weight 곱과 view 를 한번에 하는 커널
        # 직접구현: triton
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xq, xk = self.rotary_emb(xq, xk)
        xq = xq.transpose(1, 2)
        keys = xk.transpose(1, 2)
        values = xv.transpose(1, 2)

        output = scaled_dot_product_attention(xq, keys, values, is_causal=True)

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # native 구현
        return self.wo(output)


@nvtx_annotate
class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )
        self.w2 = RowParallelLinear(
            hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        )
        self.w3 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )

    def forward(self, x):
        # fused mlp
        # xformer triton을 implemet 하면 될 듯
        # https://github.com/facebookresearch/xformers/blob/cfad52d18f80170e6446ec810b13e526a0e5a46d/xformers/triton/k_fused_matmul_fw.py
        # silu 관련 triton issue
        # https://github.com/openai/triton/issues/1836
        x1 = self.w1(x)
        x2 = self.w3(x)
        out = torch.zeros_like(x1)

        return self.w2(silu_and_mul(out, x1, x2))


@nvtx_annotate
class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        # mask: Optional[torch.Tensor],
    ):
        h = x + self.attention.forward(self.attention_norm(x), start_pos)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


@nvtx_annotate
class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)

        # self.tok_embeddings = ParallelEmbedding(
        #     params.vocab_size, params.dim, init_method=lambda x: x
        # )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = ColumnParallelLinear(
            params.dim, params.vocab_size, bias=False, init_method=lambda x: x
        )

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        for i, layer in enumerate(self.layers):
            h = layer(h, start_pos)
        h = self.norm(h)
        output = self.output(h)  # only compute last logits
        return output.float()
