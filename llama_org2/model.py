# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

import math
from dataclasses import dataclass
from typing import Optional, Tuple

# import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn.functional as F

# from fairscale.nn.model_parallel.layers import (
#     ColumnParallelLinear,
#     ParallelEmbedding,
#     RowParallelLinear,
# )
from torch import nn

from .nvtx_annotation import nvtx_annotate, nvtx_annotate_function


@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5

    max_batch_size: int = 64
    max_seq_len: int = 256

    pipeline_size: int = 4


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def attention_native(xq, keys, values):
    # scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(head_dim)
    # if mask is not None:
    #     scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
    # scores = F.softmax(scores.float(), dim=-1).type_as(xq)
    # return torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
    return F.scaled_dot_product_attention(xq, keys, values)


def SwiGLU(x_1, x_2):
    return F.silu(x_1) * x_2


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        # self.tp_size = args.tp_size
        self.n_local_heads = args.n_heads

        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            # gather_output=False,
            # init_method=lambda x: x,
        )
        self.wk = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            # gather_output=False,
            # init_method=lambda x: x,
        )
        self.wv = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            # gather_output=False,
            # init_method=lambda x: x,
        )
        self.wo = nn.Linear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            # input_is_parallel=True,
            # init_method=lambda x: x,
        )

        # self.cache_k = torch.zeros(
        #     (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)
        # ).cuda()
        # self.cache_v = torch.zeros(
        #     (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)
        # ).cuda()

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        # mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # self.cache_k = self.cache_k.to(xq)
        # self.cache_v = self.cache_v.to(xq)

        # self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        # self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        # keys = self.cache_k[:bsz, : start_pos + seqlen]
        # values = self.cache_v[:bsz, : start_pos + seqlen]

        xq = xq.transpose(1, 2)
        keys = xk.transpose(1, 2)
        values = xv.transpose(1, 2)

        output = attention_native(xq, keys, values)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        return self.wo(output)


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

        self.w1 = nn.Linear(
            dim,
            hidden_dim,
            bias=False,
            # gather_output=False, init_method=lambda x: x
        )
        self.w2 = nn.Linear(
            hidden_dim,
            dim,
            bias=False,
            # input_is_parallel=True, init_method=lambda x: x
        )
        self.w3 = nn.Linear(
            dim,
            hidden_dim,
            bias=False,
            # gather_output=False, init_method=lambda x: x
        )

    def forward(self, x):
        return self.w2(SwiGLU(self.w1(x), self.w3(x)))


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
        freqs_cis: torch.Tensor,
        # mask: Optional[torch.Tensor],
    ):
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs, gpu_num):
        super().__init__()

        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        if gpu_num == 0:
            self.tok_embeddings = nn.Embedding(
                params.vocab_size,
                params.dim,
                # init_method=lambda x: x
            )

        self.layers = torch.nn.ModuleList()
        layer_vol = params.n_layers // params.pipeline_size

        layer_index = [(0, 30), (30, 60)]
        for layer_id in range(
            # layer_vol * gpu_num,
            # layer_vol * (gpu_num + 1),
            layer_index[gpu_num][0],
            layer_index[gpu_num][1],
        ):
            self.layers.append(TransformerBlock(layer_id, params))

        if gpu_num == 1:
            self.norm = RMSNorm(params.dim, eps=params.norm_eps)
            self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

    @torch.inference_mode()
    def forward(self, h: torch.Tensor, start_pos: int, gpu_num: int):
        """
        h => token or hidden
        """
        if gpu_num == 0:
            _bsz, seqlen = h.shape
            h = self.tok_embeddings(h)
        else:
            _bsz, seqlen, _ = h.shape

        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis)

        if gpu_num == 1:
            h = self.norm(h)
            output = self.output(h)  # only compute last logits
            return output
        else:
            return (h, start_pos, gpu_num + 1)


import os

if os.getenv("CUDA_LAUNCH_BLOCKING"):
    # ColumnParallelLinear = nvtx_annotate(ColumnParallelLinear)
    # ParallelEmbedding = nvtx_annotate(ParallelEmbedding)
    # RowParallelLinear = nvtx_annotate(RowParallelLinear)

    Attention = nvtx_annotate(Attention)
    RMSNorm = nvtx_annotate(RMSNorm)
    FeedForward = nvtx_annotate(FeedForward)
    TransformerBlock = nvtx_annotate(TransformerBlock)
    Transformer = nvtx_annotate(Transformer)

    reshape_for_broadcast = nvtx_annotate_function(reshape_for_broadcast)
    apply_rotary_emb = nvtx_annotate_function(apply_rotary_emb)
    attention_native = nvtx_annotate_function(attention_native)
    SwiGLU = nvtx_annotate_function(SwiGLU)
# precompute_freqs_cis = nvtx_annotate_function(precompute_freqs_cis)
