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
from xformers.ops import memory_efficient_attention

from .kernel.Emb import RotaryEmbedding
from .kernel.Norm import RMSNorm
from .profile import nvtx_annotate, nvtx_annotate_function


@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5

    max_seq_len: int = 512
    max_batch_size: int = 512


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_local_heads = args.n_heads // fs_init.get_model_parallel_world_size()
        self.head_dim = args.dim // args.n_heads

        self.wqk = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim * 2,
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
            dim=self.head_dim, max_seq_len=args.max_seq_len, interleaved=True
        )

        self.split_val = args.n_heads * self.head_dim

    def forward(self, x: torch.Tensor, mask, offsts, layer_id):
        # input shape : [seqlen, hidden]
        bsz_x_seqlen, seqlen = x.shape

        xqk = self.wqk(x)
        xv = self.wv(x)

        # rotary input [seqlen, 1, 2, localheads, head_dim]
        xqk = xqk.view(bsz_x_seqlen, 1, 2, self.n_local_heads, self.head_dim)
        # value 는 rotary 통과할 필요 없음
        # value shape [1, seqlen, localhead, head_dim]
        xv = xv.view(1, bsz_x_seqlen, self.n_local_heads, self.head_dim)

        # rotary output [seqlen, 1, localheads, head_dim]
        xq, xk = self.rotary_emb(xqk, seqlen_offset=offsts)

        # attention input [1, seqlen, localheads, head_dim]
        xq = xq.transpose(0, 1).contiguous()
        keys = xk.transpose(0, 1).contiguous()
        # values = xv.transpose(1, 2).contiguous()

        # output [1, localheads, seqlen, head_dim]
        # output = scaled_dot_product_attention(xq, keys, values, attn_mask=mask)
        output = memory_efficient_attention(xq, keys, xv, attn_bias=mask)

        # output [seqlen, hidden]
        output = output.view(bsz_x_seqlen, -1)

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
        self.hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(
            dim,
            self.hidden_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.w2 = RowParallelLinear(
            self.hidden_dim,
            dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )
        self.w3 = ColumnParallelLinear(
            dim,
            self.hidden_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )

    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w3(x)
        out = torch.zeros_like(x1)

        return self.w2(silu_and_mul(out, x1, x2))


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

        self.cached_x = {}

    def forward(
        self,
        x: torch.Tensor,
        mask,
        offsets
        # mask: Optional[torch.Tensor],
    ):
        h = x + self.attention.forward(
            self.attention_norm(x), mask, offsets, self.layer_id
        )
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


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
    def forward(self, tokens: torch.Tensor, mask, offsets):
        """
        cache_info: Tuple(seqlen_offset: int, cache_key: int, follow_num: int)
        if cache_key == -1: -> Non Cache Mode
        """
        # with torch.backends.cuda.sdp_kernel(
        #     enable_flash=False,
        #     enable_math=False,
        #     enable_mem_efficient=True,
        # ):
        h = self.tok_embeddings(tokens)
        for i, layer in enumerate(self.layers):
            h = layer(h, mask, offsets)
        h = self.norm(h)
        output = self.output(h)  # only compute last logits
        return output.float()


import os

if os.getenv("CUDA_LAUNCH_BLOCKING"):
    ColumnParallelLinear = nvtx_annotate(ColumnParallelLinear)
    ParallelEmbedding = nvtx_annotate(ParallelEmbedding)
    RowParallelLinear = nvtx_annotate(RowParallelLinear)
    nn.Embedding = nvtx_annotate(nn.Embedding)
    RMSNorm = nvtx_annotate(RMSNorm)
    RotaryEmbedding = nvtx_annotate(RotaryEmbedding)

    Attention = nvtx_annotate(Attention)
    FeedForward = nvtx_annotate(FeedForward)
    TransformerBlock = nvtx_annotate(TransformerBlock)
    Transformer = nvtx_annotate(Transformer)

    scaled_dot_product_attention = nvtx_annotate_function(scaled_dot_product_attention)
    silu_and_mul = nvtx_annotate_function(silu_and_mul)
    silu_and_mul = nvtx_annotate_function(silu_and_mul)
    memory_efficient_attention = nvtx_annotate_function(memory_efficient_attention)
