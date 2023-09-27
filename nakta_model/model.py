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
            dim=self.head_dim, max_seq_len=512, interleaved=True
        )

        self.split_val = args.n_heads * self.head_dim

        self.xq_cache = {}
        self.xk_cache = {}
        self.xv_cache = {}

    def forward(self, x: torch.Tensor, cache_info, layer_id):
        bsz, seqlen, _ = x.shape

        xqk = self.wqk(x)
        xv = self.wv(x)

        # xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        # xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xqk = xqk.view(bsz, seqlen, 2, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        if cache_info[0] == 0:
            xq, xk = self.rotary_emb(xqk, seqlen_offset=0)
            self.xq_cache[cache_info[1]] = xq
            self.xk_cache[cache_info[1]] = xk
            self.xv_cache[cache_info[1]] = xv
        else:
            # Load the cached values and expand them along the batch dimension
            xq_cached = self.xq_cache[cache_info[1]].repeat(4, 1, 1, 1)
            xk_cached = self.xk_cache[cache_info[1]].repeat(4, 1, 1, 1)
            xv_cached = self.xv_cache[cache_info[1]].repeat(4, 1, 1, 1)

            # Compute the new rotary embeddings for xq and xk
            xq_new, xk_new = self.rotary_emb(xqk, seqlen_offset=cache_info[0])
            # print(xq_cached.shape)
            # print(xq_new.shape)
            # Concatenate the cached values with the new values
            xq = torch.cat((xq_cached, xq_new), dim=1)
            xk = torch.cat((xk_cached, xk_new), dim=1)
            xv = torch.cat((xv_cached, xv), dim=1)
            # print(xq.shape)

        # xq, xk = self.rotary_emb(xqk, seqlen_offset=)
        xq = xq.transpose(1, 2).contiguous()
        keys = xk.transpose(1, 2).contiguous()
        values = xv.transpose(1, 2).contiguous()

        output = scaled_dot_product_attention(xq, keys, values, is_causal=True)
        # if cache_info[0] != 0:
        #     print(output.shape)
        output = (
            output.transpose(1, 2).contiguous().view(bsz, seqlen + cache_info[0], -1)
        )
        if layer_id != 59:
            output = output[:, cache_info[0] :, :]
        # if cache_info[0] != 0:
        #     print(output.shape)

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
        cache_info,
        # mask: Optional[torch.Tensor],
    ):
        if cache_info[0] != 0 and self.layer_id == 59:
            x_concat = torch.cat(
                (self.cached_x[cache_info[1]].repeat(4, 1, 1), x), dim=1
            )
            h = x_concat + self.attention.forward(
                self.attention_norm(x), cache_info, self.layer_id
            )
        elif cache_info[0] == 0 and self.layer_id == 59:
            self.cached_x[cache_info[1]] = x
            h = x + self.attention.forward(
                self.attention_norm(x), cache_info, self.layer_id
            )
        else:
            h = x + self.attention.forward(
                self.attention_norm(x), cache_info, self.layer_id
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
    def forward(self, tokens: torch.Tensor, cache_info):
        """
        cache_info: Tuple(seqlen_offset: int, cache_key: int)
        """
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_math=False,
            enable_mem_efficient=False,
        ):
            _bsz, seqlen = tokens.shape
            h = self.tok_embeddings(tokens)
            for i, layer in enumerate(self.layers):
                h = layer(h, cache_info)
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
