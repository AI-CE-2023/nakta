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
from torch.nn.utils.rnn import pad_sequence

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


def _remove_padding(output, batch_info):
    # Flatten the sequences based on batch_info
    if type(batch_info) == int:
        return output
    flattened_output = torch.cat(
        [output[i, : batch_info[i]] for i in range(len(batch_info))], dim=0
    )
    return flattened_output


def _rebuild_padding(Q, batch_info):
    if type(batch_info) == int:
        return Q
    Q = Q.split(batch_info)
    return pad_sequence(Q, batch_first=True)


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

        self.xq_cache = {}
        self.xk_cache = {}
        self.xv_cache = {}

    def forward(self, x: torch.Tensor, cache_info, layer_id, batch_info):
        xqk = self.wqk(x)
        xv = self.wv(x)

        # xqk = _rebuild_padding(xqk, batch_info)
        # xv = _rebuild_padding(xv, batch_info)
        # ---attention start---
        bsz_x_seqlen, _ = xv.shape

        xqk = xqk.view(bsz_x_seqlen, 2, self.n_local_heads, self.head_dim)
        xq = xqk[:, 0, :, :]
        xk = xqk[:, 1, :, :]
        xv = xv.view(bsz_x_seqlen, self.n_local_heads, self.head_dim)

        if cache_info[1] == -1:
            xq, xk = self.rotary_emb(xqk, seqlen_offset=0)
        elif cache_info[0] == 0:
            # To do: rotary kernel 수정
            # xq, xk = self.rotary_emb(xqk, seqlen_offset=0)
            self.xq_cache[cache_info[1]] = xq
            self.xk_cache[cache_info[1]] = xk
            self.xv_cache[cache_info[1]] = xv
        else:
            # Load the cached values and expand them along the batch dimension
            xq_cached = self.xq_cache[cache_info[1]]
            xk_cached = self.xk_cache[cache_info[1]]
            xv_cached = self.xv_cache[cache_info[1]]

            # Compute the new rotary embeddings for xq and xk
            # xq_new, xk_new = self.rotary_emb(xqk, seqlen_offset=cache_info[0])

            # Concatenate the cached values with the new values
            xq = torch.cat((xq_cached, xq), dim=0)
            xk = torch.cat((xk_cached, xk), dim=0)
            xv = torch.cat((xv_cached, xv), dim=0)

        xq = xq.unsqueeze(dim=0).transpose(1, 2).contiguous()
        keys = xk.unsqueeze(dim=0).transpose(1, 2).contiguous()
        values = xv.unsqueeze(dim=0).transpose(1, 2).contiguous()
        if type(batch_info) == int:
            output = scaled_dot_product_attention(xq, keys, values, is_causal=True)
        else:
            output = scaled_dot_product_attention(xq, keys, values)

        output = output.transpose(1, 2).contiguous().view(output.shape[2], -1)

        if layer_id != 59 and cache_info[1] != -1:
            output = output[:bsz_x_seqlen, :]
        # ---attention end---
        if layer_id == 59:
            # output = _rebuild_padding(output, batch_info)
            output = output[:bsz_x_seqlen, :]
            return self.wo(output)

        # output = _remove_padding(output, batch_info)
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
        self.ctx_len = 0

    def forward(
        self,
        x: torch.Tensor,
        cache_info,
        batch_info
        # mask: Optional[torch.Tensor],
    ):
        if cache_info[1] == -1:
            h = x + self.attention.forward(
                self.attention_norm(x), cache_info, self.layer_id, batch_info
            )
        elif self.layer_id == 59:
            if cache_info[0] != 0:
                # cache = self.cached_x[cache_info[1]]
                # cache = cache.view(len(batch_info) // 4, self.ctx_len, 6656).repeat(
                #     cache_info[2], 1, 1
                # )
                # x_concat = torch.cat((cache, _rebuild_padding(x, batch_info)), dim=1)

                h_attn = self.attention.forward(
                    self.attention_norm(x), cache_info, self.layer_id, batch_info
                )

                h = x + h_attn

            else:
                # self.cached_x[cache_info[1]] = x
                self.ctx_len = batch_info
                h = x + self.attention.forward(
                    self.attention_norm(x), cache_info, self.layer_id, batch_info
                )
        else:
            h = x + self.attention.forward(
                self.attention_norm(x), cache_info, self.layer_id, batch_info
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
    def forward(self, tokens: torch.Tensor, cache_info, batch_info):
        """
        cache_info: Tuple(seqlen_offset: int, cache_key: int, follow_num: int)
        if cache_key == -1: -> Non Cache Mode
        """
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False,
            enable_math=False,
            enable_mem_efficient=True,
        ):
            # _bsz, seqlen = tokens.shape
            h = self.tok_embeddings(tokens)
            for i, layer in enumerate(self.layers):
                h = layer(h, cache_info, batch_info)
            if cache_info[0] == 0:
                return None
            else:
                h = self.norm(h)
                output = self.output(h)  # only compute last logits
                output = _rebuild_padding(output, batch_info)
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
    silu_and_mul = nvtx_annotate_function(silu_and_mul)
