# Copyright (c) 2023, Tri Dao.

from typing import Optional, Union

import torch
import triton
import triton.language as tl


# @triton.autotune(
#     configs=[
#         triton.Config({"BLOCK_M": 2}),
#         triton.Config({"BLOCK_M": 4}),
#         triton.Config({"BLOCK_M": 8}),
#         triton.Config({"BLOCK_M": 16}),
#     ],
#     key=["CACHE_KEY_SEQLEN", "BLOCK_K", "INTERLEAVED"],
# )
@triton.jit
def rotary_kernel(
    OUT,  # Pointers to matrices
    X,
    COS,
    SIN,
    CU_SEQLENS,
    SEQLEN_OFFSETS,  # this could be int or a pointer
    # Matrix dimensions
    seqlen,
    nheads,
    rotary_dim,
    seqlen_ro,
    CACHE_KEY_SEQLEN,
    # strides
    stride_out_batch,
    stride_out_seqlen,
    stride_out_nheads,
    stride_out_headdim,
    stride_x_batch,
    stride_x_seqlen,
    stride_x_nheads,
    stride_x_headdim,
    # Meta-parameters
    BLOCK_K: tl.constexpr,
    IS_SEQLEN_OFFSETS_TENSOR: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    CONJUGATE: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_batch = tl.program_id(axis=1)
    pid_head = tl.program_id(axis=2)
    rotary_dim_half = rotary_dim // 2

    if not IS_VARLEN:
        X = X + pid_batch * stride_x_batch + pid_head * stride_x_nheads
        OUT = OUT + pid_batch * stride_out_batch + pid_head * stride_out_nheads
    else:
        start_idx = tl.load(CU_SEQLENS + pid_batch)
        seqlen = tl.load(CU_SEQLENS + pid_batch + 1) - start_idx
        X = X + start_idx * stride_x_seqlen + pid_head * stride_x_nheads
        OUT = OUT + start_idx * stride_out_seqlen + pid_head * stride_out_nheads

    if pid_m * BLOCK_M >= seqlen:
        return
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    if not IS_SEQLEN_OFFSETS_TENSOR:
        rm_cs = rm + SEQLEN_OFFSETS
    else:
        rm_cs = rm + tl.load(SEQLEN_OFFSETS + pid_batch)
    rk = tl.arange(0, BLOCK_K)
    rk_half = tl.arange(0, BLOCK_K // 2)

    if not INTERLEAVED:
        # Load the 1st and 2nd halves of X, do calculation, then store to 1st and 2nd halves of OUT
        X = X + (rm[:, None] * stride_x_seqlen + rk_half[None, :] * stride_x_headdim)
        COS = COS + (rm_cs[:, None] * rotary_dim_half + rk_half[None, :])
        SIN = SIN + (rm_cs[:, None] * rotary_dim_half + rk_half[None, :])
        cos = tl.load(
            COS,
            mask=(rm_cs[:, None] < seqlen_ro) & (rk_half[None, :] < rotary_dim_half),
            other=1.0,
        ).to(tl.float32)
        sin = tl.load(
            SIN,
            mask=(rm_cs[:, None] < seqlen_ro) & (rk_half[None, :] < rotary_dim_half),
            other=0.0,
        ).to(tl.float32)
        x0 = tl.load(
            X,
            mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half),
            other=0.0,
        ).to(tl.float32)
        x1 = tl.load(
            X + rotary_dim_half * stride_x_headdim,
            mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half),
            other=0.0,
        ).to(tl.float32)
        if CONJUGATE:
            sin = -sin
        o0 = x0 * cos - x1 * sin
        o1 = x0 * sin + x1 * cos
        # write back result
        OUT = OUT + (
            rm[:, None] * stride_out_seqlen + rk_half[None, :] * stride_out_headdim
        )
        tl.store(
            OUT, o0, mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half)
        )
        tl.store(
            OUT + rotary_dim_half * stride_out_headdim,
            o1,
            mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half),
        )
    else:
        # We don't want to load X[0, 2, 4, ...] and X[1, 3, 5, ...] separately since both are slow.
        # Instead, we load x0 = X[0, 1, 2, 3, ...] and x1 = X[1, 0, 3, 2, ...].
        # Loading x0 will be fast but x1 will be slow.
        # Then we load cos = COS[0, 0, 1, 1, ...] and sin = SIN[0, 0, 1, 1, ...].
        # Then we do the calculation and use tl.where to pick put the right outputs for the even
        # and for the odd indices.
        rk_swap = rk + ((rk + 1) % 2) * 2 - 1  # 1, 0, 3, 2, 5, 4, ...
        rk_repeat = tl.arange(0, BLOCK_K) // 2
        X0 = X + (rm[:, None] * stride_x_seqlen + rk[None, :] * stride_x_headdim)
        X1 = X + (rm[:, None] * stride_x_seqlen + rk_swap[None, :] * stride_x_headdim)
        COS = COS + (rm_cs[:, None] * rotary_dim_half + rk_repeat[None, :])
        SIN = SIN + (rm_cs[:, None] * rotary_dim_half + rk_repeat[None, :])
        cos = tl.load(
            COS,
            mask=(rm_cs[:, None] < seqlen_ro) & (rk_repeat[None, :] < rotary_dim_half),
            other=1.0,
        ).to(tl.float32)
        sin = tl.load(
            SIN,
            mask=(rm_cs[:, None] < seqlen_ro) & (rk_repeat[None, :] < rotary_dim_half),
            other=0.0,
        ).to(tl.float32)
        x0 = tl.load(
            X0, mask=(rm[:, None] < seqlen) & (rk[None, :] < rotary_dim), other=0.0
        ).to(tl.float32)
        x1 = tl.load(
            X1, mask=(rm[:, None] < seqlen) & (rk_swap[None, :] < rotary_dim), other=0.0
        ).to(tl.float32)
        if CONJUGATE:
            sin = -sin
        x0_cos = x0 * cos
        x1_sin = x1 * sin
        out = tl.where(rk[None, :] % 2 == 0, x0_cos - x1_sin, x0_cos + x1_sin)
        OUT = OUT + (rm[:, None] * stride_out_seqlen + rk[None, :] * stride_out_headdim)
        tl.store(OUT, out, mask=(rm[:, None] < seqlen) & (rk[None, :] < rotary_dim))


def apply_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    interleaved=False,
    inplace=False,
    conjugate=False,
) -> torch.Tensor:
    """
    Arguments:
        x: (batch, seqlen, nheads, headdim) if cu_seqlens is None
            else (total_seqlen, nheads, headdim).
        cos: (seqlen_ro, rotary_dim / 2)
        sin: (seqlen_ro, rotary_dim / 2)
        seqlen_offsets: integer or integer tensor of size (batch,)
        cu_seqlens: (batch + 1,) or None
        max_seqlen: int
    Returns:
        y: (batch, seqlen, nheads, headdim)
    """
    is_varlen = cu_seqlens is not None
    if not is_varlen:
        batch, seqlen, nheads, headdim = x.shape
    else:
        assert (
            max_seqlen is not None
        ), "If cu_seqlens is passed in, then max_seqlen must be passed"
        total_seqlen, nheads, headdim = x.shape
        batch_p_1 = cu_seqlens.shape[0]
        batch = batch_p_1 - 1
        seqlen = max_seqlen
    seqlen_ro, rotary_dim = cos.shape
    assert sin.shape == cos.shape
    rotary_dim *= 2
    assert rotary_dim <= headdim, "rotary_dim must be <= headdim"
    assert headdim <= 256, "Only support headdim <= 256"
    assert seqlen_ro >= seqlen, "seqlen_ro must be >= seqlen"

    assert (
        cos.dtype == sin.dtype
    ), f"cos and sin must have the same dtype, got {cos.dtype} and {sin.dtype}"
    assert (
        x.dtype == cos.dtype
    ), f"Input and cos/sin must have the same dtype, got {x.dtype} and {cos.dtype}"

    cos, sin = cos.contiguous(), sin.contiguous()
    if isinstance(seqlen_offsets, torch.Tensor):
        assert seqlen_offsets.shape == (batch,)
        assert seqlen_offsets.dtype in [torch.int32, torch.int64]
        seqlen_offsets = seqlen_offsets.contiguous()
    else:
        assert seqlen_offsets + seqlen <= seqlen_ro

    output = torch.empty_like(x) if not inplace else x
    if rotary_dim < headdim and not inplace:
        output[..., rotary_dim:].copy_(x[..., rotary_dim:])

    BLOCK_K = (
        32
        if rotary_dim <= 32
        else (64 if rotary_dim <= 64 else (128 if rotary_dim <= 128 else 256))
    )
    grid = lambda META: (triton.cdiv(seqlen, META["BLOCK_M"]), batch, nheads)  # noqa
    BLOCK_M = 4 if interleaved else (8 if rotary_dim <= 64 else 4)

    # Need this, otherwise Triton tries to launch from cuda:0 and we get
    # ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)
    with torch.cuda.device(x.device.index):
        rotary_kernel[grid](
            output,  # data ptrs
            x,
            cos,
            sin,
            cu_seqlens,
            seqlen_offsets,
            seqlen,  # shapes
            nheads,
            rotary_dim,
            seqlen_ro,
            seqlen // 128,  # key for triton cache (limit number of compilations)
            output.stride(0)
            if not is_varlen
            else 0,  # batch_strides if not varlen else 0
            output.stride(-3),  # seqlen_stride or total_seqlen_stride
            output.stride(-2),  # nheads_stride
            output.stride(-1),  # headdim_stride
            x.stride(0) if not is_varlen else 0,  # batch_strides if not varlen else 0
            x.stride(-3),  # seqlen stride or total_seqlen_stride
            x.stride(-2),  # nheads stride
            x.stride(-1),  # headdim stride
            BLOCK_K,
            isinstance(seqlen_offsets, torch.Tensor),
            is_varlen,
            interleaved,
            conjugate,
            BLOCK_M,
        )
    return output


# Copyright (c) 2023, Tri Dao.

import math
from typing import Optional, Tuple, Union

import torch
from einops import rearrange, repeat


class ApplyRotaryEmbQKV_(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qk,
        cos,
        sin,
        cos_k=None,
        sin_k=None,
        interleaved=False,
        seqlen_offsets: Union[int, torch.Tensor] = 0,
    ):
        batch, seqlen, two, nheads, headdim = qk.shape
        assert two == 2
        if cos_k is None and sin_k is None and qk.is_contiguous():
            # Call 1 kernel instead of 2 kernels
            # We need qkv to be contiguous so that when we reshape to combine (3, nheads)
            # dimensions, we get the same tensor
            # qk = rearrange(qkv[:, :, :2], "b s t h d -> b s (t h) d")
            qk = qk.reshape(batch, seqlen, -1, headdim)
            apply_rotary(
                qk,
                cos,
                sin,
                seqlen_offsets=seqlen_offsets,
                interleaved=interleaved,
                inplace=True,
            )
        return qk.split(nheads, dim=2)


def apply_rotary_emb_qkv_(
    qkv,
    cos,
    sin,
    cos_k=None,
    sin_k=None,
    interleaved=False,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
):
    """
    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, headdim)
        cos, sin: (seqlen, rotary_dim / 2)
        cos_k, sin_k: (seqlen, rotary_dim / 2), optional
        interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead of
            1st half and 2nd half (GPT-NeoX style).
        seqlen_offsets: (batch_size,) or int. Each sequence in Q and K is shifted by this amount.
            Most commonly used in inference when we have KV cache.
    Return:
        qkv: (batch_size, seqlen, 3, nheads, headdim)
    rotary_dim must be <= headdim
    Apply rotary embedding *inplace* to the first rotary_dim of Q and K.
    """
    return ApplyRotaryEmbQKV_.apply(
        qkv, cos, sin, cos_k, sin_k, interleaved, seqlen_offsets
    )


class RotaryEmbedding(torch.nn.Module):
    """
    The rotary position embeddings from RoFormer_ (Su et. al).
    A crucial insight from the method is that the query and keys are
    transformed by rotation matrices which depend on the relative positions.

    Other implementations are available in the Rotary Transformer repo_ and in
    GPT-NeoX_, GPT-NeoX was an inspiration

    .. _RoFormer: https://arxiv.org/abs/2104.09864
    .. _repo: https://github.com/ZhuiyiTechnology/roformer
    .. _GPT-NeoX: https://github.com/EleutherAI/gpt-neox

    If scale_base is not None, this implements XPos (Sun et al., https://arxiv.org/abs/2212.10554).
    A recommended value for scale_base is 512: https://github.com/HazyResearch/flash-attention/issues/96
    Reference: https://github.com/sunyt32/torchscale/blob/main/torchscale/component/xpos_relative_position.py
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int,
        device=None,
        base=10000.0,
        interleaved=False,
        scale_base=None,
        pos_idx_in_fp32=True,
    ):
        """
        interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
            of 1st half and 2nd half (GPT-NeoX style).
        pos_idx_in_fp32: if True, the position indices [0.0, ..., seqlen - 1] are in fp32,
            otherwise they might be in lower precision.
            This option was added because previously (before 2023-07-02), when we construct
            the position indices, we use the dtype of self.inv_freq. In most cases this would
            be fp32, but if the model is trained in pure bf16 (not mixed precision), then
            self.inv_freq would be bf16, and the position indices are also in bf16.
            Because of the limited precision of bf16 (e.g. 1995.0 is rounded to 2000.0), the
            embeddings for some positions will coincide.
            To maintain compatibility with models previously trained in pure bf16,
            we add this option.
        """
        super().__init__()
        self.dim = dim
        self.base = float(base)
        self.pos_idx_in_fp32 = pos_idx_in_fp32
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = self._compute_inv_freq(device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.interleaved = interleaved
        self.scale_base = scale_base
        scale = (
            (torch.arange(0, dim, 2, device=device, dtype=torch.float32) + 0.4 * dim)
            / (1.4 * dim)
            if scale_base is not None
            else None
        )
        self.register_buffer("scale", scale, persistent=False)

        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cos_k_cached = None
        self._sin_k_cached = None

        self._update_cos_sin_cache(max_seq_len, device=device, dtype=torch.float16)

    def _compute_inv_freq(self, device=None):
        return 1.0 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, device=device, dtype=torch.float32)
                / self.dim
            )
        )

    def _update_cos_sin_cache(self, seqlen, device=None, dtype=None):
        # Reset the tables if the sequence length has changed,
        # if we're on a new device (possibly due to tracing for instance),
        # or if we're switching from inference mode to training
        if (
            seqlen > self._seq_len_cached
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
            or (self.training and self._cos_cached.is_inference())
        ):
            self._seq_len_cached = seqlen
            # We want fp32 here, not self.inv_freq.dtype, since the model could be loaded in bf16
            # And the output of arange can be quite large, so bf16 would lose a lot of precision.
            # However, for compatibility reason, we add an option to use the dtype of self.inv_freq.
            if self.pos_idx_in_fp32:
                t = torch.arange(seqlen, device=device, dtype=torch.float32)
                # We want fp32 here as well since inv_freq will be multiplied with t, and the output
                # will be large. Having it in bf16 will lose a lot of precision and cause the
                # cos & sin output to change significantly.
                # We want to recompute self.inv_freq if it was not loaded in fp32
                if self.inv_freq.dtype != torch.float32:
                    inv_freq = self._compute_inv_freq(device=device)
                else:
                    inv_freq = self.inv_freq
            else:
                t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
                inv_freq = self.inv_freq
            # Don't do einsum, it converts fp32 to fp16 under AMP
            # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            freqs = torch.outer(t, inv_freq)
            if self.scale is None:
                self._cos_cached = torch.cos(freqs).to(dtype)
                self._sin_cached = torch.sin(freqs).to(dtype)
            else:
                power = (
                    torch.arange(
                        seqlen, dtype=self.scale.dtype, device=self.scale.device
                    )
                    - seqlen // 2
                ) / self.scale_base
                scale = self.scale.to(device=power.device) ** rearrange(
                    power, "s -> s 1"
                )
                # We want the multiplication by scale to happen in fp32
                self._cos_cached = (torch.cos(freqs) * scale).to(dtype)
                self._sin_cached = (torch.sin(freqs) * scale).to(dtype)
                self._cos_k_cached = (torch.cos(freqs) / scale).to(dtype)
                self._sin_k_cached = (torch.sin(freqs) / scale).to(dtype)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seqlen_offset: Union[int, torch.Tensor] = 0,
        max_seqlen: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        qkv: (batch, seqlen, 3, nheads, headdim) if kv is none,
             else it's just q of shape (batch, seqlen, nheads, headdim)
        kv: (batch, seqlen, 2, nheads, headdim)
        seqlen_offset: (batch_size,) or int. Each sequence in x is shifted by this amount.
            Most commonly used in inference when we have KV cache.
            If it's a tensor of shape (batch_size,), then to update the cos / sin cache, one
            should pass in max_seqlen, which will update the cos / sin cache up to that length.
        Apply rotary embedding *inplace* to qkv and / or kv.
        """
        # seqlen = q.shape[1]
        qk = torch.stack([q, k], dim=2)
        # if max_seqlen is not None:
        #     self._update_cos_sin_cache(max_seqlen, device=q.device, dtype=q.dtype)
        # elif isinstance(seqlen_offset, int):
        #     self._update_cos_sin_cache(
        #         seqlen + seqlen_offset, device=q.device, dtype=q.dtype
        #     )
        if self.scale is None:
            return apply_rotary_emb_qkv_(
                qk,
                self._cos_cached,
                self._sin_cached,
                interleaved=self.interleaved,
                seqlen_offsets=seqlen_offset,
            )
        # else:
        #     return apply_rotary_emb_qkv_(
        #         qkv,
        #         self._cos_cached,
        #         self._sin_cached,
        #         self._cos_k_cached,
        #         self._sin_k_cached,
        #         interleaved=self.interleaved,
        #         seqlen_offsets=seqlen_offset,
        #     )

        from typing import Optional, Tuple


import torch
import triton


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


class RotaryEmbeddingTorch(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        max_seq_len: int,
    ):
        super().__init__()
        self.dim = dim
        self.freqs = self.precompute_freqs(self.dim)
        self.freq_cis = self.precompute_cis(max_seq_len)

    def precompute_freqs(self, dim: int, theta: float = 10000.0):
        return (
            1.0
            / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)).cuda()
        )

    def precompute_cis(self, end: int):
        t = torch.arange(end, device=self.freqs.device)  # type: ignore
        freqs = torch.outer(t, self.freqs).float()  # type: ignore
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
        return freqs_cis

    def apply_rotary_emb(
        self,
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

    def forward(self, query, key):
        return self.apply_rotary_emb(query, key, self.freq_cis)


def test_rotary_embedding():
    dim = 128
    end = 60

    rotary_emb_torch = RotaryEmbeddingTorch(dim=dim, max_seq_len=end)

    query = torch.rand(64, end, 13, dim, dtype=torch.float16).cuda()
    key = torch.rand(64, end, 13, dim, dtype=torch.float16).cuda()
    value = torch.rand(64, end, 13, dim, dtype=torch.float16).cuda()

    q_out, k_out = rotary_emb_torch(query, key)

    rotary_emb = RotaryEmbedding(
        dim=dim, max_seq_len=end, interleaved=True, device=query.device
    )
    q, k = rotary_emb(query, key)

    # assert torch.allclose(q_out, q, atol=1e-2)
    # assert torch.allclose(k_out, k, atol=1e-2)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["seq_len"],
        x_vals=[64 * i for i in range(1, 10)],
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "Torch"],
        styles=[("blue", "-"), ("green", "-"), ("orange", "-")],
        ylabel="GB/s",
        plot_name="rotary-emb-forward",
        args={"hidden": 128, "dtype": torch.float16, "mode": "forward"},
    )
)
def bench_layer_norm(
    hidden, seq_len, dtype, provider, mode="backward", eps=1e-5, device="cuda"
):
    # create data
    x_shape = (64, seq_len, 13, hidden)
    query = torch.randn(x_shape, dtype=dtype, device="cuda")
    key = torch.randn(x_shape, dtype=dtype, device="cuda")

    quantiles = [0.5, 0.2, 0.8]
    torch_rotary = RotaryEmbeddingTorch(dim=hidden, max_seq_len=seq_len)
    triton_rotary = RotaryEmbedding(
        dim=hidden, max_seq_len=seq_len, interleaved=True, device=query.device
    )
    # utility functions
    if provider == "triton":

        def y_fwd():
            return triton_rotary(query, key)  # noqa: F811, E704

    if provider == "torch":

        def y_fwd():
            return torch_rotary(query, key)  # noqa: F811, E704

    # forward pass
    if mode == "forward":
        gbps = lambda ms: 4 * query.numel() * query.element_size() / ms * 1e-6
        ms, min_ms, max_ms = triton.testing.do_bench(
            y_fwd, quantiles=quantiles, rep=500
        )

    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    test_rotary_embedding()
    # bench_layer_norm.run(save_path=".", print_data=True)
