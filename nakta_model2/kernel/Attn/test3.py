import torch
import triton
import triton.language as tl


@triton.jit
def buggy_k(
    query,
    value,
    out,
    stride_q0,
    stride_q1,
    stride_o0,
    stride_o1,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    # BLOCK_K: tl.constexpr,
):
    Q_block_ptr = tl.make_block_ptr(
        base=query,
        shape=(BLOCK_M, BLOCK_N),
        strides=(stride_q0, stride_q1),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        base=query,
        shape=(BLOCK_N, BLOCK_M),
        strides=(stride_q0, stride_q1),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_M),
        order=(0, 1),
    )

    V_block_ptr = tl.make_block_ptr(
        base=query,
        shape=(BLOCK_M, BLOCK_N),
        strides=(stride_q0, stride_q1),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        base=out,
        shape=(BLOCK_M, BLOCK_N),
        strides=(stride_o0, stride_o1),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    q = tl.load(Q_block_ptr)
    k = tl.load(K_block_ptr)
    v = tl.load(V_block_ptr)

    qk = tl.zeros([BLOCK_M, BLOCK_M], dtype=tl.float16)

    qk += tl.dot(q, k)
    qk *= sm_scale

    max = tl.max(qk)
    p = tl.exp(qk - max)
    l = tl.sum(p, 1)
    p = p / l
    out = tl.dot(p.to(tl.float16), v)

    tl.store(O_block_ptr, out.to(tl.float16))


m, n = 64, 128
c = torch.empty((m, n), device="cuda", dtype=torch.float16)

query = torch.randn(m, n, dtype=torch.float16).cuda()
key = torch.randn(m, n, dtype=torch.float16).cuda()
value = torch.randn(m, n, dtype=torch.float16).cuda()
out = torch.empty((m, n), dtype=torch.float16).cuda()

import math

sm_scale = 1 / math.sqrt(n)

buggy_k[(1,)](
    query,
    value,
    out,
    query.stride(0),
    query.stride(1),
    out.stride(0),
    out.stride(1),
    sm_scale,
    m,
    n,
)

torch_out = torch.softmax((query @ key.T) * sm_scale, dim=-1) @ value

print(out.shape)
print(torch_out.shape)
print((out - torch_out).abs().max())
