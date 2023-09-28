import torch
import triton
import triton.language as tl


@triton.jit
def attention(
    q_ptr,
    k_ptr,
    o_ptr,
    qa,
    qb,
    ka,
    kb,
    oa,
    ob,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    Q_block_ptr = tl.make_block_ptr(
        q_ptr,
        shape=(BLOCK_M, BLOCK_DMODEL),
        strides=(qa, qb),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        k_ptr,
        shape=(BLOCK_DMODEL, BLOCK_M),
        strides=(ka, kb),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_M),
        order=(0, 1),
    )

    O_block_ptr = tl.make_block_ptr(
        o_ptr,
        shape=(BLOCK_M, BLOCK_M),
        strides=(oa, ob),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_M),
        order=(1, 0),
    )

    query = tl.load(Q_block_ptr)
    key = tl.load(K_block_ptr)

    tl.store(O_block_ptr, tl.dot(query, key).to(tl.float16))


m, d = 32, 64
query = torch.rand(m, d, dtype=torch.float16, device="cuda")
key = torch.rand_like(query, device="cuda")
out = torch.zeros((m, m), dtype=torch.float16, device="cuda")
attention[(1,)](
    query,
    key,
    out,
    query.stride(0),
    query.stride(1),
    key.stride(0),
    key.stride(1),
    out.stride(0),
    out.stride(1),
    BLOCK_M=m,
    BLOCK_N=m,
    BLOCK_DMODEL=d,
)
torch_out = torch.matmul(query, key.transpose(0, 1))
print(torch_out)
print((torch_out - out).abs().max())
