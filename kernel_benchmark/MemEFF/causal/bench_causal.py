import math

import torch
import triton

HAS_FLASH = False

BATCH, N_HEADS, N_CTX, D_HEAD = 4, 13, 4096, 128
# vary seq length for fixed head and batch=4
CAUSAL = True
configs = [
    triton.testing.Benchmark(
        x_names=["N_CTX"],
        x_vals=[i * 16 for i in range(1, 8)],
        line_arg="provider",
        line_vals=
        # ["triton"]
        [] + (["native"]) + (["native_r"]),
        # line_vals=["triton"] + (["flash"] if HAS_FLASH else []),
        line_names=
        # ["Triton"]
        [] + (["pytorch-mem-eff"]) + (["pytorch-native"]),
        # line_names=["Triton"] + ([f"Flash-{FLASH_VER}"] if HAS_FLASH else []),
        styles=[
            # ("red", "-"),
            ("green", "-"),
            ("orange", "-"),
        ],
        ylabel="tf/s",
        plot_name=f"fused-attention-batch{BATCH}-head{N_HEADS}-d{D_HEAD}-{mode}-{'causal' if CAUSAL else 'noncausal'}",
        args={
            "H": N_HEADS,
            "BATCH": BATCH,
            "D_HEAD": D_HEAD,
            "dtype": torch.float16,
            "mode": mode,
            "causal": CAUSAL,
        },
    )
    # for mode in ["fwd", "bwd"]
    for mode in ["fwd"]
    # for causal in [False, True]
    # for causal in [True]
]

import torch.nn.functional as F


@triton.testing.perf_report(configs)
def bench_flash_attention(
    BATCH, H, N_CTX, D_HEAD, causal, mode, provider, dtype=torch.float16, device="cuda"
):
    assert mode in ["fwd", "bwd"]
    warmup = 25
    rep = 100

    if provider == "native":
        q = torch.randn(
            (BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True
        )
        k = torch.randn(
            (BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True
        )
        v = torch.randn(
            (BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True
        )
        dropout_rate = 0.0

        def fn():
            with torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_math=False,
                enable_mem_efficient=True,
            ):
                return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_rate)

        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    if provider == "native_r":
        q = torch.randn(
            (BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True
        )
        k = torch.randn(
            (BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True
        )
        v = torch.randn(
            (BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True
        )
        dropout_rate = 0.0

        def fn():
            attn_weight = torch.softmax(
                (q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))), dim=-1
            )
            # attn_weight = torch.dropout(attn_weight, dropout_rate)
            x = attn_weight @ v
            return x

        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)

    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * D_HEAD
    total_flops = 2 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    if mode == "bwd":
        total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
    return total_flops / ms * 1e-9


# only works on post-Ampere GPUs right now
bench_flash_attention.run(save_path=".", print_data=True)
