import matplotlib as mpl
import matplotlib.pyplot as plt
import nakta_attn
import numpy as np
import pandas as pd
import torch

font = {
    "family": "serif",
    # 'weight' : 'bold',
    "size": 18,
}

mpl.rc("font", **font)


def silu_and_mul_cuda(
    len_seq: int,
    hidden_dim: int = 1280,
    dtype: torch.dtype = torch.float16,
):
    x1 = torch.randn(
        (len_seq, hidden_dim),
        dtype=dtype,
        device="cuda",
    )
    x2 = torch.randn(
        (len_seq, hidden_dim),
        dtype=dtype,
        device="cuda",
    )
    out = torch.zeros_like(x1)
    fn = lambda: nakta_attn.silu_and_mul(out, x1, x2)

    r = nakta_attn.bench.do_bench(fn)
    r.update(
        {
            "length": len_seq,
            "factor": 3 * x1.nelement() * x1.element_size() * 1e-09,
        }
    )

    return r


def silu_and_mul_torch(
    len_seq: int,
    hidden_dim: int = 1280,
    dtype: torch.dtype = torch.float16,
):
    x1 = torch.randn(
        (len_seq, hidden_dim),
        dtype=dtype,
        device="cuda",
    )
    x2 = torch.randn(
        (len_seq, hidden_dim),
        dtype=dtype,
        device="cuda",
    )
    fn = lambda: torch.nn.functional.silu(x1) * x2

    r = nakta_attn.bench.do_bench(fn)
    r.update(
        {
            "length": len_seq,
            "factor": 3 * x1.nelement() * x1.element_size() * 1e-09,
        }
    )

    return r


def save_to_csv(df_cuda, df_torch, filename="benchmark_results.csv"):
    # Extract relevant columns and rename for the desired CSV format
    df_nakta = df_cuda[["length", "mean"]].rename(columns={"mean": "Nakta"})
    df_torch_val = df_torch[["length", "mean"]].rename(columns={"mean": "Torch"})

    # Merge the two dataframes on 'length'
    merged = pd.merge(df_nakta, df_torch_val, on="length", how="outer")

    # Save to CSV
    merged.to_csv(filename, index=False)


# In the main function, after the benchmarks:


def main():
    hidden_dim = 1280
    x_vals = [int(2**i) for i in np.arange(8, 20, 1.0)]
    dtype = torch.float16

    print(">>> Benchmark fused kernels")
    rec_cuda = [
        silu_and_mul_cuda(x, hidden_dim=hidden_dim, dtype=dtype) for x in x_vals
    ]
    df_cuda = pd.DataFrame.from_records(rec_cuda, exclude=["median"])
    print(df_cuda)

    print(">>> Benchmark vanilla")
    rec_torch = [
        silu_and_mul_torch(x, hidden_dim=hidden_dim, dtype=dtype) for x in x_vals
    ]
    df_torch = pd.DataFrame.from_records(rec_torch, exclude=["median"])
    print(df_torch)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot()

    ax.set_title("Fused SiLU-and-MUL Kernel ($hidden=1280$)")
    for label, df in [("Fused", df_cuda), ("Torch", df_torch)]:
        x = df["length"]
        factor = df["factor"]
        factor *= 1e03  # ms -> sec

        y = factor / df["mean"]
        lower = factor / df["max"]
        upper = factor / df["min"]

        ax.plot(x, y, label=label)
        ax.fill_between(x, lower, upper, alpha=0.15)

    ax.set_xscale("log")
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Effective bandwidth [GB/s]")
    ax.grid(which="both")
    fig.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig("./fused-silu-and-mul.png")
    fig.show()

    save_to_csv(df_cuda, df_torch)


if __name__ == "__main__":
    main()

"""
CUDA_LAUNCH_BLOCKING=1 python bench_swiglu.py
"""
