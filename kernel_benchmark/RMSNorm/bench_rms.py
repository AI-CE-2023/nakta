import torch
import torch.nn as nn
import triton

from nakta_model.kernel.Norm import RMSNorm
from nakta_model.kernel.Norm.RmsNorm import RMSNorm_torch


def test_layer_norm(M, N, dtype, eps=1e-6, device="cuda"):
    # create data
    x_shape = (10, M, N)
    w_shape = (x_shape[-1],)
    weight = torch.rand(w_shape, dtype=dtype, device="cuda", requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device="cuda")
    dy = 0.1 * torch.randn_like(x)
    x.requires_grad_(True)
    # forward pass
    rms_norm.weight = nn.Parameter(weight)
    y_tri = rms_norm(x)
    torch_rms = RMSNorm_torch(N, weight)
    y_ref = torch_rms(x).to(dtype)

    # compare
    print((y_tri - y_ref).abs().max())
    assert torch.allclose(y_tri, y_ref, atol=1e-2, rtol=0)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[64 * i for i in range(1, 10)],
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "Torch"],
        styles=[("blue", "-"), ("green", "-"), ("orange", "-")],
        ylabel="GB/s",
        plot_name="RMS-Norm-forward",
        args={"M": 4096, "dtype": torch.float16, "mode": "forward"},
    )
)
def bench_layer_norm(M, N, dtype, provider, mode="backward", eps=1e-5, device="cuda"):
    # create data
    x_shape = (10, M, N)
    w_shape = (x_shape[-1],)
    weight = torch.rand(w_shape, dtype=dtype, device="cuda", requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device="cuda")
    quantiles = [0.5, 0.2, 0.8]
    torch_rms = RMSNorm_torch(N, weight)
    rms_norm.weight = nn.Parameter(weight)
    # utility functions
    if provider == "triton":

        def y_fwd():
            return rms_norm(x)  # noqa: F811, E704

    if provider == "torch":

        def y_fwd():
            return torch_rms(x)  # noqa: F811, E704

    # forward pass
    if mode == "forward":
        gbps = lambda ms: 2 * x.numel() * x.element_size() / ms * 1e-6
        ms, min_ms, max_ms = triton.testing.do_bench(
            y_fwd, quantiles=quantiles, rep=500
        )

    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    dim = 8192
    rms_norm = RMSNorm(dim=8192)
    test_layer_norm(1151, 8192, torch.float16)
    # a = torch.randn(20,13,8192, device='cuda:0')
    # rms_norm(a)
    bench_layer_norm.run(save_path=".", print_data=True)
