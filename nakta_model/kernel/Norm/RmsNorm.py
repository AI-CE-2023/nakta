import torch
import torch.nn as nn
import triton
import triton.language as tl

from ...profile import nvtx_annotate


@triton.jit
def _layer_norm_fwd_fused(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    # Compute mean

    # Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = tl.math.rsqrt(var + eps)
    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = x * rstd
        y = x_hat * w
        # Write output
        tl.store(Y + cols, y, mask=mask)


class RmsNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, weight, eps):
        # allocate output
        y = torch.empty_like(x)
        # reshape input data into 2D tensor
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        mean = torch.empty((M,), dtype=torch.float32, device="cuda")
        rstd = torch.empty((M,), dtype=torch.float32, device="cuda")
        # Less than 64KB per feature: enqueue fused kernel
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE:
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
        # heuristics for number of warps
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
        # enqueue kernel
        _layer_norm_fwd_fused[(M,)](
            x_arg,
            y,
            weight,
            x_arg.stride(0),
            N,
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            num_ctas=1,
        )
        ctx.save_for_backward(x, weight, mean, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.eps = eps
        return y


class RMSNorm_torch(torch.nn.Module):
    def __init__(self, dim: int, weight: torch.Tensor, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


@nvtx_annotate
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.weight = nn.Parameter(torch.ones(dim))
        self.rms_norm = RmsNormFunction.apply

    def forward(self, x):
        return self.rms_norm(x, self.dim, self.weight, self.eps)


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
