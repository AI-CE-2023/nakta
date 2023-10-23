from typing import Optional, Tuple

import torch
import triton

from nakta_model.kernel.Emb import RotaryEmbedding


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
    dim = 32
    end = 8

    rotary_emb_torch = RotaryEmbeddingTorch(dim=dim, max_seq_len=end)

    query = torch.rand(1, end, 10, dim, dtype=torch.float16).cuda()
    key = torch.rand(1, end, 10, dim, dtype=torch.float16).cuda()
    value = torch.rand(1, end, 10, dim, dtype=torch.float16).cuda()
    qk = torch.cat((query.unsqueeze(2), key.unsqueeze(2)), dim=2)
    q_out, k_out = rotary_emb_torch(query, key)

    rotary_emb = RotaryEmbedding(
        dim=dim, max_seq_len=end, interleaved=True, device=query.device
    )
    q, k = rotary_emb(qk)

    assert torch.allclose(q_out, q, atol=1e-2)
    assert torch.allclose(k_out, k, atol=1e-2)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size"],
        x_vals=[128 * i for i in range(70, 75)],
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "Torch"],
        styles=[("blue", "-"), ("green", "-"), ("orange", "-")],
        ylabel="GB/s",
        plot_name="rotary-emb-forward",
        args={"hidden": 128, "seq_len": 1, "dtype": torch.float16, "mode": "forward"},
    )
)
def bench_layer_norm(
    batch_size,
    hidden,
    seq_len,
    dtype,
    provider,
    mode="backward",
    eps=1e-5,
    device="cuda",
):
    # create data
    x_shape = (batch_size, seq_len, 13, hidden)
    query = torch.randn(x_shape, dtype=dtype, device="cuda")
    key = torch.randn(x_shape, dtype=dtype, device="cuda")
    qk = torch.cat((query.unsqueeze(2), key.unsqueeze(2)), dim=2)

    quantiles = [0.5, 0.2, 0.8]
    torch_rotary = RotaryEmbeddingTorch(dim=hidden, max_seq_len=seq_len)
    triton_rotary = RotaryEmbedding(
        dim=hidden, max_seq_len=seq_len, interleaved=True, device=query.device
    )
    # utility functions
    if provider == "triton":

        def y_fwd():
            return triton_rotary(qk)  # noqa: F811, E704

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
    bench_layer_norm.run(save_path=".", print_data=True)
