import torch
import torch.nn as nn

from nakta_model.kernel.Norm.RmsNorm import RMSNorm


def test_layer_norm(M, N, dtype, eps=1e-6, device="cuda"):
    # create data
    x_shape = (10, M, N)
    w_shape = (x_shape[-1],)
    weight = torch.rand(w_shape, dtype=dtype, device="cuda", requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device="cuda")
    # forward pass
    rms_norm.weight = nn.Parameter(weight)
    y_tri = rms_norm(x)


if __name__ == "__main__":
    dim = 8192
    rms_norm = RMSNorm(dim=6656)
    test_layer_norm(3840, 6656, torch.float16)
