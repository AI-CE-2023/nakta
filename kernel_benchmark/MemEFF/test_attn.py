import torch
from torch.nn.functional import scaled_dot_product_attention

bsz = 64
seqlen = 60
n_local_heads = 13
head_dim = 128

xq = torch.rand(bsz, seqlen, n_local_heads, head_dim)
xk = torch.rand(bsz, seqlen, n_local_heads, head_dim)
xv = torch.rand(bsz, seqlen, n_local_heads, head_dim)

with torch.backends.cuda.sdp_kernel(
    enable_flash=False,
    enable_math=False,
    enable_mem_efficient=True,
):
    output = scaled_dot_product_attention(xq, xk, xv, is_causal=True)
