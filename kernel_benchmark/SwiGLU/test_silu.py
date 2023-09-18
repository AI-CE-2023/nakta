import torch
from nakta_attn import silu_and_mul

bsz = 64
seqlen = 60
hidden_dim = 4480

x1 = torch.rand_like(bsz, seqlen, hidden_dim)
x2 = torch.rand_like(bsz, seqlen, hidden_dim)
out = torch.zeros_like(x1)
silu_and_mul(out, x1, x2)
