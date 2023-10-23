import torch
from torch.nn.utils.rnn import pad_sequence

from nakta_model4.kernel.mask.offset import create_offset_v4
from nakta_model.kernel.Emb import RotaryEmbedding

torch.set_default_tensor_type(torch.cuda.HalfTensor)

bsz = 128
seqlen1 = 100
seqlen2 = 80
qk = 2
n_local_heads = 13
head_dim = 128

rotary = RotaryEmbedding(dim=head_dim, max_seq_len=100)

a = torch.randn(seqlen1, qk, n_local_heads, head_dim)
b = torch.randn(seqlen2, qk, n_local_heads, head_dim)

original = pad_sequence([a, b], batch_first=True)
print(original.shape)
test_in = torch.cat((a, b), dim=0)
test_in = test_in.view(seqlen1 + seqlen2, 1, qk, n_local_heads, head_dim)


# offset = torch.arange(0, seqlen1, dtype=torch.long).cuda()
# offset2 = torch.arange(0, seqlen2, dtype=torch.long).cuda()
# offset = torch.cat((offset, offset2), dim=0)
# offset = torch.cat([offset for _ in range(bsz)], dim=0)
q, k = rotary(original)
q_test, k_test = rotary(test_in, seqlen_offset=offset)
print(q.shape)
print(k.shape)
print(q_test.shape)
print(k_test.shape)
q_test = q_test[:seqlen1].view(1, seqlen1, n_local_heads, head_dim)
q = q[:1, :seqlen1, :, :]
print(q_test.shape)
print(q.shape)
# k_test = k_test.view(bsz, seqlen, n_local_heads, head_dim)
# print((q - q_test).abs().max())
# print((k - k_test).abs().max())

# import torch
# import torch.nn.functional as F

# # Optionally use the context manager to ensure one of the fused kernels is run
# query = torch.rand(64, 13, 128, 128, dtype=torch.float16, device="cuda")
# key = torch.rand(64, 13, 128, 128, dtype=torch.float16, device="cuda")
# value = torch.rand(64, 13, 128, 128, dtype=torch.float16, device="cuda")
# import time

# with torch.backends.cuda.sdp_kernel(
#     enable_flash=False,
#     enable_math=False,
#     enable_mem_efficient=True,
# ):
#     for _ in range(10):
#         a = time.time()
#         F.scaled_dot_product_attention(query, key, value)
#         b = time.time()
#         print(b - a)
#         r = b - a
# with torch.backends.cuda.sdp_kernel(
#     enable_flash=False,
#     enable_math=True,
#     enable_mem_efficient=False,
# ):
#     for _ in range(10):
#         a = time.time()
#         F.scaled_dot_product_attention(query, key, value)
#         b = time.time()
#         print(b - a)
#         r2 = b - a

# print(r2 / r)
