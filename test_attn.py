import torch
import torch.nn.functional as F

seq_len = 100
# Optionally use the context manager to ensure one of the fused kernels is run
query = torch.rand(64, seq_len, 13, 128, dtype=torch.float16, device="cuda")
key = torch.rand(64, seq_len, 13, 128, dtype=torch.float16, device="cuda")
value = torch.rand(64, seq_len, 13, 128, dtype=torch.float16, device="cuda")

# query_s = query.view(1, 64 * seq_len, 13, 128)
# key_s = query.view(1, 64 * seq_len, 13, 128)
# value_s = query.view(1, 64 * seq_len, 13, 128)

query_s = torch.rand(1, seq_len * 64, 13, 128, dtype=torch.float16, device="cuda")
key_s = torch.rand(1, seq_len * 64, 13, 128, dtype=torch.float16, device="cuda")
value_s = torch.rand(1, seq_len * 64, 13, 128, dtype=torch.float16, device="cuda")


query = query.transpose(1, 2)
key = key.transpose(1, 2)
value = value.transpose(1, 2)

query_s = query_s.transpose(1, 2)
key_s = key_s.transpose(1, 2)
value_s = value_s.transpose(1, 2)

import time

with torch.backends.cuda.sdp_kernel(
    enable_flash=False,
    enable_math=False,
    enable_mem_efficient=True,
):
    for _ in range(10):
        a = time.time()
        re = F.scaled_dot_product_attention(query, key, value)
        b = time.time()
        print(b - a)
        r = b - a
    print(re.shape)
print("-" * 10)
with torch.backends.cuda.sdp_kernel(
    enable_flash=False,
    enable_math=True,
    enable_mem_efficient=False,
):
    for _ in range(10):
        a = time.time()
        re = F.scaled_dot_product_attention(query_s, key_s, value_s)
        b = time.time()
        print(b - a)
        r = b - a
    print(re.shape)
