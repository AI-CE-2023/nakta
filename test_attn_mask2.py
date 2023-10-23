import torch
import torch.nn.functional as F
import xformers.ops as xops
from torch.nn.utils.rnn import pad_sequence
from xformers.ops import memory_efficient_attention

from nakta_model4.kernel.mask.mask import create_hellaswag_mask_v5

ctx = torch.load("./test/ctx.pt", map_location="cuda").half()
follows = []
for i in range(4):
    follows.append(torch.load(f"./test/follow{i}.pt", map_location="cuda").half())

tokens = torch.cat([ctx, ctx, *follows, *follows], dim=0).unsqueeze(dim=0)
print(tokens.shape)
tokens = tokens

ctx_len = [60] * 2
follow_lens = [20, 30, 40, 20] * 2
print(ctx_len)
print(follow_lens)
mask = create_hellaswag_mask_v5(ctx_len, follow_lens, device="cuda")
# with torch.backends.cuda.sdp_kernel(
#     enable_flash=False,
#     enable_math=True,
#     enable_mem_efficient=False,
# ):
#     result = F.scaled_dot_product_attention(tokens, tokens, tokens, attn_mask=mask)
# mask = xops.LowerTriangularMask()
result = memory_efficient_attention(tokens, tokens, tokens, attn_bias=mask)

torch.save(
    result.cpu(),
    "./test/test.pt",
)
print(result.shape)
# print(mask)
# print(mask.materialize(dtype=torch.float16).shape)
