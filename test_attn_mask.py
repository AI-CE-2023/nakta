import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

ctx = torch.load("./test/ctx.pt", map_location="cuda").half()
follows = []
for i in range(4):
    follows.append(torch.load(f"./test/follow{i}.pt", map_location="cuda").half())

ctx = ctx.unsqueeze(dim=0)
ctx = ctx.repeat(8, 1, 1, 1)

follows = pad_sequence(follows, batch_first=True)
# follows = follows[0].unsqueeze(dim=0)
follows = follows.repeat(2, 1, 1, 1)
tokens = torch.cat([ctx, follows], dim=1)
tokens = tokens.transpose(1, 2)

with torch.backends.cuda.sdp_kernel(
    enable_flash=False,
    enable_math=False,
    enable_mem_efficient=True,
):
    result = F.scaled_dot_product_attention(tokens, tokens, tokens, is_causal=True)
print(result.shape)
torch.save(
    result.transpose(1, 2).cpu(),
    "./test/ref.pt",
)
