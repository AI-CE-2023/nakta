import torch

torch.manual_seed(0)
# total = 170
ctx_len = 60
follow_lens = [20, 30, 40, 20]

num_heads = 13
head_dim = 128

c = torch.randn(ctx_len, num_heads, head_dim)
torch.save(c, "ctx.pt")
for i, val in enumerate(follow_lens):
    torch.save(torch.randn(val, num_heads, head_dim), f"follow{i}.pt")
