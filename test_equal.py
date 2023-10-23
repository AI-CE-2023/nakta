import torch

ctx_lens = [60] * 16
follow_lens = [40] * 64

test = torch.load("./nakta.pt", map_location="cpu")
test = test[sum(ctx_lens) :]
test = test.split(follow_lens)
ref = torch.load("./original.pt", map_location="cpu")

for i, val in enumerate(test):
    to_comp = ref[i, ctx_lens[i // 4] : ctx_lens[i // 4] + follow_lens[i], :]
    print((to_comp - val).abs().mean())
