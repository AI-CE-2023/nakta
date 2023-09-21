import torch

# 데이터 불러오기
a = torch.load("./weights/original/30B/consolidated.00.pth", map_location="cpu")
b = torch.load("./weights/original/30B/consolidated.01.pth", map_location="cpu")
c = torch.load("./weights/original/30B/consolidated.02.pth", map_location="cpu")
d = torch.load("./weights/original/30B/consolidated.03.pth", map_location="cpu")

# 모든 weight 모음
weights = [a, b, c, d]

num_layers = 60

for weight in weights:
    # Attention weights 합치기
    for i in range(num_layers):  # 0부터 15까지의 layer 번호 가정
        prefix = f"layers.{i}.attention."
        wq = weight[prefix + "wq.weight"]
        wk = weight[prefix + "wk.weight"]

        wattn = torch.cat([wq, wk], dim=0)
        weight[prefix + "wqk.weight"] = wattn.cpu()

        # 기존의 weights 삭제
        del weight[prefix + "wq.weight"]
        del weight[prefix + "wk.weight"]

    # Feed Forward weights 합치기
    # for i in range(num_layers):  # 0부터 15까지의 layer 번호 가정
    #     prefix = f"layers.{i}.feed_forward."
    #     w1 = weight[prefix + "w1.weight"]
    #     w3 = weight[prefix + "w3.weight"]

    #     wff = torch.cat([w1, w3], dim=0)
    #     weight[prefix + "wff.weight"] = wff.cpu()

    #     # 기존의 weights 삭제
    #     del weight[prefix + "w1.weight"]
    #     del weight[prefix + "w3.weight"]

# 모든 weights에 대해 tok_embeddings 합치기
tok_emb_total = torch.cat([w["tok_embeddings.weight"] for w in weights], dim=-1)

# 각 weight dictionary에 동일한 값 저장
for weight in weights:
    weight["tok_embeddings.weight"] = tok_emb_total


# 저장하기
torch.save(a, "./weights/modified/30B_2/consolidated.00.pth")
torch.save(b, "./weights/modified/30B_2/consolidated.01.pth")
torch.save(c, "./weights/modified/30B_2/consolidated.02.pth")
torch.save(d, "./weights/modified/30B_2/consolidated.03.pth")

import shutil

shutil.copy2(
    "./weights/original/30B/params.json", "./weights/modified/30B_2/params.json"
)
