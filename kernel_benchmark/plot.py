import matplotlib.pyplot as plt
import pandas as pd

# 데이터 불러오기
data = pd.read_csv("./MemEFF/causal/fused-attention-batch4-head13-d128-fwd-causal.csv")
print(data)


plt.plot(
    data["N_CTX"],
    data["pytorch-mem-eff"],
    marker="o",
    color="blue",
    label="PyTorch Memory Efficiency",
)
plt.plot(
    data["N_CTX"],
    data["pytorch-native"],
    marker="o",
    color="red",
    label="PyTorch Native",
)

# 평균적으로 몇 배 빨라졌는지 계산
speedup = (data["pytorch-native"] / data["pytorch-mem-eff"]).mean()
print(f"Average Speedup: {speedup:.2f}x")

title = "Attention(Causal)(batch = 4, head = 13, head dim = 128)"
plt.xlabel("N_CTX")
plt.ylabel("TFLops")
plt.title(f"{title}\n")

# X 축 표시값 설정
plt.xticks(data["N_CTX"], labels=data["N_CTX"].astype(str), rotation=45)

# 범례 추가
plt.legend()

# 그래프 저장
plt.tight_layout()  # 레이블이 그래프와 겹치지 않도록 자동 조정
plt.savefig(f"{title}.png")

# 그래프 화면에 표시
plt.show()
