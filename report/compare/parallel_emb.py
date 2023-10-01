import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 데이터 생성
data = {
    "Name": ["SwiGLU", "Fused SwiGLU"],
    "Duration": ["32.655 ms", "26.445 ms"],
}

df = pd.DataFrame(data)

# Duration을 ms 단위의 float로 변환
df["Duration"] = df["Duration"].apply(
    lambda x: float(x.replace(" μs", "")) / 1000
    if " μs" in x
    else float(x.replace(" ms", ""))
)
colors = sns.color_palette("Set2", len(df["Name"]))
# 속도 (연산속도 = 1/시간) 계산
df["Speed"] = 1 / df["Duration"]
title = "SwiGLU"
# 바 그래프 그리기
plt.bar(df["Name"], df["Speed"], color=colors)
plt.ylabel("Speed (1/ms)")
plt.title(title)
# plt.savefig(f"./graph/{title}.png")
plt.show()

# 속도 향상률 계산
speed_increase = ((df.loc[1, "Speed"] - df.loc[0, "Speed"]) / df.loc[0, "Speed"]) * 100

# 결과 출력
print(f"Fused SwiGlu is {speed_increase:.2f}% faster than SwiGLU.")
