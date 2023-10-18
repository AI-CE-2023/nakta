import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

csv_file_path = "./data/transformer_block.csv"
title = "Transformer Block"
df = pd.read_csv(csv_file_path)

# Duration 열에서 " μs"를 추출하고 1/1000을 곱하여 "ms"로 변환
df["Duration"] = df["Duration"].apply(
    lambda x: float(x.replace(" μs", "")) / 1000
    if " μs" in x
    else float(x.replace(" ms", ""))
)

# Name으로 그룹화하고, Duration의 합계를 계산
grouped_df = df.groupby("Name")["Duration"].sum().reset_index()

# 데이터 준비
names = grouped_df["Name"]
durations = grouped_df["Duration"]
colors = sns.color_palette("Set2", len(names))


def my_autopct(pct):
    return f"{pct:.1f}%" if pct >= 2 else ""


plt.figure(figsize=(8, 8))
plt.pie(
    durations,
    labels=names,
    autopct=my_autopct,
    startangle=140,
    colors=colors,
)
plt.axis("equal")  # 원 모양으로 조정

# 타이틀 추가 및 간격 조절
plt.title(title, pad=40, fontsize=20)

# 그래프와 타이틀 간의 간격 조절
plt.subplots_adjust(top=0.85)

graph_file_path = f"./graph/{title}.png"
plt.savefig(graph_file_path, bbox_inches="tight")
plt.show()
