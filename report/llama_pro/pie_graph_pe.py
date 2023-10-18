import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 파일 경로 및 타이틀 설정
csv_file_path = "./data/parallel_emb.csv"
title = "Parallel Embedding"

# 데이터 불러오기 및 전처리
df = pd.read_csv(csv_file_path)
df["Duration"] = df["Duration"].str.replace(" μs", "").astype(float)

# Name으로 그룹화하고, Duration의 합계를 계산
grouped_df = df.groupby("Name")["Duration"].sum().reset_index()

# 데이터 준비
names = grouped_df["Name"]
durations = grouped_df["Duration"]
colors = sns.color_palette("Set2", len(names))

# 파이 차트 그리기
plt.figure(figsize=(8, 8))
plt.pie(
    durations,
    labels=names,
    autopct="%1.1f%%",
    startangle=140,
    colors=colors,
)
plt.axis("equal")  # 원 모양으로 조정

# 타이틀 및 간격 설정
plt.title(title, pad=40, fontsize=20)
plt.subplots_adjust(top=0.85)

# 그래프 저장 및 출력
graph_file_path = f"./graph/{title}.png"
plt.savefig(graph_file_path, bbox_inches="tight")
plt.show()
