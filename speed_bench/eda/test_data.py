import pickle
from collections import Counter
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def gather_statistics(data: List[Tuple[List, List]]) -> List[Tuple[int, float]]:
    """
    Returns a list of tuples where the first item is the length of the context
    and the second item is the average length of the following sentences.
    """
    statistics = []
    for i in range(0, len(data), 4):  # Group by 4
        ctx = data[i][1]
        ctx_length = len(ctx)

        # Calculate average length of the next 4 sentences
        avg_next_length = sum([len(data[j][2]) for j in range(i, i + 4)]) / 4

        statistics.append((ctx_length, avg_next_length))

    return statistics


def gather_statistics_and_counts(
    data: List[Tuple[List, List]]
) -> Tuple[List[Tuple[int, float]], Counter]:
    """
    Returns a list of tuples where the first item is the length of the context
    and the second item is the average length of the following sentences.
    Also returns a Counter object with the counts of each context length.
    """
    statistics = []
    ctx_lengths = []
    for i in range(0, len(data), 4):  # Group by 4
        ctx = data[i][1]
        ctx_length = len(ctx)
        ctx_lengths.append(ctx_length)

        # Calculate average length of the next 4 sentences
        avg_next_length = sum([len(data[j][2]) for j in range(i, i + 4)]) / 4

        statistics.append((ctx_length, avg_next_length))

    counts = Counter(ctx_lengths)
    return statistics, counts


def bin_data(value: int, bin_size: int) -> int:
    """Bin data into specific intervals."""
    return int(round(value / bin_size) * bin_size)


def plot_statistics(statistics: List[Tuple[int, float]], bin_size):
    df = pd.DataFrame(
        statistics, columns=["Context Length", "Average Length of Following Sentences"]
    )

    # Bin the data
    df["Context Length"] = df["Context Length"].apply(lambda x: bin_data(x, bin_size))
    df["Average Length of Following Sentences"] = df[
        "Average Length of Following Sentences"
    ].apply(lambda x: bin_data(x, bin_size))

    # Create a heatmap using seaborn
    plt.figure(figsize=(12, 8))
    heatmap_data = (
        df.groupby(["Context Length", "Average Length of Following Sentences"])
        .size()
        .unstack()
        .fillna(0)
    )

    # Convert counts to ratios
    total_counts = heatmap_data.sum().sum()
    heatmap_data_ratio = heatmap_data / total_counts

    sns.heatmap(heatmap_data_ratio, cmap="YlGnBu", annot=True, fmt=".2%")

    plt.title(
        f"Heatmap (Binned by {bin_size}): Context Length vs. Average Length of Following Sentences"
    )
    plt.savefig(
        f"./Heatmap_Binned_by_{bin_size}_Context_Length_vs_Avg_Length_of_Following_Sentences_with_Ratios.png"
    )
    plt.show()


def plot_satatistic_bar(statistics: List[Tuple[int, float]], counts: Counter):
    ctx_lengths, avg_next_lengths = zip(*statistics)

    fig, ax1 = plt.subplots(figsize=(12, 7))

    # ax2 = ax1.twinx()
    unique_ctx_lengths = list(counts.keys())
    counts_list = [counts[length] for length in unique_ctx_lengths]
    ax1.scatter(
        unique_ctx_lengths,
        counts_list,
        # color="gray",
        # alpha=0.4,
        # width=5,
        label="Count of Context Lengths",
    )
    ax1.set_ylabel("Count", color="gray")
    ax1.tick_params(axis="y", labelcolor="gray")

    title = "Context Length vs. Count"
    fig.suptitle(title)
    fig.tight_layout()
    plt.savefig(f"./{title}.png")
    plt.show()


def gather_std_statistics(data: List[Tuple[List, List]]) -> List[Tuple[int, float]]:
    """
    Returns a list of tuples where the first item is the length of the context
    and the second item is the ratio of standard deviation to the average length of the following sentences.
    """
    statistics = []
    for i in range(0, len(data), 4):  # Group by 4
        ctx = data[i][1]
        ctx_length = len(ctx)

        # Calculate standard deviation and average length of the next 4 sentences
        lengths = [len(data[j][2]) for j in range(i, i + 4)]
        std_dev = np.std(lengths)
        avg_length = np.mean(lengths)

        # Calculate ratio
        ratio = std_dev / avg_length if avg_length != 0 else 0

        statistics.append((ctx_length, ratio))

    return statistics


def bin_data_for_ratio(value: float, bin_size: float) -> float:
    """Bin data into specific intervals for float values."""
    return round(value / bin_size) * bin_size


def plot_std_statistics(
    statistics: List[Tuple[int, float]],
    ctx_bin_size: int = 10,
    ratio_bin_size: float = 0.1,
):
    # Convert statistics to DataFrame
    df = pd.DataFrame(statistics, columns=["Context Length", "ratio(Std,AvgLen)"])

    # Bin the data
    df["Context Length"] = df["Context Length"].apply(
        lambda x: bin_data(x, ctx_bin_size)
    )
    df["ratio(Std,AvgLen)"] = df["ratio(Std,AvgLen)"].apply(
        lambda x: bin_data_for_ratio(x, ratio_bin_size)
    )

    # Create a heatmap using seaborn
    plt.figure(figsize=(12, 8))
    heatmap_data = (
        df.groupby(["Context Length", "ratio(Std,AvgLen)"]).size().unstack().fillna(0)
    )

    # Convert counts to ratios based on context length
    heatmap_data_ratio = heatmap_data.divide(heatmap_data.sum(axis=1), axis=0)

    sns.heatmap(heatmap_data_ratio, cmap="YlGnBu", annot=True, fmt=".2%")

    title = "Heatmap: Context Length Ratio vs. ratio(Std,AvgLen) of Following Sentences"
    plt.title(title)
    plt.savefig(f"./{title}.png")
    plt.show()


if __name__ == "__main__":
    # Sample data for demonstration purposes
    # data_sample = [
    #     (["This", "is", "a", "context"], ["This", "is", "a", "sentence"]),
    #     (["This", "is", "a", "context"], ["Another", "sentence"]),
    #     (["This", "is", "a", "context"], ["Yet", "another", "one"]),
    #     (["This", "is", "a", "context"], ["The", "last", "one", "here"]),
    #     (["A", "different", "context"], ["Different", "sentence"]),
    #     (["A", "different", "context"], ["Yet", "another", "different", "sentence"]),
    #     (["A", "different", "context"], ["More", "sentences"]),
    #     (["A", "different", "context"], ["Last", "different", "one"]),
    # ]

    with open("../test.pickle", "rb") as fr:
        data = pickle.load(fr)

    statistics = gather_statistics(data)
    plot_statistics(statistics, 10)

    statistics, counts = gather_statistics_and_counts(data)
    plot_satatistic_bar(statistics, counts)

    statistics = gather_std_statistics(data)
    plot_std_statistics(statistics)
