import csv

import matplotlib.pyplot as plt


def read_csv_data(filename: str):
    """Reads data from a CSV file and returns lists for ctx_len, cached_speed, and n_cached_speed"""
    with open(filename, "r") as csvfile:
        csvreader = csv.reader(csvfile)
        headers = next(csvreader)

        ctx_lens, cached_speeds, n_cached_speeds = [], [], []
        for row in csvreader:
            try:
                ctx_lens.append(int(row[0]))
                cached_speeds.append(float(row[1]))
                n_cached_speeds.append(float(row[2]))
            except:
                pass

    return ctx_lens, cached_speeds, n_cached_speeds


def plot_combined_data_total_speed(batch_sizes: list):
    """Plots data from multiple CSV files on a single graph in terms of total speed for the entire batch"""
    plt.figure(figsize=(10, 6))

    for batch_size in batch_sizes:
        filename = f"speed_test_results_batch_{batch_size}.csv"
        ctx_lens, cached_speeds, n_cached_speeds = read_csv_data(filename)

        # Convert per batch speeds to total speeds for the entire batch
        cached_speeds_total = [speed for speed in cached_speeds]
        n_cached_speeds_total = [speed for speed in n_cached_speeds]

        plt.plot(
            ctx_lens,
            cached_speeds_total,
            label=f"cached_speed (Batch Size: {batch_size})",
            marker="o",
        )
        plt.plot(
            ctx_lens,
            n_cached_speeds_total,
            label=f"n_cached_speed (Batch Size: {batch_size})",
            marker="x",
        )

    plt.xlabel("ctx_len")
    plt.ylabel("Total Speed (batch/second)")
    plt.title(
        "Combined Cached Speed vs Non-Cached Speed by Context Length (Total Speed)"
    )
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        "Combined Cached Speed vs Non-Cached Speed by Context Length (Total Speed).png"
    )
    plt.show()


# List of batch sizes for which data was saved in CSV files
batch_sizes = [16, 32, 64, 72, 80, 92]
plot_combined_data_total_speed(batch_sizes)
