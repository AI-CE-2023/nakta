import torch


def create_hellaswag_mask_v5(ctx_lengths, follow_lengths, device="cpu"):
    """
    Create a mask for the Hellaswag dataset based on the lengths of ctx and follow.

    Args:
    - ctx_lengths (list or torch.Tensor): List or tensor of lengths for each ctx.
    - follow_lengths (list or torch.Tensor): List or tensor of lengths for each follow (length should be 4 times of ctx_lengths).
    - device (str): Device to run the operations on, either 'cpu' or 'cuda'.

    Returns:
    - torch.Tensor: Mask for Hellaswag dataset.
    """
    follow_nums = 4
    assert len(follow_lengths) == follow_nums * len(
        ctx_lengths
    ), "The length of follow_lengths should be 4 times of ctx_lengths."

    ctx_lengths = torch.tensor(ctx_lengths, device=device)
    follow_lengths = torch.tensor(follow_lengths, device=device)

    # Calculate total lengths for ctx and follow
    total_ctx_length = ctx_lengths.sum().item()
    total_follow_length = follow_lengths.sum().item()

    # Initialize a mask with all zeros (indicating masking) and dtype as torch.bool
    total_length = total_ctx_length + total_follow_length
    mask = torch.zeros(
        (total_length, total_length + total_length % 8), dtype=torch.bool, device=device
    )

    # Non-masking for ctx and follow based on their lengths
    start_idx = 0
    for ctx_len in ctx_lengths:
        end_idx = start_idx + ctx_len
        mask[start_idx:end_idx, start_idx:end_idx] = torch.tril(
            torch.ones((ctx_len, ctx_len), dtype=torch.bool, device=device)
        )
        start_idx = end_idx

    for follow_len in follow_lengths:
        end_idx = start_idx + follow_len
        mask[start_idx:end_idx, start_idx:end_idx] = torch.tril(
            torch.ones((follow_len, follow_len), dtype=torch.bool, device=device)
        )
        start_idx = end_idx

    # Connect each ctx to its follows (making sure to connect each ctx to 4 follows)
    ctx_start = 0
    follow_start = total_ctx_length
    for ctx_len in ctx_lengths:
        ctx_end = ctx_start + ctx_len
        follow_end = (
            follow_start + follow_lengths[:follow_nums].sum().item()
        )  # connect to 4 follows
        mask[follow_start:follow_end, ctx_start:ctx_end] = True
        ctx_start = ctx_end
        follow_start = follow_end
        follow_lengths = follow_lengths[follow_nums:]  # move to the next set of follows

    mask = torch.where(
        mask,
        torch.tensor(0.0, device=device, dtype=torch.float16),
        torch.tensor(float("-inf"), device=device, dtype=torch.float16),
    )
    mask = mask.view(1, 1, mask.shape[0], mask.shape[1]).repeat(1, 13, 1, 1)
    # print(mask.shape)
    return mask[:, :, :, :total_length]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Test the corrected function with the given ctx and follow lengths
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_ctx_lengths = [5, 2]
    test_follow_lengths = [5, 6, 7, 8, 2, 3, 4, 5]
    corrected_hellaswag_mask_v3 = create_hellaswag_mask_v5(
        test_ctx_lengths, test_follow_lengths, device=device
    )

    # Visualize the corrected hellaswag mask
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corrected_hellaswag_mask_v3.cpu().int().numpy(),
        cmap="Blues",
        cbar=False,
        annot=True,
        fmt=".0f",
    )
    plt.title("Corrected Hellaswag Mask (v3)")
    plt.savefig("./test.png")
