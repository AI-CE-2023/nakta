import torch


def create_offset_v4(ctx_lengths, follow_lengths, device="cpu"):
    """
    Create an offset list based on the lengths of ctx and follow with corrected logic for follow_lengths.

    Args:
    - ctx_lengths (list): List of lengths for each ctx.
    - follow_lengths (list): List of lengths for each follow (length should be 4 times of ctx_lengths).

    Returns:
    - torch.Tensor: Offset tensor.
    """
    assert len(ctx_lengths) * 4 == len(follow_lengths)

    offset = []

    # For ctx_lengths
    for ctx_len in ctx_lengths:
        offset.extend(range(ctx_len))

    # For each group of 4 in follow_lengths
    ctx_idx = 0
    for i in range(0, len(follow_lengths), 4):
        follow_offset_start = ctx_lengths[ctx_idx]
        for j in range(4):  # since there are 4 follow_lengths for each ctx
            offset.extend(
                [
                    follow_offset_start + x % follow_lengths[i + j]
                    for x in range(follow_lengths[i + j])
                ]
            )
        ctx_idx += 1

    return torch.tensor(offset, dtype=torch.long, device=device)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ctx_lengths = [3, 4]
    follow_lengths = [1, 2, 3, 4, 1, 2, 3, 4]
    # Re-test the function with the new logic
    offset_v4 = create_offset_v4(ctx_lengths, follow_lengths)
    print(offset_v4)
    print(sum(ctx_lengths) + sum(follow_lengths))
    print(offset_v4.shape)
