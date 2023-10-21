import math

import matplotlib.pyplot as plt
import torch

with torch.backends.cuda.sdp_kernel(
    enable_flash=False,
    enable_math=False,
    enable_mem_efficient=True,
):
    # Dummy data
    query = torch.randn(5, 10, 64)  # (batch_size, sequence_length, feature_dim)
    key = torch.randn(5, 10, 64)
    value = torch.randn(5, 10, 64)

    # 1. Run with is_causal=True
    L, S = query.size(-2), key.size(-2)
    attn_bias = torch.zeros(L, S, dtype=query.dtype)
    temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
    attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
    attn_bias.to(query.dtype)
    result_causal = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=None, is_causal=True
    )

    # 2. Manually construct the attention mask
    Your_ATTN_MASK = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)

    # 3. Run with attn_mask=Your_ATTN_MASK
    result_mask = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=Your_ATTN_MASK, is_causal=False
    )

    # 4. Check if the results are the same
    are_same = torch.allclose(result_causal, result_mask)

    print(are_same)
    # Visualize the attention mask using a heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(Your_ATTN_MASK, cmap="gray")
    plt.colorbar(label="Attention Mask Value")
    plt.title("Your_ATTN_MASK Heatmap")
    plt.xlabel("Key Positions")
    plt.ylabel("Query Positions")
    plt.savefig("test.png")
    plt.show()
