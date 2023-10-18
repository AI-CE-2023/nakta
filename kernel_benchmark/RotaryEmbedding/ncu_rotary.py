import torch

from nakta_model.kernel.Emb import RotaryEmbedding


def test_rotary_embedding():
    dim = 128
    end = 60

    query = torch.rand(64, end, 13, dim, dtype=torch.float16).cuda()
    key = torch.rand(64, end, 13, dim, dtype=torch.float16).cuda()
    value = torch.rand(64, end, 13, dim, dtype=torch.float16).cuda()

    rotary_emb = RotaryEmbedding(
        dim=dim, max_seq_len=end, interleaved=True, device=query.device
    )
    q, k = rotary_emb(query, key)


if __name__ == "__main__":
    test_rotary_embedding()
