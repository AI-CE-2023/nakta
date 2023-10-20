from contextlib import contextmanager
from time import perf_counter

import torch
import triton
import triton.language as tl


@triton.jit
def split_pad_kernel(
    input_ptr,
    output_ptr,
    start_ptr,
    end_ptr,
    max_len,
    hidden_dim,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)

    i_start = tl.load(start_ptr + bid)
    i_end = tl.load(end_ptr + bid)

    block_start = pid * BLOCK_SIZE

    off_1 = i_start + block_start + tl.arange(0, BLOCK_SIZE)
    mask = off_1 < i_end
    x = tl.load(input_ptr + off_1, mask=mask)

    off_2 = bid * max_len * hidden_dim + block_start + tl.arange(0, BLOCK_SIZE)
    mask = off_2 < i_end - i_start
    tl.store(output_ptr + off_2, x, mask=mask)


def split_and_pad(input: torch.Tensor, batch_info: torch.IntTensor) -> torch.Tensor:
    assert input.ndim == 2

    batch_size = len(batch_info)
    max_len = torch.max(batch_info).item()
    pos_end = torch.cumsum(batch_info, dim=0)
    pos_start = pos_end - batch_info[0]

    out = torch.zeros((batch_size, max_len, input.size(1))).cuda()
    split_pad_kernel[(32, batch_size)](
        input_ptr=input,
        output_ptr=out,
        start_ptr=pos_start,
        end_ptr=pos_end,
        max_len=max_len,
        hidden_dim=input.size(1),
        BLOCK_SIZE=1024,
    )

    return out


@contextmanager
def catchtime() -> float:
    start = perf_counter()
    yield lambda: perf_counter() - start
    print(f"Time: {perf_counter() - start:.3f} seconds")


def main():
    batch_size = 16
    eff_seqlen = 128
    hidden_dim = 1610

    batch_info = torch.randint(low=10, high=eff_seqlen, size=(batch_size,)).cuda()
    q = torch.randn((batch_size * sum(batch_info), hidden_dim)).cuda()

    with catchtime():
        max_len = torch.max(batch_info)
        seqs1 = torch.stack(
            [
                torch.nn.functional.pad(
                    q[torch.sum(batch_info[:i]) : torch.sum(batch_info[: i + 1]), :],
                    (0, 0, 0, max_len - batch_info[i]),
                )
                for i in range(len(batch_info))
            ]
        )

    with catchtime():
        seqs2 = split_and_pad(q, batch_info)

    dseq = seqs1 - seqs2
    dseq.abs_()
    dseq = dseq.cpu()
    print(dseq.mean(), dseq.std(), dseq.max(), dseq.min())


if __name__ == "__main__":
    main()
