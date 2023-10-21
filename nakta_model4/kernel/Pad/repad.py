from contextlib import contextmanager
from time import perf_counter

import torch
import triton
import triton.language as tl
from torch.nn.utils.rnn import pad_sequence


@triton.jit
def split_pad_kernel_corrected(
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

    off_1 = i_start * hidden_dim + block_start + tl.arange(0, BLOCK_SIZE)
    mask = off_1 < i_end * hidden_dim
    x = tl.load(input_ptr + off_1, mask=mask)

    off_2 = bid * max_len * hidden_dim + block_start + tl.arange(0, BLOCK_SIZE)
    mask = block_start + tl.arange(0, BLOCK_SIZE) < (i_end - i_start) * hidden_dim
    tl.store(output_ptr + off_2, x, mask=mask)


def split_and_pad(input: torch.Tensor, batch_set) -> torch.Tensor:
    assert input.ndim == 2

    batch_size, max_len, pos_end, pos_start = batch_set

    out = torch.zeros((batch_size, max_len, input.size(1))).cuda()
    split_pad_kernel_corrected[(32, batch_size)](
        input_ptr=input,
        output_ptr=out,
        start_ptr=pos_start,
        end_ptr=pos_end,
        max_len=max_len,
        hidden_dim=input.size(1),
        BLOCK_SIZE=1024,
    )

    return out


def torch_native(Q, batch_info):
    Q = Q.split(batch_info)
    return pad_sequence(Q, batch_first=True)


def torch_native2(Q, batch_info, out):
    Q = Q.split(batch_info)
    for i, val in enumerate(Q):
        out[i, : batch_info[i], :] = val
    return out


@contextmanager
def catchtime() -> float:
    start = perf_counter()
    yield lambda: perf_counter() - start
    print(f"Time: {perf_counter() - start:.5f} seconds")


def adjust_batch_info_corrected(batch_info, target_sum):
    """Adjust the elements of batch_info so that they sum up to target_sum."""
    diff = int(target_sum - batch_info.sum().item())
    quotient, remainder = divmod(diff, len(batch_info))

    # Distribute the difference uniformly across the elements
    batch_info += quotient
    for i in range(remainder):
        batch_info[i] += 1

    return batch_info


def main():
    batch_size = 64
    eff_seqlen = 128
    hidden_dim = 1610

    batch_info = torch.randint(low=10, high=eff_seqlen, size=(batch_size,)).cuda()
    q = torch.randn((sum(batch_info), hidden_dim)).cuda()

    batch_size = len(batch_info)
    max_len = torch.max(batch_info).item()
    pos_end = torch.cumsum(batch_info, dim=0)
    pos_start = pos_end - batch_info[0]

    batch_set = (batch_size, max_len, pos_end, pos_start)

    batch_info_native = batch_info.tolist()
    print(q.shape)
    print(sum(batch_info_native))

    out = torch.zeros(batch_size, max_len, hidden_dim, device="cuda")

    for _ in range(10):
        with catchtime():
            max_len = torch.max(batch_info)
            seqs1 = torch.stack(
                [
                    torch.nn.functional.pad(
                        q[
                            torch.sum(batch_info[:i]) : torch.sum(batch_info[: i + 1]),
                            :,
                        ],
                        (0, 0, 0, max_len - batch_info[i]),
                    )
                    for i in range(len(batch_info))
                ]
            )
    print("-" * 10)
    for _ in range(10):
        with catchtime():
            seqs2 = split_and_pad(q, batch_set)
    print("-" * 10)
    for _ in range(10):
        with catchtime():
            seqs3 = torch_native(q, batch_info_native)
    print("-" * 10)

    for _ in range(10):
        with catchtime():
            seqs4 = torch_native2(q, batch_info_native, out)

    dseq = seqs1 - seqs3
    dseq.abs_()
    dseq = dseq.cpu()
    print(dseq.mean(), dseq.std(), dseq.max(), dseq.min())


if __name__ == "__main__":
    main()
