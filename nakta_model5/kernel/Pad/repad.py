from contextlib import contextmanager
from time import perf_counter

import torch
import triton
import triton.language as tl
from torch.nn.utils.rnn import pad_sequence

torch.set_printoptions(profile="full")
torch.set_printoptions(linewidth=300, sci_mode=False, precision=4)


@triton.jit
def split_pad_kernel(
    input_ptr,
    output_ptr,
    start_ptr,
    len_ptr,
    hidden_dim,
    stride_i0,
    stride_o0,
    stride_o1,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)

    i_start = tl.load(start_ptr + bid)
    len = tl.load(len_ptr + bid)

    off = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = off < len * hidden_dim
    vec = tl.load(input_ptr + i_start * stride_i0 + off, mask=mask)

    off1 = off // hidden_dim
    off2 = off % hidden_dim

    # mask = off1 < len
    tl.store(output_ptr + bid * stride_o0 + off1 * stride_o1 + off2, vec, mask=mask)


def split_and_pad(input: torch.Tensor, batch_info_set) -> torch.Tensor:
    if type(batch_info_set) == int:
        return input
    assert input.ndim == 2
    assert input.is_contiguous()

    batch_info, batch_size, hidden_dim, max_len, start, output = batch_info_set
    split_pad_kernel[
        lambda meta: (triton.cdiv(hidden_dim * max_len, meta["BLOCK_SIZE"]), batch_size)
    ](
        input_ptr=input,
        output_ptr=output,
        start_ptr=start,
        len_ptr=batch_info,
        hidden_dim=hidden_dim,
        stride_i0=input.stride(0),
        stride_o0=output.stride(0),
        stride_o1=output.stride(1),
        BLOCK_SIZE=2048,
    )

    return output


@contextmanager
def catchtime() -> float:
    start = perf_counter()
    yield lambda: perf_counter() - start
    print(f"Time: {perf_counter() - start:.5f} seconds")


def create_batch_info_set(batch_info, hidden_dim):
    batch_size = len(batch_info)
    max_len = torch.max(batch_info).item()
    start = torch.zeros(
        batch_size,
        dtype=torch.long,
        device=batch_info.device,
    )
    start[1:] = torch.cumsum(batch_info, dim=0)[:-1]

    return [batch_info, batch_size, hidden_dim, max_len, start]


def create_batch_info_set2(batch_info, hidden_dim):
    batch_size = len(batch_info)
    batch_info_tensor = torch.tensor(batch_info, dtype=torch.long, device="cuda")
    max_len = torch.max(batch_info_tensor).item()
    start = torch.zeros(
        batch_size,
        dtype=torch.long,
        device=batch_info_tensor.device,
    )
    start[1:] = torch.cumsum(batch_info_tensor, dim=0)[:-1]
    output1 = torch.zeros(
        (batch_size, max_len, hidden_dim), device="cuda", dtype=torch.float16
    )
    output2 = torch.zeros(
        (batch_size, max_len, hidden_dim * 2), device="cuda", dtype=torch.float16
    )
    output3 = torch.zeros(
        (batch_size, max_len, hidden_dim * 4), device="cuda", dtype=torch.float16
    )
    return [
        # query
        [batch_info_tensor, batch_size, hidden_dim, max_len, start, output1],
        # key
        [batch_info_tensor, batch_size, hidden_dim * 2, max_len, start, output2],
        # total
        [batch_info_tensor, batch_size, hidden_dim * 4, max_len, start, output3],
        # for remove
        batch_info,
    ]


def _rebuild_padding(Q, batch_info):
    if type(batch_info) == int:
        return Q
    Q = Q.split(batch_info)
    return pad_sequence(Q, batch_first=True)


def main():
    batch_size = 24 * 4
    eff_seqlen = 100
    hidden_dim = 3328

    batch_info = torch.randint(
        low=20, high=eff_seqlen, size=(batch_size,), device="cuda"
    )
    batch_info_set = create_batch_info_set(batch_info, hidden_dim)
    # batch_info = torch.LongTensor([4, 3, 2, 3]).cuda()
    q = torch.randn(size=(batch_info.sum().item(), hidden_dim), device="cuda")

    print("batch_info:", batch_info, batch_info.float().mean().item())

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
    # print(seqs1)
    for _ in range(10):
        with catchtime():
            seqs2 = split_and_pad(q, batch_info_set)
    print("-" * 10)
    # print(seqs2)
    batch_info = batch_info.tolist()
    for _ in range(10):
        with catchtime():
            torch.cuda.nvtx.range_push("torch")
            seqs3 = _rebuild_padding(q, batch_info)
            torch.cuda.nvtx.range_pop()
    print("-" * 10)
    # print(seqs1)
    for _ in range(10):
        with catchtime():
            torch.cuda.nvtx.range_push("triton")
            seqs2 = split_and_pad(q, batch_info_set)
            torch.cuda.nvtx.range_pop()

    dseq = seqs2 - seqs3
    dseq.abs_()
    dseq = dseq.cpu()
    print("Mean:", dseq.mean(), "Max:", dseq.max())


if __name__ == "__main__":
    main()

"""
CUDA_LAUNCH_BLOCKING=1 nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas --force-overwrite true -o ./repad.nsys-rep python repad.py
"""
