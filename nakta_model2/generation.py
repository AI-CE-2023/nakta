import time
from typing import List

import torch

from .model import Transformer, Embedder
from .tokenizer import Tokenizer


class LLaMA:
    def __init__(self, model: Transformer, tokenizer: Tokenizer, embedder: Embedder):
        self.model = model
        self.tokenizer = tokenizer
        self.embedder = embedder

    def generate(
        self,
        prompts: List[str],
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> List[str]:
        bsz = len(prompts)
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos

        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[: len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))
        return decoded

    def prof(self, ctx_len: int, follow_len: int, batch_size: int, cached: bool):
        """
        Profiles the forward pass of a model using given seq_len and batch_size.

        Parameters:
        - seq_len (int): The sequence length for the input tokens.
        - batch_size (int): The number of samples in the batch.
        """

        torch.manual_seed(0)

        follow = 4

        assert batch_size % follow == 0

        # Generate random integers between 1 and 32000
        ctx_tokens = (
            torch.randint(1, 32001, (batch_size // follow, ctx_len)).cuda().long()
        )
        follow_tokens = torch.randint(1, 32001, (batch_size, follow_len)).cuda().long()

        if cached:
            for _ in range(2):
                self.model.forward(ctx_tokens, (0, 1, follow))
                self.model.forward(follow_tokens, (ctx_len, 1, follow))

            for _ in range(2):
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_push("forward")

                torch.cuda.nvtx.range_push("ctx")
                self.model.forward(ctx_tokens, (0, 1, follow))
                torch.cuda.nvtx.range_pop()

                torch.cuda.nvtx.range_push("follow")
                result = self.model.forward(follow_tokens, (ctx_len, 1, follow))
                torch.cuda.nvtx.range_pop()

                torch.cuda.nvtx.range_pop()
                torch.cuda.synchronize()
        else:
            ctx_tokens = ctx_tokens.repeat(follow, 1)
            tokens = torch.cat((ctx_tokens, follow_tokens), dim=1)

            cached_info = (0, -1, follow)
            assert cached_info[1] == -1

            for _ in range(2):
                h = self.embedder.forward(tokens)
                self.model.forward(h, cached_info)

            for _ in range(2):
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_push("forward")
                h = self.embedder.forward(tokens)
                self.model.forward(h, cached_info)
                torch.cuda.nvtx.range_pop()
                torch.cuda.synchronize()

        return result

    def bench(self, batch, cache: bool):
        if cache:
            self.model.forward(batch[0], (0, 1, 4))
            self.model.forward(batch[1], (batch[2], 1, 4))
        else:
            self.model.forward(batch, (0, -1, 4))

    def speed_test(self, ctx_len: int, follow_len: int, batch_size: int):
        torch.manual_seed(0)

        follow = 4
        num_iters = 5

        assert batch_size % follow == 0

        # Generate random integers between 1 and 32000
        ctx_tokens = (
            torch.randint(1, 32001, (batch_size // follow, ctx_len)).cuda().long()
        )
        follow_tokens = torch.randint(1, 32001, (batch_size, follow_len)).cuda().long()

        cached_times = []

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        for _ in range(num_iters):
            torch.cuda.synchronize()
            start_event.record()

            self.model.forward(ctx_tokens, (0, 1, follow))
            result = self.model.forward(follow_tokens, (ctx_len, 1, follow))

            end_event.record()
            torch.cuda.synchronize()

            cached_times.append(start_event.elapsed_time(end_event) / 1000)

        ctx_tokens = ctx_tokens.repeat(follow, 1)
        tokens = torch.cat((ctx_tokens, follow_tokens), dim=1)

        n_cached_times = []
        for _ in range(num_iters):
            torch.cuda.synchronize()
            start_event.record()

            self.model.forward(tokens, (0, -1, follow))

            end_event.record()
            torch.cuda.synchronize()

            n_cached_times.append(start_event.elapsed_time(end_event) / 1000)

        return (
            batch_size / (sum(cached_times) / len(cached_times)),
            batch_size / (sum(n_cached_times) / len(cached_times)),
        )


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
