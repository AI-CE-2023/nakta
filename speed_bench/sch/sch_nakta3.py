import math
import random
from typing import Callable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

from nakta_model5 import Tokenizer


def _remove_padding(output, batch_info):
    # Flatten the sequences based on batch_info
    if type(batch_info) == int:
        return output
    flattened_output = torch.cat(
        [output[i, : batch_info[i]] for i in range(len(batch_info))], dim=0
    )
    return flattened_output


class SpeedDataset(Dataset):
    def __init__(
        self,
        # strings: List[str],
        tokenizer_path: str,
        order: str = "random",
        default_batch_size: int = 32,
        min_batch_size: int = 10,
        batch_scheduler: Optional[Callable[[int, List[List[int]]], int]] = None,
        device: str = "cpu",
        candidate_multiple: int = 4,
    ):
        self.device = device
        self.candidate_multipe = candidate_multiple

        self.dataset = load_dataset("hellaswag", split="validation")

        self.order = order
        self.default_batch_size = default_batch_size
        self.min_batch_size = min_batch_size
        self.batch_scheduler = batch_scheduler
        self.tokenizer = Tokenizer(model_path=tokenizer_path)
        self.tokenized_strings = self._concat_strings()

        self.batches = self._create_dataset()

    def _concat_strings(
        self,
    ) -> List:
        to_return = []
        for s_z in self.dataset:
            cont_tokens = []
            query = (
                s_z["activity_label"]
                + ": "
                + s_z["ctx_a"]
                + " "
                + s_z["ctx_b"].capitalize()
            )
            query = self.tokenizer.encode(query, bos=False, eos=False)
            ce_len = 0
            query_len = len(query)
            for c in s_z["endings"]:
                cont_encode = self.tokenizer.encode(c[:], bos=False, eos=False)
                cont_len = len(cont_encode)
                cont_tokens.append(
                    (cont_encode, cont_len, len(c), query_len + cont_len)
                )
                ce_len += cont_len
            to_return.append(
                (
                    query,
                    cont_tokens,
                    query_len,
                    ce_len / 4,
                    int(s_z["label"]),
                )
            )
        return to_return

    def _sort_tokenized_strings(self):
        """Sorts the tokenized strings based on the specified order."""
        if self.order == "ascending":
            self.tokenized_strings.sort(key=lambda x: x[2])
        elif self.order == "descending":
            self.tokenized_strings.sort(key=lambda x: x[2], reverse=True)
        # elif self.order == "None":
        #     pass
        else:
            raise ValueError("Order must be 'ascending', 'descending' or 'None")

    def _get_batch_size(self, index):
        """Returns the batch size based on the scheduler or the default value."""
        # To do: Implement
        if self.batch_scheduler:
            return self.batch_scheduler(index, self.tokenized_strings[index:])
        return self.default_batch_size

    def _adjust_batch_size(self, ctx, batch_size):
        """Adjusts the batch size based on the difference in context lengths."""
        min_ctx = min([len(c) for c in ctx])
        max_ctx = max([len(c) for c in ctx])
        if (max_ctx - min_ctx) / min_ctx > 0.8:
            return max(int(batch_size * 0.5), self.min_batch_size)
        return batch_size

    def _process_batch(self, batch):
        """Processes a batch to create context and following tokens."""
        targets = [b[4] for b in batch]

        ctx = [b[0] for b in batch]
        ctx_lens = [len(c) for c in ctx]
        min_ctx = min(ctx_lens)

        f_tokens = [f[0] for b in batch for f in b[1]]
        f_lens = [f[1] for b in batch for f in b[1]]
        fs_lens = [f[2] for b in batch for f in b[1]]
        inp_lens = [f[3] for b in batch for f in b[1]]

        # for_inp_lens = ctx_lens * 4
        # assert len(for_inp_lens) == len(fs_lens)

        assert len(ctx) * 4 == len(f_tokens)

        group_size = 4
        new_followings = []
        for i, f in enumerate(f_tokens):
            current_ctx = ctx[i // group_size]
            new_following = current_ctx[min_ctx:] + f
            new_followings.append(new_following)

        num_groups = len(new_followings) // group_size
        new_followings = [
            new_followings[i + j * group_size]
            for j in range(num_groups)
            for i in range(group_size)
        ]
        new_ctx = [c[:min_ctx] for c in ctx]

        ctx_tokens = torch.tensor(new_ctx, dtype=torch.long).to(self.device)

        following_lens = [len(f) for f in new_followings]
        max_following = max(following_lens)
        min_following = min(following_lens)
        avg_following = sum(following_lens) / len(following_lens)

        mean_following = avg_following
        squared_diff = [(x - mean_following) ** 2 for x in following_lens]
        variance = sum(squared_diff) / len(following_lens)
        std_following = math.sqrt(variance)

        padded_followings = [f + [0] * (max_following - len(f)) for f in new_followings]

        following_tokens = torch.tensor(padded_followings, dtype=torch.long).to(
            self.device
        )

        following_tokens = (
            following_tokens.reshape(
                following_tokens.shape[0] // 4, 4, following_tokens.shape[1]
            )
            .permute(1, 0, 2)
            .contiguous()
            .view(following_tokens.shape[0], -1)
        )

        following_lens = (
            torch.tensor(following_lens, dtype=torch.long)
            .reshape(len(following_lens) // 4, 4)
            .permute(1, 0)
            .contiguous()
            .flatten()
            .tolist()
        )

        following_tokens_in = _remove_padding(following_tokens, following_lens)

        # assert ctx_tokens.shape[0] * 4 == following_tokens.shape[0]

        return (
            ctx_tokens,
            following_tokens_in,
            min_ctx,
            min_following,
            std_following / avg_following,
            inp_lens,
            f_lens,
            targets,
            fs_lens,
            following_lens,
            # following_tokens,
        )

    def _list_chunk(self, lst, n):
        return [lst[i : i + n] for i in range(0, len(lst), n)]

    def _create_dataset(self):
        """Main method to create the dataset."""
        self._sort_tokenized_strings()

        batches = []
        index = 0

        while index < len(self.tokenized_strings):
            # 4배로 우선 뽑고 안에서 following 에 의해 정렬
            original_batch_size = self._get_batch_size(index)
            batch_size = original_batch_size * self.candidate_multipe
            batch = self.tokenized_strings[index : index + batch_size]

            # adjust by ctx
            # ctx = [b[0] for b in batch]
            # batch_size = self._adjust_batch_size(ctx, batch_size)
            # batch = self.tokenized_strings[index : index + batch_size]

            # adjust by followings
            ## Important
            batch.sort(key=lambda x: x[3])
            batch = self._list_chunk(batch, original_batch_size)
            # f_batch_size = self._adjust_f_batch_size(ctx, batch_size)

            for b in batch:
                batches.append(self._process_batch(b))
            index += batch_size

        return batches

    def gather_statistics(self):
        min_ctxs = []
        ratios = []

        for batch in self.batches:
            min_ctxs.append(batch[2])
            ratios.append(batch[4])

        # Binning for min_ctx and ratios
        binned_min_ctxs = [5 * round(value / 5) for value in min_ctxs]
        binned_ratios = [0.2 * round(value / 0.2) for value in ratios]

        return binned_min_ctxs, binned_ratios

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, index):
        return self.batches[index]


def length_based_batch_scheduler(index: int, tokenized_strings: List[List[int]]) -> int:
    next_batch_size = 32  # default batch size
    next_batch = tokenized_strings[:next_batch_size]
    average_length = sum(len(s) for s in next_batch) / len(next_batch)

    if average_length <= 50:
        return 64
    elif average_length <= 100:
        return 32
    else:
        return 16


def collate_fn(batch):
    return batch[0]


if __name__ == "__main__":
    import pickle

    import matplotlib.pyplot as plt
    import seaborn as sns

    # Test
    # Test
    # with open("../test.pickle", "rb") as fr:
    #     strings = pickle.load(fr)
    default_batch_size = 16
    # Create the SpeedDatasetTorch object
    candidate_multipe = 6
    speed_dataset_torch = SpeedDataset(
        tokenizer_path="../../weights/original/tokenizer.model",
        order="ascending",
        default_batch_size=default_batch_size,
        candidate_multiple=candidate_multipe
        # batch_scheduler=length_based_batch_scheduler,
    )
    check_num = 1
    print(speed_dataset_torch.batches[check_num][0].shape)
    print(speed_dataset_torch.batches[check_num][1].shape)

    print(len(speed_dataset_torch.batches))

    # print(speed_dataset_torch.batches[0])

    # Using DataLoader to load the dataset
    dataloader = DataLoader(
        speed_dataset_torch, batch_size=1, shuffle=False, collate_fn=collate_fn
    )

    # # Print shape of the first batch
    # first_batch = next(iter(dataloader))
    # print(first_batch[0].shape)
    # print(first_batch[1].shape)

    # Gather statistics from the dataset
    binned_min_ctxs, binned_ratios = speed_dataset_torch.gather_statistics()

    # Convert data to DataFrame and bin
    heatmap_data = pd.DataFrame({"min_ctx": binned_min_ctxs, "ratios": binned_ratios})

    heatmap_data = (
        heatmap_data.groupby(["min_ctx", "ratios"]).size().reset_index(name="counts")
    )
    heatmap_pivot = heatmap_data.pivot("min_ctx", "ratios", "counts")

    # Plot the heatmap using Seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_pivot, cmap="YlGnBu", annot=True)

    plt.title(
        "Heatmap of binned min_ctx vs. binned ((max_following - min_following) / min_following)"
    )
    plt.xlabel("Binned std/avg of following lengths")
    plt.ylabel("Binned min_ctx")
    plt.tight_layout()

    # Save the heatmap as an image
    plt.savefig(f"binned_heatmap_{default_batch_size}_{candidate_multipe}.png")
    plt.show()
