import random
from typing import Callable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

from nakta_model import Tokenizer


class SpeedDataset(Dataset):
    def __init__(
        self,
        tokenizer_path: str,
        order: str = "random",
        default_batch_size: int = 32,
        batch_scheduler: Optional[Callable[[int, List[List[int]]], int]] = None,
        device: str = "cpu",
    ):
        self.device = device

        self.dataset = load_dataset("hellaswag", split="validation")

        self.order = order
        self.default_batch_size = default_batch_size
        self.batch_scheduler = batch_scheduler
        self.tokenizer = Tokenizer(model_path=tokenizer_path)
        self.tokenized_strings = self._concat_strings()[:1]

        self.batches = self._create_dataset()

    def _concat_strings(self) -> List[List[int]]:
        to_return = []
        for s_z in self.dataset:
            tokens = []
            query = (
                s_z["activity_label"]
                + ": "
                + s_z["ctx_a"]
                + " "
                + s_z["ctx_b"].capitalize()
            )
            query = self.tokenizer.encode(query, bos=False, eos=False)
            for c in s_z["endings"]:
                cont_encode = self.tokenizer.encode(c[:], bos=False, eos=False)
                tokens.append((query + cont_encode, len(cont_encode), len(c)))
            to_return.append((tokens, int(s_z["label"])))
        return to_return

    def _create_dataset(self):
        if self.order == "ascending":
            self.tokenized_strings.sort(key=lambda x: sum([len(i[0]) for i in x[0]]))
        elif self.order == "descending":
            self.tokenized_strings.sort(
                key=lambda x: sum([len(i[0]) for i in x[0]]), reverse=True
            )
        elif self.order == "None":
            pass
        else:
            raise ValueError("Order must be 'ascending', 'None' or 'descending'")

        batches = []
        index = 0
        golds = []
        # Loop until all tokenized strings are batched
        while index < len(self.tokenized_strings):
            if self.batch_scheduler:
                batch_size = self.batch_scheduler(index, self.tokenized_strings[index:])
            else:
                batch_size = max(self.default_batch_size // 4, 1)
            batch_pack = self.tokenized_strings[index : index + batch_size]
            batch = []
            gold = []
            continuation_lens = []
            continuation_str_lens = []
            for i in batch_pack:
                gold.append(i[1])
                for j in i[0]:
                    batch.append(j[0])
                    continuation_lens.append(j[1])
                    continuation_str_lens.append(j[2])
            inp_lens = [len(t) for t in batch]
            max_length = max(inp_lens)
            # tokens = [
            #     t + [self.tokenizer.pad_id] * (max_length - len(t)) for t in batch
            # ]
            tokens = [t + [0] * (max_length - len(t)) for t in batch]

            # Convert the list of lists to a tensor
            tokens = torch.tensor(tokens, dtype=torch.long).to(self.device)

            batches.append(
                (tokens, inp_lens, continuation_lens, gold, continuation_str_lens)
            )
            index += batch_size

        return batches

    def gather_statistics(self) -> pd.DataFrame:
        # Gather tokenized_length and batch_size
        data = []
        for batch in self.batches:
            average_length = sum(len(s) for s in batch) / len(batch)
            batch_size = len(batch)
            data.append({"tokenized_length": average_length, "batch_size": batch_size})

        # Convert to DataFrame
        df = pd.DataFrame(data)
        return df

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

    from tqdm import tqdm

    # # Test
    # with open("../test2.pickle", "rb") as fr:
    #     strings = pickle.load(fr)
    # Create the SpeedDatasetTorch object
    speed_dataset_torch = SpeedDataset(
        tokenizer_path="../../weights/original/tokenizer.model",
        order="descending",
        default_batch_size=64,
        # batch_scheduler=length_based_batch_scheduler,
    )

    # print(speed_dataset_torch)
    # print(len(speed_dataset_torch.batches))
    # Using DataLoader to load the dataset
    dataloader = DataLoader(
        speed_dataset_torch, batch_size=1, shuffle=False, collate_fn=collate_fn
    )
    cnt = 0
    for tokens, inp_lens, continuation_lens, golds, cont_str_lens in tqdm(dataloader):
        print(tokens.shape)
        cnt += 1
        if cnt == 100:
            break
        # print(strings[i])
    # for i in dataloader:
    #     for j in i:
    #         print(j)
    #     break

    # Gathering statistics
    # df = speed_dataset_torch.gather_statistics()
    # print(df)

    # # Plotting the statistics
    # plt.figure(figsize=(10, 6))
    # plt.plot(df["tokenized_length"], df["batch_size"], marker="o", linestyle="-")
    # plt.xlabel("Average Tokenized Length")
    # plt.ylabel("Batch Size")
    # plt.title("Batch Size vs. Average Tokenized Length")
    # plt.grid(True)
    # plt.savefig("./scheduled_info.png")

    # # Print shape of the first batch
    # first_batch = next(iter(dataloader))
    # print(first_batch.shape)
