import random
from typing import Callable, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import torch

from nakta_model import Tokenizer


class SpeedDataset:
    def __init__(
        self,
        strings: List[str],
        tokenizer_path: "str",
        order: str = "random",
        default_batch_size: int = 32,
        batch_scheduler: Optional[Callable[[int, List[List[int]]], int]] = None,
        device: str = "cpu",
    ):
        self.device = device

        self.strings = strings
        self.order = order
        self.default_batch_size = default_batch_size
        self.batch_scheduler = batch_scheduler
        self.tokenizer = Tokenizer(model_path=tokenizer_path)
        self.tokenized_strings = self._tokenize_strings(self.strings)

        self.batches = self._create_dataset()

    def _tokenize_strings(self, strings: List[str]) -> List[List[int]]:
        return [self.tokenizer.encode(s, bos=False, eos=False) for s in strings]

    def _create_dataset(self) -> List[List[List[int]]]:
        if self.order == "ascending":
            self.tokenized_strings.sort(key=len)
        elif self.order == "descending":
            self.tokenized_strings.sort(key=len, reverse=True)
        elif self.order == "random":
            random.shuffle(self.tokenized_strings)
        else:
            raise ValueError("Order must be 'random', 'ascending', or 'descending'")

        batches = []
        index = 0

        while self.tokenized_strings:
            current_batch_size = (
                self.batch_scheduler(index, self.tokenized_strings)
                if self.batch_scheduler
                else self.default_batch_size
            )

            ending = (
                current_batch_size
                if len(self.tokenized_strings) >= current_batch_size
                else len(self.tokenized_strings)
            )

            batch = self.tokenized_strings[:ending]
            max_prompt_size = max([len(t) for t in batch])

            tokens = (
                torch.full((ending, max_prompt_size), self.tokenizer.pad_id)
                .to(device=self.device)
                .long()
            )

            batches.append(tokens)
            self.tokenized_strings = self.tokenized_strings[current_batch_size:]
            index += 1

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


if __name__ == "__main__":
    # Test
    strings = [
        "hello world",
        "I love HuggingFace",
        "Tokenization is fun",
        "Bert",
        "Additional sentences can be added here. This is a bit longer than previous ones.",
    ] * 64
    speed_dataset = SpeedDataset(
        strings,
        tokenizer_path="../../weights/original/tokenizer.model",
        order="ascending",
        default_batch_size=32,
        batch_scheduler=length_based_batch_scheduler,
    )
    df = speed_dataset.gather_statistics()
    print(df)
    plt.figure(figsize=(10, 6))
    plt.plot(df["tokenized_length"], df["batch_size"], marker="o", linestyle="-")
    plt.xlabel("Average Tokenized Length")
    plt.ylabel("Batch Size")
    plt.title("Batch Size vs. Average Tokenized Length")
    plt.grid(True)
    plt.savefig("./scheduled_info.png")
    print(speed_dataset.batches[0].shape)
