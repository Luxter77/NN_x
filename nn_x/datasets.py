from typing import Dict, List, Iterator, TypeVar
from glob import glob
import os

from tqdm.auto import tqdm

from .embedding import NOMIC_EMBEDDING_MAX_WINDOW_SIZE, nomic_embedding_tokenizer

T = TypeVar('T')

def dataset_from_txt_dir(directory: os.PathLike = "") -> List[str]:
    data = dict()
    for file in tqdm(list(glob(os.path.join(directory, "*.txt"))), desc="Loading files"):
        with open(file, "r", encoding="utf-8") as f:
            data[file.split(os.sep)[-1].replace('.txt', '')](f.read())
    return data

def tokenize_dataset(dataset: List[str]) -> List[List[int]]:
    return [nomic_embedding_tokenizer(item, padding=True, truncation=False)["input_ids"] for item in dataset]

def reverse_dataset(dataset: List[T]) -> List[T]:
    return list(list(reversed(tokens)) for tokens in dataset)

def windows(items: List[List[T]], window_size: int, stride: int) -> Iterator[T]:
    for start in range(0, max(1, len(items) - window_size + 1), stride):
        yield items[start : start + window_size]


def windows_anchor_words(items: List[T], window_size: int = NOMIC_EMBEDDING_MAX_WINDOW_SIZE // 7, stride: int = NOMIC_EMBEDDING_MAX_WINDOW_SIZE // 14, word_seps: set = {' ', '\n'}, max_pull: int = 14, max_push: int = 14) -> Iterator[List[T]]:
    n = len(items)
    nominal_start = 0

    while nominal_start < n:
        start = nominal_start
        pulled = 0
        if start > 0 and items[start] not in word_seps:
            while start > 0 and items[start] not in word_seps and pulled < max_pull:
                start -= 1
                pulled += 1
            if items[start] in word_seps:
                start += 1

        end = min(start + window_size, n)
        pushed = 0
        if end < n and items[end] not in word_seps:
            while end < n and items[end] not in word_seps and pushed < max_push:
                end += 1
                pushed += 1

        yield items[start:end]

        nominal_start += stride

