from typing import Any, Dict, List, Tuple, Iterator, TypeVar
from collections import defaultdict
from glob import glob
import os

from tqdm.auto import tqdm
import numpy as np
from sklearn.cluster import KMeans

from .embedding import NOMIC_EMBEDDING_MAX_WINDOW_SIZE, nomic_embedding_tokenizer

T = TypeVar('T')

def dataset_from_txt_dir(directory: os.PathLike = "") -> List[str]:
    data = defaultdict(list)
    for file in tqdm(list(glob(os.path.join(directory, "*.txt"))), desc="Loading files"):
        with open(file, "r", encoding="utf-8") as f:
            data[file.split(os.sep)[-1].replace('.txt', '')] = f.read()
    return dict(data)

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

def bucket_items_by_length_kmeans(items_input: Dict[str, str], max_len_filter: int = NOMIC_EMBEDDING_MAX_WINDOW_SIZE // 3,
                                  k_clusters: int = 10, random_state: int = 42) -> Dict[int, List[str]]:
    """
    Group item IDs by KMeans clusters over the lengths of their associated values.

    Args:
        items_input: Mapping from item IDs to their string values.
        max_len_filter: Maximum length (exclusive) to include in clustering.
        k_clusters: Desired number of clusters.
        random_state: Seed for KMeans initialization.

    Returns:
        A dict mapping each cluster's maximum value length to the list of item IDs in that cluster.
    """
    # Filter out items with non-positive or too-long values
    valid = [(item_id, val) for item_id, val in items_input.items() if 0 < len(val) < max_len_filter]
    
    if not valid: return {}

    ids, values     = zip(*valid)
    lengths         = np.array([len(v) for v in values])
    n, unique_count = len(lengths), len(np.unique(lengths))

    n_clusters = max(min(k_clusters, n, unique_count), 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
    kmeans.fit(lengths.reshape(-1, 1))

    cluster_to_ids  = defaultdict(list)
    cluster_max_len = defaultdict(int)
    for item_id, length, label in zip(ids, lengths, kmeans.labels_):
        cluster_to_ids[label].append(item_id)
        if length > cluster_max_len[label]:
            cluster_max_len[label] = length
    
    output: Dict[int, List[str]] = {}
    for label, id_list in cluster_to_ids.items():
        max_len = int(cluster_max_len[label])
        output.setdefault(max_len, []).extend(id_list)

    return output
