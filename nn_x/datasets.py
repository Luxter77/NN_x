from typing import Any, Dict, List, Tuple, Iterator, TypeVar
from collections import defaultdict
from glob import glob

import datetime as dt
import os

from sklearn.cluster import KMeans
from tqdm import trange
from tqdm.auto import tqdm
import numpy as np

from .embedding import NOMIC_EMBEDDING_MAX_WINDOW_SIZE, process_buffer

DEFAULT_CONFIG = config_parameters = {
    "window_size": NOMIC_EMBEDDING_MAX_WINDOW_SIZE,
    "stride":      NOMIC_EMBEDDING_MAX_WINDOW_SIZE // 4,
    "prefix":      "clustering: ",
    "model_name":  "nomic-ai/nomic-embed-text:v1.5",
    "output_dir":  os.path.join("outputs", "some_run"),
    "datetime":    dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
}

T = TypeVar('T')

def dataset_from_txt_dir(directory: os.PathLike = "", search=".txt", recursive=True) -> List[str]:
    data = defaultdict(list)
    for file in tqdm(list(glob(os.path.join(directory, "**", f"*{search}"), recursive=recursive)), desc="Loading files"):
        with open(file, "r", encoding="utf-8") as f:
            content = f.read()
            if content:
                # tqdm.write(f"Adding {file} to dataset")
                data[file.split(os.sep)[-1].replace(search, '')] = content
            # else:
                # tqdm.write(f"File {file} is empty?")
    return dict(data)

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
    valid = [(item_id, val) for item_id, val in items_input.items() if 0 < len(val) <= max_len_filter]
    
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

def batch(iterable, n=1):
    iterable = list(iterable)
    l = len(iterable)
    for ndx in trange(0, l, n, desc="batching", leave=False):
        yield iterable[ndx:min(ndx + n, l)]

def window(string: str, config_parameters: Dict[str, Any] = DEFAULT_CONFIG) -> Iterator[str]:
    total = (len(string) + config_parameters["stride"] - 1) // config_parameters["stride"]
    for window in tqdm(windows_anchor_words(string, window_size=config_parameters["window_size"], stride=config_parameters["stride"]), total=total, desc="Windowing", leave=False, unit=" windows"):
        yield config_parameters["prefix"] + window

def produce_linear_batches(sentences: dict[str, str], MAX_BATCH = 200, max_acceptable_lenght = NOMIC_EMBEDDING_MAX_WINDOW_SIZE // 3):
    buffer_texts: list[str] = []
    buffer_names: list[str] = []

    batches: list[tuple[list[str], list[str]]] = []

    long_sentences = [(name, text) for name, text in sentences.items() if len(text) > max_acceptable_lenght]

    # Progress bar over all sentences
    total_pbar = tqdm(long_sentences, total=len(long_sentences), desc="encoding sentences")

    for name, text in total_pbar:
        total_pbar.set_postfix_str(f"processing: {name}")
        max_sent_len = 0
        for sent in window(string=text):
            max_sent_len = max(max_sent_len, len(sent))
            buffer_texts.append(sent)
            buffer_names.append(name)

            if ( max_sent_len * len(buffer_texts) > (max_acceptable_lenght * MAX_BATCH)):
                batches.append((buffer_texts, buffer_names))
                buffer_texts, buffer_names = [], []

    if buffer_texts:
        batches.append((buffer_texts, buffer_names))

    return batches

def produce_clustered_batches(sentences: dict[str, str], MAX_BATCH = 200, max_acceptable_lenght = NOMIC_EMBEDDING_MAX_WINDOW_SIZE // 3):
    buffer_texts: list[str] = []
    buffer_names: list[str] = []

    batches: list[tuple[list[str], list[str]]] = []

    clustered_sentences = bucket_items_by_length_kmeans(sentences, max_acceptable_lenght, 30, 2)

    total_pbar = tqdm(enumerate(clustered_sentences.items()), total=len(clustered_sentences), desc="encoding sentences by clusters, loading buffer.")

    for i, (cluster, names) in total_pbar:
        total_pbar.set_postfix_str(f"ID {i} LEN({len(names)}) MAX({cluster})")
        for name in tqdm(names, desc="encoding sentences", leave=False, unit="sentences"):
            for sent in window(string=sentences[name]):
                buffer_texts.append(sent)
                buffer_names.append(name)

                if (len(buffer_texts) >= MAX_BATCH) or (cluster * len(buffer_texts) > (max_acceptable_lenght * MAX_BATCH)):
                    batches.append((buffer_texts, buffer_names))
                    buffer_texts = []
                    buffer_names = []

    if buffer_texts:
        batches.append((buffer_texts, buffer_names))

    return batches

def batched_consumer(batches: List[Tuple[List[str], List[str]]], sentence_embeddings: defaultdict[str, List[str]]):
    for buffer_texts, buffer_names in tqdm(batches, desc="encoding sentences", leave=True, unit="sentences"):
        process_buffer(buffer_names, buffer_texts, sentence_embeddings)
