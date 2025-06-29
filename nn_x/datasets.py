from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Union, Any, Dict, List, Tuple, Iterator, TypeVar
from collections import defaultdict
from glob import glob

import datetime as dt
import os

from tqdm import trange
from tqdm.auto import tqdm

from .embedding import NOMIC_EMBEDDING_MAX_WINDOW_SIZE, process_batch

DEFAULT_CONFIG = config_parameters = {
    "window_size": NOMIC_EMBEDDING_MAX_WINDOW_SIZE,
    "stride":      NOMIC_EMBEDDING_MAX_WINDOW_SIZE // 4,
    "prefix":      "clustering: ",
    "model_name":  "nomic-ai/nomic-embed-text:v1.5",
    "output_dir":  os.path.join("outputs", "some_run"),
    "datetime":    dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),

}

T = TypeVar('T')
T_name = str
T_doc  = Union[str, 'torch.Tensor'] # type: ignore

def dataset_from_txt_dir(directory: os.PathLike = "", search=".txt", recursive=True) -> Dict[T_name, T_doc]:
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

def windows_anchor_words(items: List[T], window_size: int = NOMIC_EMBEDDING_MAX_WINDOW_SIZE // 7, stride: int = NOMIC_EMBEDDING_MAX_WINDOW_SIZE // 14, word_seps: set = {' ', '\n'}, max_pull: int = 14, max_push: int = 14) -> List[List[T]]:
    o: List[List[T]] = []
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
        o.append(items[start:end])
        nominal_start += stride
    return o

@dataclass
class WindowItem:
    doc_name: T_name
    part_idx: int
    doc_ctnt: T_doc
    
    def __len__(self) -> int:
        return len(self.doc_ctnt)

@dataclass
class BatchedWindowItems:
    doc_name: List[T_name]
    part_idx: List[int]
    doc_ctnt: List[T_doc]
    
    def unbach(self) -> List[WindowItem]:
        return [WindowItem(name, idx, content) for name, idx, content in zip(self.doc_name, self.part_idx, self.doc_ctnt)]

    def __len__(self) -> int:
        return len(self.doc_ctnt)

class WindowDataset:
    def __init__(self, documents: Dict[T_name, T_doc], window_size: int = NOMIC_EMBEDDING_MAX_WINDOW_SIZE // 7,
                 stride: int = NOMIC_EMBEDDING_MAX_WINDOW_SIZE // 14, word_seps: set = {' ', '\n'},
                 num_workers: int = os.cpu_count(), max_pull: int = 14, max_push: int = 14):
        self.documents = documents

        self.window_size = window_size
        self.stride      = stride
        self.word_seps   = word_seps
        self.max_pull    = max_pull
        self.max_push    = max_push

        self._windows: List[WindowItem] = []
        self._batches: List[BatchedWindowItems] = []
        
        self.pool = ProcessPoolExecutor(num_workers)

    def window(self):
        "process parallelly iterates over documents, and populates self.windows with WindowItems"
        futures: Dict[str, List[str]] = dict()

        for doc_name, doc_content in self.documents.items():
            future = self.pool.submit(
                windows_anchor_words,
                doc_content,
                self.window_size,
                self.stride,
                self.word_seps,
                self.max_pull,
                self.max_push
            )
            futures[future] = doc_name

        self._windows.clear()
        for v_future in tqdm(as_completed(futures.keys()), total=len(futures), desc="Windowing documents", unit=" docs", leave=True):
            k_doc_name = futures[v_future]
            for i_part_idx, r_doc_ctnt in enumerate(v_future.result()):
                self._windows.append(WindowItem(k_doc_name, i_part_idx, r_doc_ctnt))
        
        self._windows.sort(key=(lambda window: window.length), reverse=True)
        
    def batch(self, max_batch = 200) -> List[List[WindowItem]]:
        current_batch: List[     WindowItem ] = []
        
        # area size_of(LLM(side(batch_size) by side(padded_max_lenght))) that roughtly fits into gpu memory
        # since one side of this rectangle formula moves with the dinamic window size, the other needs to
        # move in correspondence to keep area/ram roughlt constant and not underfeed the gpu nor blow it up
        max_batch_token_area = self.window_size * max_batch
        max_lenght           = 1

        self._batches.clear()
        for window in tqdm(self._windows, desc="batching windows", leave=True, unit=" win"):
            current_batch.append(window)
            max_lenght = max(max_lenght, window.length)

            if (len(current_batch) == max_batch) or (max_lenght * len(current_batch) > max_batch_token_area):
                self._batches.append(batch_window_list(current_batch))
                current_batch = [ ]
                max_lenght    =  1
        
    def embed(self):
        for batch in tqdm(self._batches, desc="embedding sentences", unit="batches", leave=True):
            batch.doc_ctnt = process_batch(batch.doc_ctnt)

    def __iter__(self):
        if self._batches:
            yield from self._batches
        elif self._windows:
            yield from self._windows
        else:
            yield from self.documents.items()

    def __len__(self):
        if self._batches:
            return len(self._batches)
        elif self._windows:
            return len(self._windows)
        else:
            return len(self.documents)

def batch_window_list(batch: List[WindowItem]) -> BatchedWindowItems:
    return BatchedWindowItems(
        doc_name=[window.doc_name for window in batch],
        part_idx=[window.part_idx for window in batch],
        doc_ctnt=[window.doc_ctnt for window in batch],
    )

def flatten_batch_list(batches: List[BatchedWindowItems]) -> List[WindowItem]:
    return [item for batch in batches for item in batch.unbach()]
