
from typing import Dict, List

from datasets import load_dataset, Dataset

from .embedding import NOMIC_EMBEDDING_MAX_WINDOW_SIZE, nomic_embedding_tokenizer

# # For Clustering:
# # 1. 20 Newsgroups Dataset:
# #    - Contains newsgroup documents across different topics.
# #    - Good for exploring text clustering based on topic similarity.
# dataset_config = ('newsgroup', '18828_alt.atheism')
# # 2. AG News Dataset:
# #    - Contains news articles categorized into 4 classes (World, Sports, Business, Sci/Tech).
# #    - Useful for text classification, but clustering can also be attempted based on news topics.
# #dataset_config = ('ag_news')
# # For Classification:
# # 1. IMDB Movie Reviews:
# #    - Contains movie reviews labeled as positive or negative.
# #    - Classic dataset for sentiment analysis and binary classification.
# #dataset_config = ('imdb')
# # 2. SMS Spam Collection:
# #    - Contains SMS messages labeled as spam or ham (not spam).
# #    - Useful for spam detection and binary text classification.
# #dataset_config = ('sms_spam')
#
# dataset = load_dataset(*dataset_config)
#
# sentences = [("classification: " + example) for example in dataset['train']['text'][:100]] # ignore: reportGeneralTypeIssues
#
# for line in sentences:
#     ids = nomic_embedding_tokenizer(line, padding=True, truncation=False).input_ids
#     idl = len(ids)
#     if idl > NOMIC_EMBEDDING_MAX_WINDOW_SIZE:
#         print(line)
#         print(idl)
#         raise Exception(f"oof: line is longer than context window! {idl} > {EMBEDDING_MAX_WINDOW_SIZE} in {line}")

def load_winowed_dataset(dataset_name: str, split: str = "train", tokenizer: callable = lambda x: nomic_embedding_tokenizer(x, padding=True, truncation=False), window_size: int = NOMIC_EMBEDDING_MAX_WINDOW_SIZE, stride: int = NOMIC_EMBEDDING_MAX_WINDOW_SIZE // 3, reverse: bool = False, text_key: str = "text") -> Dataset:
    """
    Load a dataset and apply the DocumentWindowPipeline to it.
    """
    ds = load_dataset(dataset_name, split=split)
    pipeline = DocumentWindowPipeline(tokenizer, window_size, stride, reverse, text_key)
    return pipeline.apply(ds)

class DocumentWindowPipeline:
    "A pipeline to chunk documents into windows of tokens for embedding."
    def __init__(self, tokenizer: callable, window_size: int, stride: int, reverse: bool = False, text_key: str = "text"):
        """
        tokenizer: any callable that maps {"text": str} → {"input_ids": List[int], ...}
        window_size: length of each chunk
        stride: hop size between chunks
        reverse: if True, reverse token order before windowing
        text_key: name of the raw text field in your dataset
        """
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.stride = stride
        self.reverse = reverse
        self.text_key = text_key

    def _tokenize(self, example: Dict) -> Dict:
        tokens = self.tokenizer(example[self.text_key],
                                 truncation=False)["input_ids"]
        if self.reverse:
            tokens = list(reversed(tokens))
        return {"input_ids": tokens}

    def _window(self, example: Dict) -> Dict[str, List]:
        tokens: List[int] = example["input_ids"]
        windows = []
        for start in range(0, max(1, len(tokens) - self.window_size + 1), self.stride):
            chunk = tokens[start : start + self.window_size]
            # if you want to pad shorter final windows, you could do it here
            windows.append(chunk)
        return {"input_ids": windows}

    def apply(self, ds: Dataset) -> Dataset:
        # 1) tokenize (and reverse if flagged)
        ds_tok = ds.map(self._tokenize,
                        batched=False,
                        remove_columns=[self.text_key])

        # 2) turn each doc→many windows, flatten into new rows
        ds_win = ds_tok.flat_map(self._window)

        # 3) (optional) set torch format for DataLoader
        ds_final = ds_win.set_format("torch", columns=["input_ids"])
        return ds_final
