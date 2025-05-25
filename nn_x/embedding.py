from typing import List
import gc

from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import torch
import torch.nn.functional as F

from nn_x.extra import mean_pooling

NOMIC_EMBEDDING_MAX_WINDOW_SIZE = 8192

# there are on average 1.5175101060015719 tokens per word in a window
NOMIC_WORD_TO_TOKEN_RATIO = 1.52
# there are on average 0.2591943988246274 tokens per char in a window
NOMIC_CHAR_TO_TOKEN_RATIO = 0.26

# if cuda
device = torch.device("cpu")
do_cuda = torch.cuda.is_available()
if do_cuda:
    device = torch.device("cuda")
#    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
#else:
#    print("Using CPU")

nomic_embedding_tokenizer = AutoTokenizer.from_pretrained('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True, max_length=NOMIC_EMBEDDING_MAX_WINDOW_SIZE)
nomic_embedding_model     = AutoModel    .from_pretrained('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True, rotary_scaling_factor=2).to(device)
nomic_embedding_model.eval()

nomic_embedding_model.requires_grad_(False)

def estimate_word_count_from_tokens(tokens: int) -> int:
    "Estimate the number of words in a given number of tokens."
    return int(tokens / NOMIC_WORD_TO_TOKEN_RATIO)

def estimate_token_count_from_words(words: int) -> int:
    "Estimate the number of tokens in a given number of words."
    return int(words * NOMIC_WORD_TO_TOKEN_RATIO)

def estimate_char_count(tokens: int) -> int:
    "Estimate the number of characters in a given number of tokens."
    return int(tokens / NOMIC_CHAR_TO_TOKEN_RATIO)

def estimate_token_count_from_chars(chars: int) -> int:
    "Estimate the number of tokens in a given number of characters."
    return int(chars * NOMIC_CHAR_TO_TOKEN_RATIO)

def process_batch(batched_text: List[str]) -> torch.Tensor:
    with torch.no_grad():
        t = nomic_embedding_tokenizer(batched_text, padding=True, truncation=True, return_tensors="pt", max_length=NOMIC_EMBEDDING_MAX_WINDOW_SIZE).to(device)
        try:
            x = nomic_embedding_model(**t)
            x = mean_pooling(x, t["attention_mask"])
            x = F.normalize(x, p=2, dim=1)
        except torch.cuda.OutOfMemoryError as oom:
            tqdm.write(f"Out of memory error: {oom}")
            tqdm.write(f"Out of memory error: " + str(t['input_ids'].shape))
            raise oom
    if do_cuda:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch._C._cuda_clearCublasWorkspaces()
    return x.detach().cpu()
