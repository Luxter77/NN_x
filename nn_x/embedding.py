from transformers import AutoTokenizer, AutoModel
import torch

NOMIC_EMBEDDING_MAX_WINDOW_SIZE = 8192

# there are on average 1.5175101060015719 tokens per word in a window
NOMIC_WORD_TO_TOKEN_RATIO = 1.52
# there are on average 0.2591943988246274 tokens per char in a window
NOMIC_CHAR_TO_TOKEN_RATIO = 0.26

nomic_embedding_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', trust_remote_code=True, max_length=NOMIC_EMBEDDING_MAX_WINDOW_SIZE)
nomic_embedding_model     = AutoModel.from_pretrained('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True, rotary_scaling_factor=2)
nomic_embedding_model.eval()

nomic_embedding_model.to('cuda' if torch.cuda.is_available() else 'cpu')
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