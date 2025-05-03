from transformers import AutoTokenizer, AutoModel
import torch

NOMIC_EMBEDDING_MAX_WINDOW_SIZE = 8192

nomic_embedding_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
nomic_embedding_model     = AutoModel.from_pretrained('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True, safe_serialization=True)
nomic_embedding_model.eval()

nomic_embedding_model.to('cuda' if torch.cuda.is_available() else 'cpu')
nomic_embedding_model.requires_grad_(False)