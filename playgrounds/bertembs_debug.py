from jax import numpy as jnp
import transformers
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForMaskedLM, 
    BertForMaskedLM, 
    BertTokenizer, 
    BertTokenizerFast, 
    BertEmbeddings,
    BfBertEmbeddings,
    BertConfig,
)

config = BertConfig.from_pretrained(pretrained_model_name_or_path="/home/kevin/code/rycolab/brunoflow/models/bert/config.json")
bf_embs = BfBertEmbeddings(config)
torch_embs = BertEmbeddings(config)
print(bf_embs)
print(torch_embs)

# Save torch BertEmbeddings to file
save_path = "bertembeddings_torch.pt"
torch.save(torch_embs.state_dict(), save_path)

torch_embs.load_state_dict(torch.load(save_path))
bf_embs.load_state_dict(torch.load(save_path))