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


# Establish data
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
text = ["hello I want to eat some [MASK] meat today. It's thanksgiving [MASK] all!", "yo yo what's up"]

# tokenize text and pass into model
tokens = tokenizer(text, return_tensors="pt", padding=True)
input_ids = tokens["input_ids"]

# bf forward pass
jax_input_ids = jnp.array(input_ids.numpy(), dtype=int)
out_bf = bf_embs(input_ids=jax_input_ids)