import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from jax import numpy as jnp
import transformers
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    BertTokenizer,
    BertTokenizerFast,
    BertEmbeddings,
    BfBertEmbeddings,
    BertConfig,
    BertPreTrainedModel,
    BertModel,
    BfBertModel,
    BertForMaskedLM,
    BfBertForMaskedLM,
)
from utils import check_bf_param_weights_match_torch

# model = BertModel.from_pretrained("google/bert_uncased_L-2_H-128_A-2")

# model = BertModel.from_pretrained("bert-base-uncased")
# bf_model = BfBertModel.from_pretrained("bert-base-uncased")
# check_bf_param_weights_match_torch(bf_model, model)  # YAY THIS WORKS!!!

model = BertForMaskedLM.from_pretrained("bert-base-uncased")
bf_model = BfBertForMaskedLM.from_pretrained("bert-base-uncased")
check_bf_param_weights_match_torch(bf_model, model)  # YAY THIS WORKS!!!


# config = BertConfig.from_pretrained(
#     pretrained_model_name_or_path="/home/kevin/code/rycolab/brunoflow/models/bert/config.json"
# )
# bf_model = BertForMaskedLM(config)
# torch_model = BertForMaskedLM(config)
# print(bf_model)
# print(torch_model)

# # Save torch BertEmbeddings to file
# save_path = "bertembeddings_torch.pt"
# torch.save(torch_embs.state_dict(), save_path)

# torch_embs.load_state_dict(torch.load(save_path))
# bf_embs.load_state_dict(torch.load(save_path))


# Establish data
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
text = ["hello I want to eat some [MASK] meat today. It's thanksgiving [MASK] all!", "yo yo what's up"]

# tokenize text and pass into model
tokens = tokenizer(text, return_tensors="pt", padding=True)
input_ids = tokens["input_ids"]
labels = torch.ones_like(input_ids)
# torch forward pass
# print(torch_model(input_ids))
# # bf forward pass
# jax_input_ids = jnp.array(input_ids.numpy(), dtype=int)
# out_bf = bf_embs(input_ids=jax_input_ids)
