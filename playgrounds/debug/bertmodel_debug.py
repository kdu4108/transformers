import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from jax import numpy as jnp
import transformers
import torch
from transformers import (
    BertConfig,
    BertTokenizerFast,
    BertModel,
    BfBertModel,
)
from utils import check_bf_param_weights_match_torch

# Init torch and bf models
FROM_MODEL_ID = True
# model_id = "bert-base-uncased"
model_id = "google/bert_uncased_L-2_H-128_A-2"
if FROM_MODEL_ID:
    torch_model = BertModel.from_pretrained(model_id)
    bf_model = BfBertModel.from_pretrained(model_id)
else:
    config = BertConfig.from_pretrained(pretrained_model_name_or_path="../../brunoflow/models/bert/config-tiny.json")
    torch_model = BertModel(config)
    bf_model = BfBertModel(config)

# Establish data
tokenizer = BertTokenizerFast.from_pretrained(model_id)
text = ["hello I want to eat some [MASK] meat today. It's thanksgiving [MASK] all!", "yo yo what's up"]
tokens = tokenizer(text, return_tensors="pt", padding=True)

# Create torch and bf inputs to model
input_ids_torch = tokens["input_ids"]
labels_torch = torch.ones_like(input_ids_torch)

input_ids_bf = jnp.array(input_ids_torch.numpy())
labels_bf = jnp.array(labels_torch.numpy())

# Forward pass
outputs_torch = torch_model(input_ids_torch)
outputs_bf = bf_model(input_ids_bf)


outputs_torch.last_hidden_state.backward(gradient=torch.ones_like(outputs_torch.last_hidden_state))
