import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from jax import numpy as jnp
import transformers
import torch
from transformers import (
    BertTokenizerFast,
    BertForMaskedLM,
    BfBertForMaskedLM,
)
from utils import check_bf_param_weights_match_torch

# Init torch and bf models
model_id = "google/bert_uncased_L-2_H-128_A-2"
bf_model = BfBertForMaskedLM.from_pretrained(model_id)

# Establish data
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
text = ["hello I want to eat some [MASK] meat today. It's thanksgiving [MASK] all!", "yo yo what's up"]
tokens = tokenizer(text, return_tensors="pt", padding=True)

# Create torch and bf inputs to model
input_ids_torch = tokens["input_ids"]
labels_torch = torch.ones_like(input_ids_torch)

input_ids_bf = jnp.array(input_ids_torch.numpy())
labels_bf = jnp.array(labels_torch.numpy())

# Forward pass
outputs_bf = bf_model(input_ids_bf)
print(outputs_bf)
