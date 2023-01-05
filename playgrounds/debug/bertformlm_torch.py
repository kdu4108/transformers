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
torch_model = BertForMaskedLM.from_pretrained(model_id)

# Establish data
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
text = ["hello I want to eat some [MASK] meat today. It's thanksgiving [MASK] all!", "yo yo what's up"]
tokens = tokenizer(text, return_tensors="pt", padding=True)

# Create torch and bf inputs to model
input_ids_torch = tokens["input_ids"]
labels_torch = torch.ones_like(input_ids_torch)

# Forward pass
outputs_torch = torch_model(input_ids_torch)
print(outputs_torch)
