import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from jax import numpy as jnp
import transformers
import torch
from transformers import (
    BertConfig,
    BertTokenizerFast,
    BertForMaskedLM,
    BfBertForMaskedLM,
)
from utils import check_bf_param_weights_match_torch

# Init torch and bf models
BF_FROM_MODEL_ID = False  # NOTE: BECAUSE THIS IS SUPER HACKY THIS SOMEWHAT DOES NOT WORK WHEN SET TO TRUE. FROM_PRETRAINED FOR BRUNOFLOW IS PROBABLY SOMEWHAT BROKEN, BUT AT LEAST THIS IS A WORKAROUND. Also it looks like the errors are only bounded by 0.01 :/.
TORCH_FROM_MODEL_ID = True
# model_id = "bert-base-uncased"
model_id = "google/bert_uncased_L-2_H-128_A-2"
config = BertConfig.from_pretrained(pretrained_model_name_or_path="../brunoflow/models/bert/config-tiny.json")

if TORCH_FROM_MODEL_ID:
    torch_model = BertForMaskedLM.from_pretrained(model_id)
else:
    torch_model = BertForMaskedLM(config)
if BF_FROM_MODEL_ID:
    bf_model = BfBertForMaskedLM.from_pretrained(model_id)
else:
    bf_model = BfBertForMaskedLM(config)

# Save torch BertForMLM to file
save_path = "bertformlm_torch.pt"
torch.save(torch_model.state_dict(), save_path)
# Load state dict for BertForMLM into BF and check weights, outputs, and backprop
if not BF_FROM_MODEL_ID:
    bf_model.load_state_dict(torch.load(save_path))

check_bf_param_weights_match_torch(bf_model, torch_model)  # YAY THIS WORKS!!!

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
outputs_torch = torch_model(input_ids_torch)
outputs_bf = bf_model(input_ids_bf)
