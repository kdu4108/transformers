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
    BertSelfAttention,
    BfBertSelfAttention,
)
from brunoflow.ad.utils import check_node_equals_tensor, check_node_allclose_tensor
from utils import check_bf_param_vals_match_torch

torch.manual_seed(0)

# Init torch and bf models
config = BertConfig.from_pretrained(
    pretrained_model_name_or_path="/home/kevin/code/rycolab/brunoflow/models/bert/config.json"
)
torch_bsa = BertSelfAttention(config)
bf_bsa = BfBertSelfAttention(config)
# Init inputs to bf and torch models
hidden_states = jnp.ones(shape=(2, 19, 768))
attention_mask = jnp.zeros(shape=(2, 1, 1, 19))

hidden_states_torch = torch.ones(size=(2, 19, 768))
attention_mask_torch = torch.zeros(size=(2, 1, 1, 19))
# Check that forward pass for bf works and matches output shape with torch
outputs_torch = torch_bsa(hidden_states_torch, attention_mask_torch)
outputs_bf = bf_bsa(hidden_states, attention_mask)
assert [out.shape for out in outputs_torch] == [out.shape for out in outputs_bf]
print([out.shape for out in outputs_bf])

# Save torch BertSelfAttention to file
save_path = "bertselfattn_torch.pt"
torch.save(torch_bsa.state_dict(), save_path)
# Load state dict for BertSelfAttention into BF and check weights, outputs, and backprop
bf_bsa.load_state_dict(torch.load(save_path))
# Check weights match
check_bf_param_vals_match_torch(bf_bsa, torch_bsa)
# Check output from forward passes match for bf and torch
torch_bsa.train(False)
outputs_bf = bf_bsa(hidden_states=hidden_states, attention_mask=attention_mask)
outputs_torch = torch_bsa(hidden_states=hidden_states_torch, attention_mask=attention_mask_torch)

assert len(outputs_bf) == len(outputs_torch)
print(len(outputs_bf))
for i in range(len(outputs_bf)):
    out_bf, out_torch = outputs_bf[i], outputs_torch[i]
    print(out_bf)
    print(out_torch)
    assert check_node_allclose_tensor(out_bf, out_torch)
