#!/usr/bin/env python
# coding: utf-8

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from jax.config import config

config.update("jax_enable_x64", True)
from dataclasses import is_dataclass
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
    BertSelfAttention,
    BfBertSelfAttention,
    BertSelfOutput,
    BfBertSelfOutput,
    BertAttention,
    BfBertAttention,
    BertLayer,
    BfBertLayer,
    BertEncoder,
    BfBertEncoder,
    BaseModelOutputWithPastAndCrossAttentions,
    BfBaseModelOutputWithPastAndCrossAttentions,
    BertForMaskedLM,
    BfBertForMaskedLM,
)
from brunoflow.ad.utils import check_node_equals_tensor, check_node_allclose_tensor
from utils import (
    check_bf_param_weights_match_torch,
    check_equivalent_class,
    check_dataclass_keys_match,
    check_model_outputs_allclose,
    check_bf_model_outputs_match_torch_outputs,
    check_bf_param_grads_allclose_torch,
)

torch.manual_seed(0)


# In[2]:


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


# In[3]:


# Establish data
tokenizer = BertTokenizerFast.from_pretrained(model_id)
text = ["hello I want to eat some [MASK] meat today. It's thanksgiving [MASK] all!", "yo yo what's up"]
tokens = tokenizer(text, return_tensors="pt", padding=True)

# Create torch and bf inputs to model
input_ids_torch = tokens["input_ids"]
labels_torch = torch.ones_like(input_ids_torch)

input_ids_bf = jnp.array(input_ids_torch.numpy())
labels_bf = jnp.array(labels_torch.numpy())

# In[7]:


# Save torch BertForMLM to file
save_path = "bertformlm_torch.pt"
torch.save(torch_model.state_dict(), save_path)


# In[8]:


# Load state dict for BertForMLM into BF and check weights, outputs, and backprop
if not BF_FROM_MODEL_ID:
    bf_model.load_state_dict(torch.load(save_path))


# ### Check weights of BF model and Torch model match exactly

# In[9]:


# Check weights match
check_bf_param_weights_match_torch(bf_model, torch_model)


# ### Check model output after forward pass matches for BF and Torch

# In[10]:


# Set all dropouts to 0
for name, module in torch_model.named_modules():
    if module._get_name() == "Dropout":
        print(name, module.p)
        module.p = 0
        print(name, module.p)


# In[11]:


# Check output from forward passes match for bf and torch
torch_model.train(False)
bf_model.train(False)

outputs_bf = bf_model(input_ids_bf)
outputs_torch = torch_model(input_ids_torch)

if isinstance(outputs_bf, (list, tuple)):
    assert len(outputs_bf) == len(outputs_torch)
    for i in range(len(outputs_bf)):
        out_bf, out_torch = outputs_bf[i], outputs_torch[i]
        check_bf_model_outputs_match_torch_outputs(out_bf, out_torch, atol=1e-6)
elif is_dataclass(outputs_bf):
    check_model_outputs_allclose(outputs_bf, outputs_torch, print_stats=True, atol=1e-2)
else:
    check_bf_model_outputs_match_torch_outputs(outputs_bf, outputs_torch, atol=1e-6)


# ### Check grad after backward pass matches for BF and torch

# In[12]:

# Torch backward pass
torch_model.train(True)

if isinstance(outputs_torch, (list, tuple)):
    assert len(outputs_bf) == len(outputs_torch)
    backprop_node_torch = outputs_torch[0]
elif is_dataclass(outputs_torch):
    backprop_node_torch = outputs_torch.logits
else:
    backprop_node_torch = outputs_torch

backprop_node_torch.backward(gradient=torch.ones_like(backprop_node_torch))


# In[13]:


# BF backward pass

if isinstance(outputs_bf, (list, tuple)):
    assert len(outputs_bf) == len(outputs_torch)
    backprop_node = outputs_bf[0]
elif is_dataclass(outputs_torch):
    backprop_node = outputs_bf.logits
else:
    backprop_node = outputs_bf

backprop_node.backprop(values_to_compute=("grad",))

# In[21]:


# param_name = 'bert.encoder.layer.0.intermediate.dense.weight'
param_name = "bert.encoder.layer.0.attention.self.key.bias"
torch_grad = dict(torch_model.named_parameters())[param_name].grad.numpy()
bf_grad = dict(bf_model.named_parameters())[param_name].grad
diff = (
    dict(torch_model.named_parameters())[param_name].grad.numpy() - dict(bf_model.named_parameters())[param_name].grad
)
max_diff_ind = diff.argmax()
max_diff = (
    dict(torch_model.named_parameters())[param_name].grad.numpy() - dict(bf_model.named_parameters())[param_name].grad
).max()
max_diff_ind = max_diff_ind.__array__()
dict(torch_model.named_parameters())[param_name].grad.flatten()[max_diff_ind], dict(bf_model.named_parameters())[
    param_name
].grad.flatten()[max_diff_ind], max_diff
-288036.53527605 * 1e-3

import numpy as np

np.allclose(torch_grad, bf_grad, rtol=1e-3, atol=0.1)


# In[15]:


# Run the actual check
check_bf_param_grads_allclose_torch(
    bf_model, torch_model, atol=0.1, print_output=True, print_stats=True, use_assert=True
)


# In[ ]:
