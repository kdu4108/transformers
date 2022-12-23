from jax import numpy as jnp
from torch import Tensor
from torch.nn import Module
from brunoflow.ad import Node
from brunoflow.net import Network
from brunoflow.ad.utils import check_node_equals_tensor, check_node_allclose_tensor


def check_bf_param_weights_match_torch(bf_network: Network, torch_module: Module):
    """Used to verify the weights of the bf model and torch module are equal."""
    bf_params = {name: param for name, param in bf_network.named_parameters()}
    torch_params = {name: param for name, param in torch_module.named_parameters()}
    assert set(bf_params.keys()) == set(
        torch_params.keys()
    ), f"BF and torch keys do not match: BF contains following extra keys {set(bf_params.keys()).difference(set(torch_params.keys()))} and is missing keys {set(torch_params.keys()).difference(set(bf_params.keys()))}"

    for name in bf_params.keys():
        print(
            f"Value of param weight {name} for bf and torch are equal? {check_node_equals_tensor(bf_params[name], torch_params[name])}"
        )
        assert check_node_equals_tensor(
            bf_params[name], torch_params[name]
        ), f"Value of param {name} for bf and torch are not equal."


def check_bf_model_outputs_match_torch_outputs(out_bf: Node, out_torch: Tensor, atol=1e-8):
    """Used to verify the output of a bf model and torch module are close."""
    bf_allclose_torch = check_node_allclose_tensor(out_bf, out_torch, atol=atol)
    print(f"Output of bf and torch are within {atol}? {bf_allclose_torch}")
    assert bf_allclose_torch


def check_bf_param_grads_allclose_torch(bf_network: Network, torch_module: Module, atol=1e-6, print_output=False):
    """Used to verify that grad after backward passes for bf and torch are close for all params in the network."""
    bf_params = {name: param for name, param in bf_network.named_parameters()}
    torch_params = {name: param for name, param in torch_module.named_parameters()}
    assert set(bf_params.keys()) == set(
        torch_params.keys()
    ), f"BF and torch keys do not match: BF contains following extra keys {set(bf_params.keys()).difference(set(torch_params.keys()))} and is missing keys {set(torch_params.keys()).difference(set(bf_params.keys()))}"

    for name in bf_params.keys():
        if print_output:
            print(
                f"Grad of param {name} for bf and torch are within {atol}? {jnp.allclose(bf_params[name].grad, torch_params[name].grad.numpy(), atol=atol)}"
            )
        assert jnp.allclose(
            bf_params[name].grad, torch_params[name].grad.numpy(), atol=atol
        ), f"Grad of param {name} for bf and torch are not within {atol}."
