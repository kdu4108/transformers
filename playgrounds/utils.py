from dataclasses import dataclass, fields, is_dataclass
from jax import numpy as jnp
import pandas as pd
from torch import Tensor
from torch.nn import Module
from brunoflow.ad import Node
from brunoflow.net import Network
from brunoflow.ad.utils import check_node_equals_tensor, check_node_allclose_tensor
from transformers.utils.generic import ModelOutput


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


def check_bf_buffers_match_torch(bf_network: Network, torch_module: Module):
    """Used to verify the weights of the bf model and torch module are equal."""
    bf_buffers = {name: param for name, param in bf_network.named_buffers()}
    torch_buffers = {name: param for name, param in torch_module.named_buffers()}
    assert set(bf_buffers.keys()) == set(
        torch_buffers.keys()
    ), f"BF and torch keys do not match: BF contains following extra keys {set(bf_buffers.keys()).difference(set(torch_buffers.keys()))} and is missing keys {set(torch_buffers.keys()).difference(set(bf_buffers.keys()))}"

    for name in bf_buffers.keys():
        print(
            f"Value of param weight {name} for bf and torch are equal? {check_node_equals_tensor(bf_buffers[name], torch_buffers[name])}"
        )
        assert check_node_equals_tensor(
            bf_buffers[name], torch_buffers[name]
        ), f"Value of param {name} for bf and torch are not equal."


def check_bf_model_outputs_match_torch_outputs(out_bf: Node, out_torch: Tensor, print_stats=False, atol=1e-8):
    """Used to verify the output of a bf model and torch module are close."""
    bf_allclose_torch = check_node_allclose_tensor(out_bf, out_torch, atol=atol)
    print(f"Output of bf and torch are within {atol}? {bf_allclose_torch}")
    if print_stats:
        diff = jnp.abs(out_bf.val - out_torch.detach().numpy())
        diff_df = pd.DataFrame(diff.ravel())
        print(f"\tStats on diff in outputs between bf and torch: {diff_df.describe()}")

    assert bf_allclose_torch


def check_bf_param_grads_allclose_torch(
    bf_network: Network, torch_module: Module, atol=1e-6, print_output=False, print_stats=True, use_assert=True
):
    """Used to verify that grad after backward passes for bf and torch are close for all params in the network."""
    bf_params = {name: param for name, param in bf_network.named_parameters()}
    torch_params = {name: param for name, param in torch_module.named_parameters()}
    assert set(bf_params.keys()) == set(
        torch_params.keys()
    ), f"BF and torch keys do not match: BF contains following extra keys {set(bf_params.keys()).difference(set(torch_params.keys()))} and is missing keys {set(torch_params.keys()).difference(set(bf_params.keys()))}"

    not_allclose_params = []
    for name in bf_params.keys():
        if torch_params[name].grad is None:
            bf_grad_is_zero = jnp.array_equal(bf_params[name].grad, jnp.zeros_like(bf_params[name].grad))
            print(f"No grad for param {name} for torch. BF grad is zero? {bf_grad_is_zero}")
            if not bf_grad_is_zero:
                not_allclose_params.append(name)
        else:
            is_allclose = jnp.allclose(bf_params[name].grad, torch_params[name].grad.numpy(), atol=atol)
            if print_output:
                print(f"Grad of param {name} for bf and torch are within {atol}? {is_allclose}")
            if not is_allclose:
                diff = jnp.abs(bf_params[name].grad - torch_params[name].grad.numpy())
                diff_df = pd.DataFrame(diff)
                not_allclose_params.append(name)
                if print_stats:
                    print(f"\tStats on diff in grad for {name} between bf and torch: {diff_df.describe()}")

    if use_assert:
        assert not not_allclose_params, f"Grad of params {not_allclose_params} for bf and torch are not within {atol}."


def check_equivalent_class(out_bf_model_output, out_torch_model_output):
    bf_name = type(out_bf_model_output).__name__
    torch_name = type(out_torch_model_output).__name__
    assert bf_name == "Bf" + torch_name  # all BF versions of torch HF classes are prefixed with "Bf".


def check_dataclass_keys_match(out_bf_model_output, out_torch_model_output):
    assert set([f.name for f in fields(out_bf_model_output)]) == set(
        [f.name for f in fields(out_torch_model_output)]
    ), f"Keys of f{out_bf_model_output} and f{out_torch_model_output} don't match.\nBF keys: {set(fields(out_bf_model_output))}.\nTorch keys: {set(fields(out_torch_model_output))}"


def check_dataclass_values_allclose(out_bf, out_torch, fieldname="root", print_stats=False, atol=1e-8):
    if isinstance(out_torch, Tensor) != isinstance(out_bf, Node):
        raise ValueError(
            f"BF ModelOutput does not equal Torch ModelOutput - type(out_torch) is {type(out_torch)} while type(out_bf) is {type(out_bf)}."
        )

    if isinstance(out_torch, Tensor) and isinstance(out_bf, Node):
        try:
            if print_stats:
                print(f"Checking diff between BF and torch for {fieldname}:")
            check_bf_model_outputs_match_torch_outputs(out_bf, out_torch, print_stats=print_stats, atol=atol)
        except AssertionError:
            raise AssertionError(f"Output value of {fieldname} for bf and torch are not within {atol}.")

    else:
        if is_dataclass(out_torch):
            for field in fields(out_torch):
                check_dataclass_values_allclose(
                    getattr(out_bf, field.name),
                    getattr(out_torch, field.name),
                    fieldname=field.name,
                    print_stats=print_stats,
                    atol=atol,
                )
        elif isinstance(out_torch, (list, tuple)):
            for i in range(len(out_torch)):
                check_dataclass_values_allclose(
                    out_bf[i], out_torch[i], fieldname=fieldname + ".tuple", print_stats=print_stats, atol=atol
                )
        else:
            if out_torch is not None:
                print(f"Comparing equality of torch object {type(out_torch)} with bf object {type(out_bf)}.")
            assert out_torch == out_bf


def check_model_outputs_allclose(
    out_bf_model_output: ModelOutput, out_torch_model_output: ModelOutput, print_stats=False, atol=1e-8
):
    check_dataclass_keys_match(out_bf_model_output, out_torch_model_output)
    check_equivalent_class(out_bf_model_output, out_torch_model_output)
    check_dataclass_values_allclose(
        out_bf_model_output, out_torch_model_output, fieldname="root", print_stats=print_stats, atol=atol
    )
