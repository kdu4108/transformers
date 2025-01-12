# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Brunoflow BERT model."""

import brunoflow as bf
from brunoflow.net import CrossEntropyLoss, Dropout, Embedding, LayerNorm, Linear, ModuleList, Network, Tanh
from jax import numpy as jnp
import numpy as np
import math
import os
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from ...activations_bf import ACT2FN
from ...modeling_bf_outputs import (
    BfBaseModelOutputWithPastAndCrossAttentions,
    BfBaseModelOutputWithPoolingAndCrossAttentions,
    BfMaskedLMOutput,
    BfSequenceClassifierOutput,
)
from ...modeling_bf_utils import BfPreTrainedModel
from ...bf_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_bert import BertConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "bert-base-uncased"
_CONFIG_FOR_DOC = "BertConfig"
_TOKENIZER_FOR_DOC = "BertTokenizer"

BERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "bert-base-uncased",
    "bert-large-uncased",
    "bert-base-cased",
    "bert-large-cased",
    "bert-base-multilingual-uncased",
    "bert-base-multilingual-cased",
    "bert-base-chinese",
    "bert-base-german-cased",
    "bert-large-uncased-whole-word-masking",
    "bert-large-cased-whole-word-masking",
    "bert-large-uncased-whole-word-masking-finetuned-squad",
    "bert-large-cased-whole-word-masking-finetuned-squad",
    "bert-base-cased-finetuned-mrpc",
    "bert-base-german-dbmdz-cased",
    "bert-base-german-dbmdz-uncased",
    "cl-tohoku/bert-base-japanese",
    "cl-tohoku/bert-base-japanese-whole-word-masking",
    "cl-tohoku/bert-base-japanese-char",
    "cl-tohoku/bert-base-japanese-char-whole-word-masking",
    "TurkuNLP/bert-base-finnish-cased-v1",
    "TurkuNLP/bert-base-finnish-uncased-v1",
    "wietsedv/bert-base-dutch-cased",
    # See all BERT models at https://huggingface.co/models?filter=bert
]


def load_tf_weights_in_bert(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    try:
        import re

        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info(f"Loading TF weight {name} with shape {shape}")
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
            for n in name
        ):
            logger.info(f"Skipping {'/'.join(name)}")
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "output_weights":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info(f"Skipping {'/'.join(name)}")
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.transpose(array)
        try:
            if pointer.shape != array.shape:
                raise ValueError(f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched")
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info(f"Initialize PyTorch weight {name}")
        pointer.data = torch.from_numpy(array)
    return model


class BfBertEmbeddings(Network):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id, extra_name="word_embeddings"
        )
        self.position_embeddings = Embedding(
            config.max_position_embeddings, config.hidden_size, extra_name="position_embeddings"
        )
        self.token_type_embeddings = Embedding(
            config.type_vocab_size, config.hidden_size, extra_name="token_type_embeddings"
        )

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, extra_name="in BfBertEmbeddings")
        self.dropout = Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer(
            "position_ids",
            bf.Parameter(jnp.expand_dims(jnp.arange(config.max_position_embeddings), axis=0), name="position_ids"),
        )
        self.register_buffer(
            "token_type_ids",
            bf.Parameter(jnp.zeros(self.position_ids.shape, dtype=jnp.int64), name="position_ids"),
            persistent=False,
        )  # todo is this 64 bit necessary?

    def forward(
        self,
        input_ids: Optional[bf.Node] = None,
        token_type_ids: Optional[bf.Node] = None,
        position_ids: Optional[bf.Node] = None,
        inputs_embeds: Optional[bf.Node] = None,
        past_key_values_length: int = 0,
    ) -> bf.Node:
        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = bf.repeat(buffered_token_type_ids, n=input_shape[0], axis=0)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = jnp.zeros(input_shape)  # dtype=torch.long

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BfBertSelfAttention(Network):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(
            config.hidden_size / config.num_attention_heads
        )  # attention head size if a function of hidden size and num attention heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size, extra_name="query")
        self.key = Linear(config.hidden_size, self.all_head_size, extra_name="key")
        self.value = Linear(config.hidden_size, self.all_head_size, extra_name="value")

        self.dropout = Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x: bf.Node) -> bf.Node:
        # x.shape = (bs, seq_len, hidden_sz)
        new_x_shape = x.shape[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )  # shape=(bs, seq_len, num_attn_heads, attn_head_size) (where num_attn_heads, attn_head_size == hidden_size)
        # x = x.view(new_x_shape)
        x = bf.reshape(x, new_x_shape)
        return bf.transpose(
            x, axes=(0, 2, 1, 3)
        )  # shape=(bs, num_attn_heads, seq_len, attn_head_size) - you can slice out the (attn_head_size,) matrix for each token for each head for each sentence and analyze paths from there.

    def forward(
        self,
        hidden_states: bf.Node,
        attention_mask: Optional[bf.Node] = None,  # float
        head_mask: Optional[bf.Node] = None,
        encoder_hidden_states: Optional[bf.Node] = None,
        encoder_attention_mask: Optional[bf.Node] = None,
        past_key_value: Optional[Tuple[Tuple[bf.Node]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[bf.Node]:
        mixed_query_layer = self.query(hidden_states)
        mixed_query_layer.name = "bertselfattention mixed_query_layer"

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = bf.concat([past_key_value[0], key_layer], axis=2)
            value_layer = bf.concat([past_key_value[1], value_layer], axis=2)
        else:  # this is the branch for a normal BertMLM use
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            key_layer.name = "bertselfattention key"
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            value_layer.name = "bertselfattention value"

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = bf.matmul(
            query_layer, bf.matrix_transpose(key_layer)
        )  # matrix_transpose switches the last two axes

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if not isinstance(key_length, int):
                raise ValueError(
                    f"Uhoh! When converting from torch to bf we expected key_length to be an int, but instead received type {type(key_length)}."
                )
            if use_cache:
                position_ids_l = bf.Parameter(
                    jnp.array([[key_length - 1]], dtype=jnp.int64), name="position_ids_l"
                )  # dtype=long - this basically does an expand_dim. Not sure whether the expand_dim/view ought to be tracked in the graph though
            else:
                position_ids_l = bf.Parameter(
                    jnp.expand_dims(jnp.arange(query_length, dtype=jnp.int64), axis=1), name="position_ids_l"
                )
            position_ids_r = bf.Parameter(
                jnp.expand_dims(jnp.arange(key_length, dtype=jnp.int64), axis=1), name="position_ids_r"
            )
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            # positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = bf.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = bf.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = bf.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = bf.softmax(attention_scores, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = bf.matmul(attention_probs, value_layer)

        context_layer = bf.transpose(context_layer, axes=(0, 2, 1, 3))
        new_context_layer_shape = context_layer.shape[:-2] + (self.all_head_size,)
        context_layer = bf.reshape(context_layer, new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class BfBertSelfOutput(Network):
    def __init__(self, config):
        super().__init__()
        self.dense = Linear(config.hidden_size, config.hidden_size, extra_name="in BfBertSelfOutput")
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, extra_name="in BfBertSelfOutput")
        self.dropout = Dropout(config.hidden_dropout_prob, extra_name="in BfBertSelfOutput")

    def forward(self, hidden_states: bf.Node, input_tensor: bf.Node) -> bf.Node:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        hidden_states.name = "combine self_attention_output and bert attention input " + str(hash(self))
        hidden_states = self.LayerNorm(hidden_states)

        return hidden_states


class BfBertAttention(Network):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        self.self = BfBertSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = BfBertSelfOutput(config)
        self.pruned_heads = set()

    # def prune_heads(self, heads): TODO(KD): implement this when we want to do pruning experiments!
    #     if len(heads) == 0:
    #         return
    #     heads, index = find_pruneable_heads_and_indices(
    #         heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
    #     )

    #     # Prune linear layers
    #     self.self.query = prune_linear_layer(self.self.query, index)
    #     self.self.key = prune_linear_layer(self.self.key, index)
    #     self.self.value = prune_linear_layer(self.self.value, index)
    #     self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

    #     # Update hyper params and store pruned heads
    #     self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
    #     self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
    #     self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: bf.Node,
        attention_mask: Optional[bf.Node] = None,
        head_mask: Optional[bf.Node] = None,
        encoder_hidden_states: Optional[bf.Node] = None,
        encoder_attention_mask: Optional[bf.Node] = None,
        past_key_value: Optional[Tuple[Tuple[bf.Node]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[bf.Node]:
        hidden_states.name = "input to bertattention " + str(hash(self))
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )  # this calls selfattention (orange "Multi-Head attention") on the hidden states/input embeddings
        attention_output = self.output(
            self_outputs[0], hidden_states
        )  # this combines the hidden states/input embeddings with the selfattention outputs and then normalizes
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BfBertIntermediate(Network):
    def __init__(self, config):
        super().__init__()
        self.dense = Linear(config.hidden_size, config.intermediate_size, extra_name="in BfBertIntermediate")
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: bf.Node) -> bf.Node:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BfBertOutput(Network):
    def __init__(self, config):
        super().__init__()
        self.dense = Linear(config.intermediate_size, config.hidden_size, extra_name="in BfBertOutput")
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, extra_name="in BfBertOutput")
        self.dropout = Dropout(config.hidden_dropout_prob, extra_name="in BfBertOutput")

    def forward(self, hidden_states: bf.Node, input_tensor: bf.Node) -> bf.Node:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # hidden_states.name = "add_and_norm attn_output and input tensor " + str(hash(self))
        return hidden_states


class BfBertLayer(Network):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BfBertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = BfBertAttention(config, position_embedding_type="absolute")
        self.intermediate = BfBertIntermediate(config)
        self.output = BfBertOutput(config)

    def forward(
        self,
        hidden_states: bf.Node,
        attention_mask: Optional[bf.Node] = None,
        head_mask: Optional[bf.Node] = None,
        encoder_hidden_states: Optional[bf.Node] = None,
        encoder_attention_mask: Optional[bf.Node] = None,
        past_key_value: Optional[Tuple[Tuple[bf.Node]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[bf.Node]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:  # TODO(KD): pay more attention and make sure this works with decoders
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        # TODO(KD): fix this so that we know apply_chunking_to_forward works when there's chunks (chunk_size>0).
        if self.chunk_size_feed_forward > 0:
            print("WARNING: self.chunk_size_feed_forward > 0, which we haven't accounted for in BF.")

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(
            attention_output
        )  # this is basically applying the (lilac) feed forward layer after the attention has been combined with the input embeddings
        layer_output = self.output(
            intermediate_output, attention_output
        )  # attention_output is sorta like a skip layer here - it gets added with intermediate output and then normalized
        return layer_output


class BfBertEncoder(Network):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = ModuleList([BfBertLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: bf.Node,
        attention_mask: Optional[bf.Node] = None,
        head_mask: Optional[bf.Node] = None,
        encoder_hidden_states: Optional[bf.Node] = None,
        encoder_attention_mask: Optional[bf.Node] = None,
        past_key_values: Optional[Tuple[Tuple[bf.Node]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[bf.Node], BfBaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                raise NotImplementedError("Gradient checkpointing is not implemented with brunoflow.")
                # if use_cache:
                #     logger.warning(
                #         "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                #     )
                #     use_cache = False

                # def create_custom_forward(module):
                #     def custom_forward(*inputs):
                #         return module(*inputs, past_key_value, output_attentions)

                #     return custom_forward

                # layer_outputs = torch.utils.checkpoint.checkpoint(
                #     create_custom_forward(layer_module),
                #     hidden_states,
                #     attention_mask,
                #     layer_head_mask,
                #     encoder_hidden_states,
                #     encoder_attention_mask,
                # )

            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BfBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class BfBertPooler(Network):
    def __init__(self, config):
        super().__init__()
        self.dense = Linear(config.hidden_size, config.hidden_size, extra_name="in BfBertPooler")
        self.activation = Tanh()
        print("WARNING: TODO(KD) - when this is used, write some tests for this!")

    def forward(self, hidden_states: bf.Node) -> bf.Node:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BfBertPredictionHeadTransform(Network):
    def __init__(self, config):
        super().__init__()
        self.dense = Linear(config.hidden_size, config.hidden_size, extra_name="in BfBertPredictionHeadTransform")
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: bf.Node) -> bf.Node:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BfBertLMPredictionHead(Network):
    def __init__(self, config):
        super().__init__()
        self.transform = BfBertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = Linear(config.hidden_size, config.vocab_size, bias=False, extra_name="in BfBertLMPredictionHead")

        self.bias = bf.Parameter(jnp.zeros(config.vocab_size), name="BfBertLMPredictionHead bias")

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BfBertOnlyMLMHead(Network):
    def __init__(self, config):
        super().__init__()
        self.predictions = BfBertLMPredictionHead(config)

    def forward(self, sequence_output: bf.Node) -> bf.Node:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BfBertPreTrainedModel(BfPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = BertConfig
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        # print(
        #     "WARNING: we don't actually initialize weights here! If we ever need to actually train we can properly implement this."
        # )
        if isinstance(module, Linear):
            module.set_weights(
                jnp.array(np.random.normal(loc=0.0, scale=self.config.initializer_range, size=module.weight.shape))
            )
            if module.bias is not None:
                module.set_bias(jnp.zeros_like(module.bias.val))
        elif isinstance(module, Embedding):
            module.weight.val = jnp.array(
                np.random.normal(loc=0.0, scale=self.config.initializer_range, size=module.weight.shape)
            )
            module._fill_padding_idx_with_zero()
        elif isinstance(module, LayerNorm):
            module.bias.val = jnp.zeros_like(module.bias.val)
            module.weight.val = jnp.ones_like(module.weight.val)
        return
        # raise NotImplementedError("This isn't implemented yet because we don't need it!")
        # if isinstance(module, Linear):
        #     # Slightly different from the TF version which uses truncated_normal for initialization
        #     # cf https://github.com/pytorch/pytorch/pull/5617
        #     module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        #     if module.bias is not None:
        #         module.bias.data.zero_()
        # elif isinstance(module, Embedding):
        #     module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        #     if module.padding_idx is not None:
        #         module.weight.data[module.padding_idx].zero_()
        # elif isinstance(module, LayerNorm):
        #     module.bias.data.zero_()
        #     module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, BfBertEncoder):
            module.gradient_checkpointing = value


BERT_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`BertConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

BERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`BertTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare Bert Model transformer outputting raw hidden-states without any specific head on top.",
    BERT_START_DOCSTRING,
)
class BfBertModel(BfBertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BfBertEmbeddings(config)
        self.encoder = BfBertEncoder(config)

        self.pooler = BfBertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(
                heads
            )  # TODO(KD): this is not yet implemented for BertAttention, so this will fail

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BfBaseModelOutputWithPoolingAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[bf.Node] = None,
        attention_mask: Optional[bf.Node] = None,
        token_type_ids: Optional[bf.Node] = None,
        position_ids: Optional[bf.Node] = None,
        head_mask: Optional[bf.Node] = None,
        inputs_embeds: Optional[bf.Node] = None,
        encoder_hidden_states: Optional[bf.Node] = None,
        encoder_attention_mask: Optional[bf.Node] = None,
        past_key_values: Optional[List[bf.Node]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[bf.Node], BfBaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = bf.Parameter(
                jnp.ones((batch_size, seq_length + past_key_values_length)), name="attention_mask"
            )  # KD: should this be a Node?

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = bf.repeat(buffered_token_type_ids, n=input_shape[0], axis=0)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = bf.Parameter(jnp.zeros(input_shape, dtype=jnp.int64), name="token_type_ids")

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: bf.Node = self.get_extended_attention_mask(
            attention_mask, input_shape
        )  # KD: implemented, except for when self.config.is_decoder=True

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.shape
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = bf.Parameter(jnp.ones(encoder_hidden_shape), name="encoder_attention_mask")
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask
            )  # TODO(KD): implement this when we need to use decoders!
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)  # TODO(KD): implement this!

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BfBaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


@add_start_docstrings("""Bert Model with a `language modeling` head on top.""", BERT_START_DOCSTRING)
class BfBertForMaskedLM(BfBertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias", r"cls.predictions.decoder.weight"]

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.bert = BfBertModel(config, add_pooling_layer=False)
        self.cls = BfBertOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BfMaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="'paris'",
        expected_loss=0.88,
    )
    def forward(
        self,
        input_ids: Optional[bf.Node] = None,
        attention_mask: Optional[bf.Node] = None,
        token_type_ids: Optional[bf.Node] = None,
        position_ids: Optional[bf.Node] = None,
        head_mask: Optional[bf.Node] = None,
        inputs_embeds: Optional[bf.Node] = None,
        encoder_hidden_states: Optional[bf.Node] = None,
        encoder_attention_mask: Optional[bf.Node] = None,
        labels: Optional[bf.Node] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[bf.Node], BfMaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(
                bf.reshape(prediction_scores, newshape=(-1, self.config.vocab_size)), bf.reshape(labels, newshape=-1)
            )

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return BfMaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        raise NotImplementedError(
            "prepare_inputs_for_generation is not needed for MLM, so it's not implemented or tested at the moment."
        )
        # input_shape = input_ids.shape
        # effective_batch_size = input_shape[0]

        # #  add a dummy token
        # if self.config.pad_token_id is None:
        #     raise ValueError("The PAD token should be defined for generation")

        # attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
        # dummy_token = torch.full(
        #     (effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
        # )
        # input_ids = torch.cat([input_ids, dummy_token], dim=1)

        # return {"input_ids": input_ids, "attention_mask": attention_mask}


class BfBertForSequenceClassification(BfBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BfBertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = Dropout(classifier_dropout)
        self.classifier = Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[bf.Node] = None,
        attention_mask: Optional[bf.Node] = None,
        token_type_ids: Optional[bf.Node] = None,
        position_ids: Optional[bf.Node] = None,
        head_mask: Optional[bf.Node] = None,
        inputs_embeds: Optional[bf.Node] = None,
        labels: Optional[bf.Node] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[bf.Node], BfSequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                self.config.problem_type = "single_label_classification"
                # if self.num_labels == 1:
                #     self.config.problem_type = "regression"
                # elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                #     self.config.problem_type = "single_label_classification"
                # else:
                #     self.config.problem_type = "multi_label_classification"

            loss_fct = CrossEntropyLoss()
            loss = loss_fct(bf.reshape(logits, newshape=(-1, self.num_labels)), bf.reshape(labels, newshape=-1))
            # if self.config.problem_type == "regression":
            #     loss_fct = MSELoss()
            #     if self.num_labels == 1:
            #         loss = loss_fct(logits.squeeze(), labels.squeeze())
            #     else:
            #         loss = loss_fct(logits, labels)
            # elif self.config.problem_type == "single_label_classification":
            #     loss_fct = CrossEntropyLoss()
            #     loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            # elif self.config.problem_type == "multi_label_classification":
            #     loss_fct = BCEWithLogitsLoss()
            #     loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return BfSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
