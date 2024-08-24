import torch
from torch import nn
from .triton_functions import (triton_sdpa_flash_attention, triton_bmm,
                               triton_element_wise_multiplication)
from modelling_utils import repeat_kv, apply_rotary_pos_emb
import math
from transformers.models.mistral.modeling_mistral import MistralRotaryEmbedding


class MistralSdpaAttention_triton(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        self.q_weight = nn.Parameter(
            torch.randn(1, self.hidden_size, self.num_heads * self.head_dim).uniform_(-1, 1) / math.sqrt(
                self.hidden_size))
        self.k_weight = nn.Parameter(
            torch.randn(1, self.hidden_size, self.num_key_value_heads * self.head_dim).uniform_(-1, 1) / math.sqrt(
                self.hidden_size))
        self.v_weight = nn.Parameter(
            torch.randn(1, self.hidden_size, self.num_key_value_heads * self.head_dim).uniform_(-1, 1) / math.sqrt(
                self.hidden_size))
        self.o_weight = nn.Parameter(
            torch.randn(1, self.num_heads * self.head_dim, self.hidden_size).uniform_(-1, 1) / math.sqrt(
                self.num_heads * self.head_dim))
        self.rotary_emb = MistralRotaryEmbedding(config.head_dim,
                                                 max_position_embeddings=config.max_position_embeddings,
                                                 base=config.rope_theta)

    def forward(self, hidden_states, position_ids=None):
        bsz, q_len, _ = hidden_states.size()

        # Replace these with your own Triton kernel implementations
        query_states = triton_bmm(hidden_states, self.q_weight)
        key_states = triton_bmm(hidden_states, self.k_weight)
        value_states = triton_bmm(hidden_states, self.v_weight)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_output = triton_sdpa_flash_attention(query_states, key_states, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)

        # Replace this with your own Triton kernel for the output projection
        attn_output = triton_bmm(attn_output, self.o_weight)

        return attn_output


class Mistral_mlp_triton(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Parameter(
            torch.randn(1, self.hidden_size, self.intermediate_size).uniform_(-1, 1) / math.sqrt(self.hidden_size))
        self.up_proj = nn.Parameter(
            torch.randn(1, self.hidden_size, self.intermediate_size).uniform_(-1, 1) / math.sqrt(self.hidden_size))
        self.down_proj = nn.Parameter(
            torch.randn(1, self.intermediate_size, self.hidden_size).uniform_(-1, 1) / math.sqrt(
                self.intermediate_size))

    def forward(self, hidden_state):
        result_up = triton_bmm(hidden_state, self.up_proj)
        result_gate = triton_bmm(hidden_state, self.gate_proj, "silu")
        temp = triton_element_wise_multiplication(result_gate, result_up)
        result = triton_bmm(temp, self.down_proj)
        return result


class MistralRMSNorm_triton(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size).unsqueeze(0).unsqueeze(0))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = triton_element_wise_multiplication(hidden_states, torch.rsqrt(variance + self.variance_epsilon))
        return triton_element_wise_multiplication(self.weight, hidden_states.to(input_dtype))
