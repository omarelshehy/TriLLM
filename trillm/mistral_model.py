import torch
from torch import nn
from mistral_layers import MistralSdpaAttention_triton, Mistral_mlp_triton, MistralRMSNorm_triton
from trillm.triton_functions import triton_bmm
import math


class MistralDecoderLayer_triton(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = MistralSdpaAttention_triton(config=config, layer_idx=layer_idx)

        self.mlp = Mistral_mlp_triton(config)
        self.input_layernorm = MistralRMSNorm_triton(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MistralRMSNorm_triton(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, position_ids=None):
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_ids=position_ids)
        hidden_states = residual + hidden_states
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        return outputs


class MistralModel_triton(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [MistralDecoderLayer_triton
             (config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        self.norm = MistralRMSNorm_triton(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Parameter(
            torch.randn(1, config.hidden_size, config.vocab_size).uniform_(-1, 1) / math.sqrt(config.hidden_size))

    @torch.no_grad()
    def forward(self, input_ids):
        embed_tokens = self.embed_tokens(input_ids)
        output_layers = None
        for idx, layer in enumerate(self.layers):
            if idx == 0:
                output_layers = layer(embed_tokens)
            else:
                output_layers = layer(output_layers)
        norm_output = self.norm(output_layers)
        lm_output = triton_bmm(norm_output, self.lm_head)
        return lm_output
