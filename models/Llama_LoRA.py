#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 11:28:43 2024

@author: umbertocappellazzo
"""

import torch 
import torch.nn as nn
from dataclasses import dataclass
from transformers.models.llama.modeling_llama import (
    LlamaForCausalLM, LlamaModel, LlamaDecoderLayer,
    LlamaAttention, apply_rotary_pos_emb, repeat_kv,
    ALL_ATTENTION_FUNCTIONS, eager_attention_forward,
)
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.cache_utils import Cache
from typing import Optional, Tuple, Callable
import math
from transformers.utils import logging

logger = logging.get_logger(__name__)

@dataclass
class LoRA_config:
    RANK: int
    ALPHA: int = 1
    IS_LLAMA3: bool = False
    IS_TINYLLAMA: bool = False
    
    

class LlamaSdpaAttention_lora(LlamaAttention):
    def __init__(self, config: LlamaConfig, lora_config: LoRA_config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        
        self.lora_rank = lora_config.RANK
        self.lora_scaling = lora_config.ALPHA / self.lora_rank
        
        hid_size = config.hidden_size
        lora_inner = round(hid_size / self.lora_rank)
        self.lora_down_Q = nn.Linear(hid_size, lora_inner, bias=False)
        self.lora_down_V = nn.Linear(hid_size, lora_inner, bias=False)
        self.lora_up_Q = nn.Linear(lora_inner, config.num_attention_heads * self.head_dim, bias=False)
        
        if lora_config.IS_LLAMA3:  # grouped query attention (GQA) in action!!
            self.lora_up_V = nn.Linear(lora_inner, config.num_key_value_heads * self.head_dim, bias=False)
        elif lora_config.IS_TINYLLAMA:
            self.lora_up_V = nn.Linear(lora_inner, config.num_key_value_heads * self.head_dim, bias=False)
        else:    
            self.lora_up_V = nn.Linear(lora_inner, config.num_key_value_heads * self.head_dim, bias=False)
        
        nn.init.zeros_(self.lora_down_Q.weight)
        nn.init.kaiming_uniform_(self.lora_up_Q.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_down_V.weight)
        nn.init.kaiming_uniform_(self.lora_up_V.weight, a=math.sqrt(5))
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # LoRA additions to Q and V
        Q_lora = self.lora_up_Q(self.lora_down_Q(hidden_states))
        V_lora = self.lora_up_V(self.lora_down_V(hidden_states))
        
        query_states = query_states + Q_lora * self.lora_scaling
        value_states = value_states + V_lora * self.lora_scaling

        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights

class LlamaForCausalLM_lora(LlamaForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]
    
    def __init__(self, config: LlamaConfig, lora_config: LoRA_config):
        super().__init__(config)
        self.lora_config= lora_config
        self.model = LlamaModel_lora(config, lora_config)
        
class LlamaModel_lora(LlamaModel):
    def __init__(self, config: LlamaConfig, lora_config: LoRA_config):
        super().__init__(config)
        self.lora_config= lora_config
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer_lora(config, layer_idx, lora_config) for layer_idx in range(config.num_hidden_layers)]
        )

class LlamaDecoderLayer_lora(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx, lora_config: LoRA_config):
        super().__init__(config, layer_idx)
        self.lora_config= lora_config
        
        self.self_attn = LlamaSdpaAttention_lora(config=config, layer_idx=layer_idx, lora_config=lora_config)
