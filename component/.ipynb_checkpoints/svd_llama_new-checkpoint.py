import math
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from transformers.utils import logging
from transformers import LlamaConfig

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    gather_indices = position_ids[:, None, :, None]  # [bs, 1, seq_len, 1]
    gather_indices = gather_indices.repeat(1, cos.shape[1], 1, cos.shape[3])
    cos = torch.gather(cos.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
    sin = torch.gather(sin.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class SVDLinear(nn.Module):
    def __init__(self, in_features, out_features, ratio=1.0, bias=False):
        super().__init__()
        low_rank = int((in_features * out_features * ratio) / (in_features + out_features))
        self.u_proj = nn.Linear(low_rank, out_features, bias=bias)
        self.v_proj = nn.Linear(in_features, low_rank, bias=bias)

    def forward(self, x):
        return self.u_proj(self.v_proj(x))

import torch
import torch.nn as nn


# 自定义学习率调度器
class DecayingCosineAnnealingLR:
    def __init__(self, optimizer, T_max, eta_min=0, decay_factor=0.9):
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.decay_factor = decay_factor
        self.last_epoch = 0
        self.base_lr = [group['lr'] for group in optimizer.param_groups]

    def step(self):
        self.last_epoch += 1
        if self.last_epoch % self.T_max == 0:
            # 每个周期结束后，降低最大学习率
            for i, group in enumerate(self.optimizer.param_groups):
                group['lr'] = self.base_lr[i] * self.decay_factor
            self.base_lr = [group['lr'] for group in self.optimizer.param_groups]

        # 计算当前的学习率
        for i, group in enumerate(self.optimizer.param_groups):
            group['lr'] = self.eta_min + (self.base_lr[i] - self.eta_min) * \
                           (1 + math.cos(math.pi * (self.last_epoch % self.T_max) / self.T_max)) / 2


class SVDLinear_mine(nn.Module):
    def __init__(self, in_features, out_features, ratio=1.0, bias=False, n=6.0):
        super().__init__()
        low_rank = max(1, int((in_features * out_features * ratio) / (in_features + out_features)))  # 避免 low_rank=0
        self.n = float(n)  # 确保 n 是 float，避免 PyTorch 类型转换问题
        
        # 低秩近似参数，修正 v_proj 的维度
        self.u_proj = nn.Parameter(torch.randn(out_features, low_rank))  # (out_features, low_rank)
        self.v_proj = nn.Parameter(torch.randn(in_features, low_rank))  # (in_features, low_rank)

    def forward(self, x):
        """
        计算: y = (sin(nU) * cos(nV)^T + cos(nU) * sin(nV)^T) * x
        避免显式构造 W
        """
        orig_shape = x.shape  # 记录原始形状，方便恢复
        if x.dim() == 3:
            batch_size, seq_len, in_dim = x.shape  # 处理三维情况
            x = x.view(-1, in_dim)  # 变为 (batch_size * seq_len, in_dim)
        elif x.dim() == 2:
            batch_size, in_dim = x.shape  # 处理二维情况
        else:
            raise ValueError(f"Unexpected input shape {x.shape}")

        # 确保 in_dim 与 v_proj.shape[0] 一致
        assert in_dim == self.v_proj.shape[0], f"输入维度 {in_dim} 与 v_proj 期望的 {self.v_proj.shape[0]} 不匹配"

        # 计算 V 方向的投影
        v_x_cos = (torch.cos(self.n * self.v_proj).T @ x.T)  # (low_rank, in_features) @ (in_features, batch_size) -> (low_rank, batch_size)
        v_x_sin = (torch.sin(self.n * self.v_proj).T @ x.T)  # (low_rank, in_features) @ (in_features, batch_size) -> (low_rank, batch_size)

        # 计算 U 方向的投影
        y1 = torch.sin(self.n * self.u_proj) @ v_x_cos  # (out_features, low_rank) @ (low_rank, batch_size) -> (out_features, batch_size)
        y2 = torch.cos(self.n * self.u_proj) @ v_x_sin  # (out_features, low_rank) @ (low_rank, batch_size) -> (out_features, batch_size)

        y = (y1 + y2).T  # 还原 batch 维度

        # 恢复原始输入形状
        if len(orig_shape) == 3:
            y = y.view(orig_shape[0], orig_shape[1], -1)  # 恢复 (batch_size, seq_len, out_features)

        return y



class SVD_LlamaMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str, ratio=1.0):
        super().__init__()
        self.ratio = ratio
        self.up_proj = SVDLinear_mine(hidden_size, intermediate_size, ratio)
        self.gate_proj = SVDLinear_mine(hidden_size, intermediate_size, ratio)
        self.down_proj = SVDLinear_mine(intermediate_size, hidden_size, ratio)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        up = self.up_proj(x)
        gate = self.gate_proj(x)
        return self.down_proj(self.act_fn(gate) * up)

class SVD_LlamaAttention(nn.Module):
    def __init__(self, config: LlamaConfig, ratio=1.0):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.ratio = ratio

        self.q_proj = SVDLinear_mine(self.hidden_size, self.num_heads * self.head_dim, ratio)
        self.k_proj = SVDLinear_mine(self.hidden_size, self.num_heads * self.head_dim, ratio)
        self.v_proj = SVDLinear_mine(self.hidden_size, self.num_heads * self.head_dim, ratio)
        self.o_proj = SVDLinear_mine(self.num_heads * self.head_dim, self.hidden_size, ratio)
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        past_key_value = (key_states, value_states) if use_cache else None

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device))

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
