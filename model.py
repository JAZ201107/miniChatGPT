from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embed: int = 384


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embed % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embed, config.n_embed * 3)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)

        self.n_head = config.n_head
        self.n_embed = config.n_embed

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)

        q, k, v = torch.chunk(qkv, 3, dim=-1)

        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * (k.size(-1) ** -0.5)
        attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)

        x = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        x = self.c_proj(x)
        return x


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, config.n_embed * 4)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(config.n_embed * 4, config.n_embed)

    def forward(self, x):
        x = self.gelu(self.c_fc(x))
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        config: GPTConfig,
    ):
        super().__init__()

        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embed),
                wpe=nn.Embedding(config.block_size, config.n_embed),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embed),
            )
        )

        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.size()

        assert (
            T <= self.config.block_size
        ), "Cannot forward, model block size is exhausted."

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer["wpe"](pos)
        tok_emb = self.transformer["wte"](idx)
        x = tok_emb + pos_emb

        for block in self.transformer["h"]:
            x = block(x)

        x = self.transformer["ln_f"](x)
        logits = self.lm_head(x)
        return logits  # (B, T, vocab_size)
