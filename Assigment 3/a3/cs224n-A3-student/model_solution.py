"""
A bare-bones GPT-2 style transformer.
"""

import math
from typing import Dict

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from jaxtyping import Float, Int
from torch.nn.functional import softmax
from dataclasses import dataclass
from einops import rearrange
from transformers import GPT2LMHeadModel
import huggingface_hub

from utils import state_dict_converter


# TODO: Add in attention mask to the entire assignment
# TODO: Maybe add KV caching


@dataclass
class ModelConfig:
    d_model: int
    n_heads: int
    n_layers: int
    context_length: int
    vocab_size: int


class CausalAttention(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()

        # Using attention dim from attention is all you need
        assert config.d_model % config.n_heads == 0
        self.n_heads = config.n_heads
        self.d_attention = int(config.d_model / config.n_heads)

        #self.c_attn = nn.Linear(config.d_model, 3 * config.d_model)

        self.W_k = nn.Linear(config.d_model, self.d_attention * config.n_heads)
        self.W_q = nn.Linear(config.d_model, self.d_attention * config.n_heads)
        self.W_v = nn.Linear(config.d_model, self.d_attention * config.n_heads)

        self.W_o = nn.Linear(self.d_attention * config.n_heads, config.d_model)

        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.context_length, config.context_length)).view(
                1, 1, config.context_length, config.context_length
            ),
            persistent=False
        )

    def forward(
        self, x: Float[Tensor, "batch seq_len d_model"]
    ) -> Float[Tensor, "batch seq_len d_model"]:

        batch, seq_len, _ = x.shape

        # Project the input into queries, keys and values, then split into heads.
        q = rearrange(self.W_q(x), "b t (h d) -> b h t d", h=self.n_heads)
        k = rearrange(self.W_k(x), "b t (h d) -> b h t d", h=self.n_heads)
        v = rearrange(self.W_v(x), "b t (h d) -> b h t d", h=self.n_heads)

        # Scaled dot-product attention scores: [b, h, t, t]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_attention)

        # Causal masking: a position may only attend to itself and the past.
        mask = self.causal_mask[:, :, :seq_len, :seq_len]
        scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = softmax(scores, dim=-1)

        # Weighted sum of the values, then merge the heads back together.
        out = attn @ v  # [b, h, t, d]
        out = rearrange(out, "b h t d -> b t (h d)")

        return self.W_o(out)



class GELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))  # fmt: skip

class MLP(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.fc1 = nn.Linear(config.d_model, 4 * config.d_model)
        self.fc2 = nn.Linear(4 * config.d_model, config.d_model)
        self.gelu = GELU()

    def forward(
        self, x: Float[Tensor, "batch seq_len d_model"]
    ) -> Float[Tensor, "batch seq_len d_model"]:

        return self.fc2(self.gelu(self.fc1(x)))


class DecoderBlock(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.mlp = MLP(config)
        self.attention = CausalAttention(config)
        self.pre_layer_norm = nn.LayerNorm(config.d_model)
        self.post_layer_norm = nn.LayerNorm(config.d_model)

    def forward(
        self, x: Float[Tensor, "batch seq_len d_model"]
    ) -> Float[Tensor, "batch seq_len d_model"]:

        # Pre-layernorm GPT-2 style residual block.
        x = x + self.attention(self.pre_layer_norm(x))
        x = x + self.mlp(self.post_layer_norm(x))
        return x


class Transformer(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config
        self.embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embeddings = nn.Embedding(config.context_length, config.d_model)
        self.backbone = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layers)])
        self.final_layer_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):

        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.ones_(module.weight)

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layers)
                )

    def forward(
        self, x: Int[Tensor, "batch_size seq_len"]
    ) -> Float[Tensor, "batch seq_len vocab_size"]:

        batch_size, seq_len = x.shape

        # Token embeddings + (learned) positional embeddings.
        positions = torch.arange(seq_len, device=x.device)
        h = self.embeddings(x) + self.position_embeddings(positions)

        # Pass through the stack of decoder blocks.
        for block in self.backbone:
            h = block(h)

        h = self.final_layer_norm(h)
        logits = self.lm_head(h)

        return logits

    @torch.no_grad()
    def generate(
        self,
        x: Int[Tensor, "batch_size seq_len"],
        num_new_tokens: int,
    ) -> Int[Tensor, "batch_size seq_len+num_new_tokens"]:

        for _ in range(num_new_tokens):
            # Never feed more than the context window into the model.
            x_cond = x[:, -self.config.context_length:]

            logits = self.forward(x_cond)

            # Greedy decoding: take the most likely next token.
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            x = torch.cat([x, next_token], dim=1)

        return x


    def get_loss_on_batch(
        self,
        input_ids: Int[Tensor, "batch_size seq_len"],
    ) -> Float[Tensor, ""]:

        logits = self.forward(input_ids)

        # Next-token prediction: align logits at position t with token at t+1.
        shifted_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
        shifted_targets = input_ids[:, 1:].reshape(-1)

        loss = F.cross_entropy(shifted_logits, shifted_targets)

        return loss


    @classmethod
    def from_pretrained(cls):
        """
        We simply always load up the GPT-2 model
        """

        # Config for GPT-2
        config = ModelConfig(
            d_model=768,
            n_heads=12,
            n_layers=12,
            context_length=1024,
            vocab_size=50257,
        )

        model = cls(config)

        # Load weights from HuggingFace
        model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
        converted_state_dict: Dict[str, Tensor] = state_dict_converter(model_hf.state_dict())

        model.load_state_dict(converted_state_dict)

        return model


if __name__ == "__main__":

    # Uncomment this if you are not logged in
    # huggingface_hub.login()
    
    model = Transformer.from_pretrained()
