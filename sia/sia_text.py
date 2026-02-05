import os
import logging
from collections import OrderedDict
from pkg_resources import packaging

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import torch.utils.checkpoint as checkpoint
import functools

logger = logging.getLogger(__name__)


# On P1, model extracted from https://huggingface.co/laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K
MODEL_PATH = 'https://huggingface.co/laion'
_MODELS = {
    "ViT-L/14": os.path.join(MODEL_PATH, "CLIP-ViT-L-14-DataComp.XL-s13B-b90K", "vit_l14_text.pth"),
    "ViT-B/16": os.path.join(MODEL_PATH, "CLIP-ViT-B-16-DataComp.XL-s13B-b90K", "vit_b16_text.pth"),
}


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

# PeFT: Added LoRA  
class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank=8, alpha=16):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = torch.nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x
        
class TransformerMLPwithLoRA(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.c_fc = nn.Linear(d_model, d_model * 4)
        self.gelu = QuickGELU()
        self.c_proj = nn.Linear(d_model * 4, d_model)
        
        self.det_lora_fc = LoRALayer(d_model, d_model * 4)
        self.det_lora_proj = LoRALayer(d_model * 4, d_model)
        
    def forward(self, x):
        x = self.c_fc(x) + self.det_lora_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x) + self.det_lora_proj(x)
        
        return x

class TransformerMLP(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.c_fc = nn.Linear(d_model, d_model * 4)
        self.gelu = QuickGELU()
        self.c_proj = nn.Linear(d_model * 4, d_model)
        
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, lora=False):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        if lora:
            self.mlp = TransformerMLPwithLoRA(d_model)
        else:
            self.mlp = TransformerMLP(d_model)
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None,
                 checkpoint_num: int = 0, lora=False):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask, lora=lora) for _ in range(layers)])

        self.checkpoint_num = checkpoint_num

    def forward(self, x: torch.Tensor):
        if self.checkpoint_num > 0:
            segments = min(self.checkpoint_num, len(self.resblocks))
            return checkpoint.checkpoint_sequential(self.resblocks, segments, x)
        else:
            return self.resblocks(x)


class CLIP_TEXT(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            context_length: int,
            vocab_size: int,
            transformer_width: int,
            transformer_heads: int,
            transformer_layers: int,
            checkpoint_num: int,
            lora=False,
        ):
        super().__init__()

        self.context_length = context_length

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
            checkpoint_num=checkpoint_num,
            lora=lora
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
    
    def no_weight_decay(self):
        return {'token_embedding', 'positional_embedding'}

    @functools.lru_cache(maxsize=None)
    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, text):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x


def clip_text_b16(
    embed_dim=512,
    context_length=77,
    vocab_size=49408,
    transformer_width=512,
    transformer_heads=8,
    transformer_layers=12,
    checkpoint_num=0,
    pretrained=True,
    lora=False,
):
    # raise NotImplementedError
    model = CLIP_TEXT(
        embed_dim,
        context_length,
        vocab_size,
        transformer_width,
        transformer_heads,
        transformer_layers,
        checkpoint_num,
        lora=lora,
    )
    # pretrained = _MODELS["ViT-B/16"]
    # logger.info(f"Load pretrained weights from {pretrained}")
    # state_dict = torch.load(pretrained, map_location='cpu')
    # model.load_state_dict(state_dict, strict=False)
    # return model.eval()
    if pretrained:
        if isinstance(pretrained, str) and pretrained != "bert-base-uncased":
            pretrained = _MODELS[pretrained]
        else:
            pretrained = _MODELS["ViT-B/16"]
        logger.info(f"Load pretrained weights from {pretrained}")
        state_dict = torch.load(pretrained, map_location='cpu')
        if context_length != state_dict["positional_embedding"].size(0):
            # assert context_length < state_dict["positional_embedding"].size(0), "Cannot increase context length."
            print(f"Resize positional embedding from {state_dict['positional_embedding'].size(0)} to {context_length}")
            if context_length < state_dict["positional_embedding"].size(0):
                state_dict["positional_embedding"] = state_dict["positional_embedding"][:context_length]
            else:
                state_dict["positional_embedding"] = F.pad(
                    state_dict["positional_embedding"],
                    (0, 0, 0, context_length - state_dict["positional_embedding"].size(0)),
                    value=0,
                )

        message = model.load_state_dict(state_dict, strict=False)
        print(f"Load pretrained weights from {pretrained}: {message}")
    return model.eval()


def clip_text_l14(
    embed_dim=768,
    context_length=77,
    vocab_size=49408,
    transformer_width=768,
    transformer_heads=12,
    transformer_layers=12,
    checkpoint_num=0,
    pretrained=True,
    lora=False,
):
    model = CLIP_TEXT(
        embed_dim,
        context_length,
        vocab_size,
        transformer_width,
        transformer_heads,
        transformer_layers,
        checkpoint_num,
        lora=lora,
    )
    if pretrained:
        if isinstance(pretrained, str) and pretrained != "bert-base-uncased":
            pretrained = _MODELS[pretrained]
        else:
            pretrained = _MODELS["ViT-L/14"]
        logger.info(f"Load pretrained weights from {pretrained}")
        state_dict = torch.load(pretrained, map_location='cpu')
        if context_length != state_dict["positional_embedding"].size(0):
            # assert context_length < state_dict["positional_embedding"].size(0), "Cannot increase context length."
            print(f"Resize positional embedding from {state_dict['positional_embedding'].size(0)} to {context_length}")
            if context_length < state_dict["positional_embedding"].size(0):
                state_dict["positional_embedding"] = state_dict["positional_embedding"][:context_length]
            else:
                state_dict["positional_embedding"] = F.pad(
                    state_dict["positional_embedding"],
                    (0, 0, 0, context_length - state_dict["positional_embedding"].size(0)),
                    value=0,
                )

        message = model.load_state_dict(state_dict, strict=False)
        print(f"Load pretrained weights from {pretrained}: {message}")
    return model.eval()


def clip_text_l14_336(
    embed_dim=768,
    context_length=77,
    vocab_size=49408,
    transformer_width=768,
    transformer_heads=12,
    transformer_layers=12,
    lora=False,
):
    raise NotImplementedError
    model = CLIP_TEXT(
        embed_dim,
        context_length,
        vocab_size,
        transformer_width,
        transformer_heads,
        transformer_layers,
        lora=lora,
    )
    pretrained = _MODELS["ViT-L/14_336"]
    logger.info(f"Load pretrained weights from {pretrained}")
    state_dict = torch.load(pretrained, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    return model.eval()


def build_clip(config):
    model_cls = config.text_encoder.clip_teacher
    model = eval(model_cls)()
    return model
