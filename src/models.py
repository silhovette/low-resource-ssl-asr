from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel


class MelCTCModel(nn.Module):
    def __init__(self, vocab_size: int, n_mels: int = 80, encoder_dim: int = 256, encoder_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        layers = []
        in_dim = n_mels
        for _ in range(encoder_layers):
            layers.append(nn.Linear(in_dim, encoder_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = encoder_dim
        self.encoder = nn.Sequential(*layers)
        self.ctc = nn.Linear(encoder_dim, vocab_size)

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(features)
        return self.ctc(hidden)


class SSLCTCModel(nn.Module):
    def __init__(
        self,
        pretrained_name: str,
        vocab_size: int,
        freeze_encoder: bool = True,
        hidden_layer: str = "last",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_name)
        self.hidden_layer = hidden_layer
        hidden_size = getattr(self.encoder.config, "hidden_size", 768)
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, vocab_size),
        )
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def _transformer_layers(self):
        """Return the ModuleList of transformer encoder layers.

        Works for wav2vec2, HuBERT, WavLM, and other similar HF models
        where the structure is: model.encoder.layers (ModuleList).
        """
        # The HuggingFace AutoModel stores the transformer inside .encoder
        if hasattr(self.encoder, "encoder") and hasattr(self.encoder.encoder, "layers"):
            return self.encoder.encoder.layers
        raise AttributeError(
            f"Cannot locate transformer layers in {type(self.encoder).__name__}. "
            f"Expected self.encoder.encoder.layers to be a ModuleList."
        )

    @property
    def num_transformer_layers(self) -> int:
        return len(self._transformer_layers())

    def unfreeze_transformer_layers(self, num_layers: int):
        """Unfreeze the top `num_layers` transformer encoder layers.

        This is the key method for fine-tuning experiments.  After a frozen
        warm-up, call ``model.unfreeze_transformer_layers(3)`` to make the top
        3 layers trainable, then continue training with a lower learning rate.
        """
        layers = self._transformer_layers()
        total = len(layers)
        if num_layers > total:
            raise ValueError(
                f"Requested {num_layers} layers but model only has {total}"
            )
        unfrozen = 0
        for i, layer in enumerate(layers):
            if i >= total - num_layers:
                for p in layer.parameters():
                    p.requires_grad = True
                unfrozen += 1
        return unfrozen

    def unfreeze_all(self):
        """Unfreeze every parameter in the encoder (full fine-tuning)."""
        for p in self.encoder.parameters():
            p.requires_grad = True
        return sum(1 for _ in self.encoder.parameters())

    def freeze_all(self):
        """Re-freeze every parameter in the encoder."""
        for p in self.encoder.parameters():
            p.requires_grad = False

    @property
    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def total_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, input_values: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        needs_hidden_states = self.hidden_layer not in {"last", "-1"}
        out = self.encoder(
            input_values=input_values,
            attention_mask=attention_mask,
            output_hidden_states=needs_hidden_states,
        )
        if self.hidden_layer in {"last", "-1"}:
            hidden = out.last_hidden_state
        elif self.hidden_layer == "mean_last_4":
            hidden = torch.stack(out.hidden_states[-4:], dim=0).mean(dim=0)
        else:
            layer_idx = int(self.hidden_layer)
            hidden = out.hidden_states[layer_idx]
        logits = self.proj(hidden)
        return logits
