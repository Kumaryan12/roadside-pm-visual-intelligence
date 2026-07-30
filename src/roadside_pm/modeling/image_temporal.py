"""GRU/LSTM image-sequence regressors with optional late tabular fusion."""

from __future__ import annotations

import torch
from torch import nn


class ImageTemporalRegressor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, *, cell: str = "gru", hidden_dim: int = 128, num_layers: int = 1, dropout: float = 0.3, tabular_dim: int = 0, tabular_hidden_dim: int = 64):
        super().__init__()
        cell = cell.lower()
        recurrent = nn.GRU if cell == "gru" else nn.LSTM if cell == "lstm" else None
        if recurrent is None: raise ValueError("cell must be 'gru' or 'lstm'")
        self.recurrent = recurrent(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.tabular = None
        fusion_dim = hidden_dim
        if tabular_dim:
            self.tabular = nn.Sequential(nn.Linear(tabular_dim, tabular_hidden_dim), nn.ReLU(), nn.Dropout(dropout))
            fusion_dim += tabular_hidden_dim
        self.head = nn.Sequential(nn.LayerNorm(fusion_dim), nn.Dropout(dropout), nn.Linear(fusion_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, output_dim))

    def forward(self, image_sequence, tabular=None):
        output, _ = self.recurrent(image_sequence)
        representation = output[:, -1, :]
        if self.tabular is not None:
            if tabular is None: raise ValueError("tabular input is required")
            representation = torch.cat([representation, self.tabular(tabular)], dim=1)
        return self.head(representation)

