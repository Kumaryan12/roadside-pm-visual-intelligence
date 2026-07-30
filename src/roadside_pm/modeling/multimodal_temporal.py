"""Gated multi-view image and tabular temporal regression models."""

from __future__ import annotations

import torch
from torch import nn


class GatedMultiViewTemporalRegressor(nn.Module):
    """Fuse lens embeddings at each timestep, then model the fused sequence.

    Each lens has its own projection layer. A learned softmax gate assigns a
    weight to each projected lens at every timestep. The fused image vector is
    concatenated with a projected tabular vector before temporal modelling.
    """

    def __init__(
        self,
        image_dim: int,
        tabular_dim: int,
        *,
        n_views: int = 3,
        image_hidden_dim: int = 128,
        tabular_hidden_dim: int = 64,
        temporal_hidden_dim: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if n_views < 1:
            raise ValueError("n_views must be positive")
        self.n_views = n_views
        self.image_projections = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(image_dim),
                nn.Linear(image_dim, image_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for _ in range(n_views)
        ])
        self.view_gate = nn.Sequential(
            nn.Linear(image_hidden_dim, max(image_hidden_dim // 2, 1)),
            nn.Tanh(),
            nn.Linear(max(image_hidden_dim // 2, 1), 1),
        )
        self.tabular_projection = nn.Sequential(
            nn.LayerNorm(tabular_dim),
            nn.Linear(tabular_dim, tabular_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.temporal = nn.GRU(
            image_hidden_dim + tabular_hidden_dim,
            temporal_hidden_dim,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(temporal_hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(temporal_hidden_dim, 1),
        )

    def forward(
        self,
        images: torch.Tensor,
        tabular: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Predict from ``[batch,time,view,image_dim]`` and tabular inputs."""
        if images.ndim != 4 or images.shape[2] != self.n_views:
            raise ValueError(
                f"Expected images [batch,time,{self.n_views},dim], got {tuple(images.shape)}"
            )
        projected = torch.stack([
            layer(images[:, :, view, :])
            for view, layer in enumerate(self.image_projections)
        ], dim=2)
        attention = torch.softmax(self.view_gate(projected).squeeze(-1), dim=2)
        fused_images = (projected * attention.unsqueeze(-1)).sum(dim=2)
        fused = torch.cat([fused_images, self.tabular_projection(tabular)], dim=-1)
        encoded, _ = self.temporal(fused)
        prediction = self.head(encoded[:, -1]).squeeze(-1)
        if return_attention:
            return prediction, attention
        return prediction
