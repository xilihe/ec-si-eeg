#!/usr/bin/env python3
"""Emotion-conditional normalized kernel HSIC for EC-SI experiments."""

from __future__ import annotations

import torch
from torch import Tensor


def _center_kernel(kernel: Tensor) -> Tensor:
    return (
        kernel
        - kernel.mean(dim=0, keepdim=True)
        - kernel.mean(dim=1, keepdim=True)
        + kernel.mean()
    )


def normalized_hsic_rbf_identity(features: Tensor, identity: Tensor, eps: float = 1e-8) -> Tensor:
    if features.ndim != 2 or identity.ndim != 1 or len(features) != len(identity):
        raise ValueError("incompatible HSIC inputs")
    squared_distance = torch.cdist(features, features, p=2).square()
    off_diagonal = ~torch.eye(len(features), dtype=torch.bool, device=features.device)
    positive = squared_distance[off_diagonal & (squared_distance > 0)]
    if len(positive) == 0:
        return features.sum() * 0.0
    bandwidth_squared = positive.detach().median().clamp_min(eps)
    feature_kernel = torch.exp(-squared_distance / (2.0 * bandwidth_squared))
    identity_kernel = (identity[:, None] == identity[None, :]).to(features.dtype)
    feature_centered = _center_kernel(feature_kernel)
    identity_centered = _center_kernel(identity_kernel)
    numerator = (feature_centered * identity_centered).sum()
    denominator = torch.sqrt(
        feature_centered.square().sum() * identity_centered.square().sum()
    ).clamp_min(eps)
    return (numerator / denominator).clamp_min(0.0)


def conditional_hsic_loss(
    features: Tensor,
    emotion: Tensor,
    identity: Tensor,
    minimum_rows: int = 4,
) -> Tensor:
    if emotion.ndim != 1 or identity.ndim != 1 or len(features) != len(emotion):
        raise ValueError("incompatible conditional HSIC inputs")
    losses: list[Tensor] = []
    for label in torch.unique(emotion):
        mask = emotion == label
        if int(mask.sum()) < minimum_rows or len(torch.unique(identity[mask])) < 2:
            continue
        losses.append(normalized_hsic_rbf_identity(features[mask], identity[mask]))
    if not losses:
        return features.sum() * 0.0
    return torch.stack(losses).mean()
