#!/usr/bin/env python3
"""Hierarchical V-REx penalties over subject and within-emotion stimulus environments."""

from __future__ import annotations

import torch
from torch import Tensor


def group_risk_variance(per_sample_loss: Tensor, group: Tensor) -> Tensor:
    means = [per_sample_loss[group == value].mean() for value in torch.unique(group)]
    if len(means) < 2:
        return per_sample_loss.sum() * 0.0
    return torch.stack(means).var(unbiased=False)


def hierarchical_rex_penalty(
    per_sample_loss: Tensor,
    emotion: Tensor,
    stimulus: Tensor,
    subject: Tensor,
) -> tuple[Tensor, Tensor]:
    if any(value.ndim != 1 for value in (per_sample_loss, emotion, stimulus, subject)):
        raise ValueError("hierarchical REx inputs must be vectors")
    if len({len(per_sample_loss), len(emotion), len(stimulus), len(subject)}) != 1:
        raise ValueError("hierarchical REx input lengths differ")
    subject_penalty = group_risk_variance(per_sample_loss, subject)
    stimulus_penalties: list[Tensor] = []
    for label in torch.unique(emotion):
        mask = emotion == label
        if len(torch.unique(stimulus[mask])) >= 2:
            stimulus_penalties.append(group_risk_variance(per_sample_loss[mask], stimulus[mask]))
    stimulus_penalty = (
        torch.stack(stimulus_penalties).mean()
        if stimulus_penalties
        else per_sample_loss.sum() * 0.0
    )
    return subject_penalty, stimulus_penalty
