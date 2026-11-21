#!/usr/bin/env python3
"""Deterministic fixed-weight averaging for same-initialization EC-SI models."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping

import torch
from torch import Tensor


FIXED_LEFT_WEIGHT = 0.5


def state_dicts_identical(
    left: Mapping[str, Tensor], right: Mapping[str, Tensor]
) -> bool:
    """Return whether two state dictionaries have identical keys and tensors."""

    if tuple(left) != tuple(right):
        return False
    return all(
        left[key].shape == right[key].shape
        and left[key].dtype == right[key].dtype
        and torch.equal(left[key], right[key])
        for key in left
    )


def fixed_equal_weight_state_dict(
    dvms_state: Mapping[str, Tensor], dvsc_state: Mapping[str, Tensor]
) -> OrderedDict[str, Tensor]:
    """Average two compatible checkpoints with the frozen 1:1 soup rule.

    Floating tensors, including BatchNorm running statistics, are averaged.
    ``num_batches_tracked`` is an integer training counter unused at inference;
    its maximum is retained. Every other non-floating buffer must be identical.
    """

    if tuple(dvms_state) != tuple(dvsc_state):
        raise ValueError("component state dictionaries have different keys or key order")
    merged: OrderedDict[str, Tensor] = OrderedDict()
    for key in dvms_state:
        left = dvms_state[key]
        right = dvsc_state[key]
        if left.shape != right.shape:
            raise ValueError(f"state shape mismatch for {key}: {left.shape} != {right.shape}")
        if left.dtype != right.dtype:
            raise ValueError(f"state dtype mismatch for {key}: {left.dtype} != {right.dtype}")
        if left.is_floating_point() or left.is_complex():
            merged[key] = (
                left.detach() * FIXED_LEFT_WEIGHT
                + right.detach() * (1.0 - FIXED_LEFT_WEIGHT)
            ).clone()
        elif key.endswith("num_batches_tracked"):
            merged[key] = torch.maximum(left.detach(), right.detach()).clone()
        elif torch.equal(left, right):
            merged[key] = left.detach().clone()
        else:
            raise ValueError(f"non-floating state differs for {key}")
    return merged
