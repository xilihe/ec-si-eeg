#!/usr/bin/env python3
"""Emotion-conditioned, different-video latent MixStyle without parameters."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


def emotion_conditioned_dual_video_mixstyle(
    features: Tensor,
    emotion: Tensor,
    video: Tensor,
    rng: np.random.Generator,
    probability: float = 0.5,
    alpha: float = 0.1,
    epsilon: float = 1e-6,
) -> tuple[Tensor, dict]:
    """Exchange group-level style only across same-emotion different videos.

    Statistics are detached, following the representation-level MixStyle
    convention. Groups with fewer than two samples are left unchanged.
    """
    if features.ndim != 2:
        raise ValueError("features must have shape [batch, feature]")
    if emotion.ndim != 1 or video.ndim != 1:
        raise ValueError("emotion and video must be one-dimensional")
    if len(features) != len(emotion) or len(features) != len(video):
        raise ValueError("features, emotion, and video must be aligned")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    if alpha <= 0 or epsilon <= 0:
        raise ValueError("alpha and epsilon must be positive")
    audit = {
        "applied": False,
        "eligible_emotions": 0,
        "eligible_video_groups": 0,
        "mixed_video_groups": 0,
        "all_partners_different_video": True,
        "all_partners_same_emotion": True,
        "lambda_min": None,
        "lambda_max": None,
    }
    if float(rng.random()) >= probability:
        return features, audit

    result = features.clone()
    lambdas: list[float] = []
    for label in sorted(int(x) for x in torch.unique(emotion).detach().cpu().tolist()):
        label_mask = emotion == label
        candidates = []
        for video_id in sorted(int(x) for x in torch.unique(video[label_mask]).detach().cpu().tolist()):
            mask = label_mask & (video == video_id)
            if int(mask.sum().item()) >= 2:
                candidates.append(video_id)
        if len(candidates) < 2:
            continue
        audit["eligible_emotions"] += 1
        audit["eligible_video_groups"] += len(candidates)
        order = np.asarray(candidates, dtype=np.int64)
        rng.shuffle(order)
        partners = np.roll(order, -1)
        for video_id, partner_id in zip(order.tolist(), partners.tolist()):
            if video_id == partner_id:
                audit["all_partners_different_video"] = False
                raise AssertionError("MixStyle partner retained the same video")
            source_mask = label_mask & (video == int(video_id))
            partner_mask = label_mask & (video == int(partner_id))
            source = features[source_mask]
            partner = features[partner_mask]
            source_mean = source.mean(dim=0, keepdim=True).detach()
            source_scale = source.var(dim=0, unbiased=False, keepdim=True).add(epsilon).sqrt().detach()
            partner_mean = partner.mean(dim=0, keepdim=True).detach()
            partner_scale = partner.var(dim=0, unbiased=False, keepdim=True).add(epsilon).sqrt().detach()
            mixing = float(rng.beta(alpha, alpha))
            mixed_mean = mixing * source_mean + (1.0 - mixing) * partner_mean
            mixed_scale = mixing * source_scale + (1.0 - mixing) * partner_scale
            result[source_mask] = (source - source_mean) / source_scale * mixed_scale + mixed_mean
            lambdas.append(mixing)
            audit["mixed_video_groups"] += 1
    audit["applied"] = audit["mixed_video_groups"] > 0
    if lambdas:
        audit["lambda_min"] = min(lambdas)
        audit["lambda_max"] = max(lambdas)
    if audit["mixed_video_groups"] != audit["eligible_video_groups"]:
        raise AssertionError("not every eligible video group was mixed")
    return result, audit


def stop_gradient_style_consistency_kl(
    style_logits: Tensor, original_logits: Tensor, temperature: float = 2.0
) -> Tensor:
    """Distil the unchanged-view prediction into the style-mixed view."""
    if style_logits.shape != original_logits.shape or style_logits.ndim != 2:
        raise ValueError("style and original logits must be aligned matrices")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return F.kl_div(
        F.log_softmax(style_logits / temperature, dim=1),
        F.softmax(original_logits.detach() / temperature, dim=1),
        reduction="batchmean",
    ) * (temperature * temperature)
