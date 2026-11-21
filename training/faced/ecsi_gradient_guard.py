#!/usr/bin/env python3
"""Conflict-aware encoder gradients for EC-SI.

The emotion head minimizes emotion cross-entropy.  The nuisance head minimizes
stimulus-identity cross-entropy.  The shared encoder instead receives the
emotion gradient plus the *negative* nuisance gradient.  When those two encoder
gradients conflict, only the adversarial component is projected; the emotion
gradient and both task-head gradients are left unchanged.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import torch
from torch import Tensor, nn


Gradient = Tensor | None


@dataclass(frozen=True)
class GuardDiagnostics:
    conflict: bool
    dot_before: float
    dot_after: float
    emotion_gradient_norm: float
    adversarial_gradient_norm_before: float
    adversarial_gradient_norm_after: float

    def to_dict(self) -> dict[str, float | bool]:
        return asdict(self)


def _materialize(parameters: Iterable[nn.Parameter]) -> list[nn.Parameter]:
    output = list(parameters)
    if len({id(parameter) for parameter in output}) != len(output):
        raise ValueError("parameter groups contain duplicate parameters")
    return output


def _dot(left: Sequence[Gradient], right: Sequence[Gradient]) -> Tensor:
    terms = [
        (a * b).sum()
        for a, b in zip(left, right, strict=True)
        if a is not None and b is not None
    ]
    if terms:
        return torch.stack(terms).sum()
    reference = next((item for item in (*left, *right) if item is not None), None)
    return torch.tensor(0.0, device=reference.device if reference is not None else "cpu")


def _squared_norm(gradients: Sequence[Gradient]) -> Tensor:
    terms = [(gradient * gradient).sum() for gradient in gradients if gradient is not None]
    if terms:
        return torch.stack(terms).sum()
    return torch.tensor(0.0)


def project_conflicting_gradient(
    emotion_gradients: Sequence[Gradient],
    adversarial_gradients: Sequence[Gradient],
    *,
    epsilon: float = 1e-12,
) -> tuple[list[Gradient], GuardDiagnostics]:
    """Project the adversarial gradient off the emotion gradient on conflict.

    A conflict is defined by a negative global dot product across the complete
    shared-encoder parameter vector.  Missing gradients stay missing.
    """

    if len(emotion_gradients) != len(adversarial_gradients):
        raise ValueError("gradient sequences must have identical lengths")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    dot_before_tensor = _dot(emotion_gradients, adversarial_gradients)
    emotion_norm_sq = _squared_norm(emotion_gradients).to(dot_before_tensor.device)
    adversarial_norm_sq = _squared_norm(adversarial_gradients).to(dot_before_tensor.device)
    conflict = bool(dot_before_tensor.detach().item() < 0.0 and emotion_norm_sq.detach().item() > epsilon)
    if conflict:
        coefficient = dot_before_tensor / (emotion_norm_sq + epsilon)
        projected = [
            None
            if adversarial is None
            else adversarial
            - coefficient * (emotion if emotion is not None else torch.zeros_like(adversarial))
            for emotion, adversarial in zip(emotion_gradients, adversarial_gradients, strict=True)
        ]
    else:
        projected = [None if gradient is None else gradient for gradient in adversarial_gradients]
    dot_after_tensor = _dot(emotion_gradients, projected)
    projected_norm_sq = _squared_norm(projected).to(dot_after_tensor.device)
    diagnostics = GuardDiagnostics(
        conflict=conflict,
        dot_before=float(dot_before_tensor.detach().cpu().item()),
        dot_after=float(dot_after_tensor.detach().cpu().item()),
        emotion_gradient_norm=float(torch.sqrt(emotion_norm_sq).detach().cpu().item()),
        adversarial_gradient_norm_before=float(
            torch.sqrt(adversarial_norm_sq).detach().cpu().item()
        ),
        adversarial_gradient_norm_after=float(
            torch.sqrt(projected_norm_sq).detach().cpu().item()
        ),
    )
    return projected, diagnostics


def _assign(parameter: nn.Parameter, gradient: Gradient) -> None:
    parameter.grad = None if gradient is None else gradient.detach().clone()


def guarded_adversarial_backward(
    emotion_loss: Tensor,
    nuisance_loss: Tensor,
    *,
    encoder_parameters: Iterable[nn.Parameter],
    emotion_head_parameters: Iterable[nn.Parameter],
    nuisance_head_parameters: Iterable[nn.Parameter],
    adversarial_strength: float,
    epsilon: float = 1e-12,
) -> GuardDiagnostics:
    """Populate gradients for one guarded EC-SI optimizer step.

    Call ``optimizer.zero_grad(set_to_none=True)`` before this function and
    ``optimizer.step()`` afterwards.  ``nuisance_loss`` must be computed without
    gradient reversal.
    """

    if adversarial_strength < 0:
        raise ValueError("adversarial_strength must be nonnegative")
    encoder = _materialize(encoder_parameters)
    emotion_head = _materialize(emotion_head_parameters)
    nuisance_head = _materialize(nuisance_head_parameters)
    groups = [*encoder, *emotion_head, *nuisance_head]
    if len({id(parameter) for parameter in groups}) != len(groups):
        raise ValueError("encoder and task-head parameter groups must be disjoint")

    emotion_all = torch.autograd.grad(
        emotion_loss,
        [*encoder, *emotion_head],
        retain_graph=True,
        allow_unused=True,
    )
    nuisance_all = torch.autograd.grad(
        nuisance_loss,
        [*encoder, *nuisance_head],
        allow_unused=True,
    )
    emotion_encoder = list(emotion_all[: len(encoder)])
    emotion_head_gradients = list(emotion_all[len(encoder) :])
    nuisance_encoder = list(nuisance_all[: len(encoder)])
    nuisance_head_gradients = list(nuisance_all[len(encoder) :])
    adversarial_encoder = [
        None if gradient is None else -float(adversarial_strength) * gradient
        for gradient in nuisance_encoder
    ]
    projected, diagnostics = project_conflicting_gradient(
        emotion_encoder, adversarial_encoder, epsilon=epsilon
    )
    for parameter, emotion_gradient, adversarial_gradient in zip(
        encoder, emotion_encoder, projected, strict=True
    ):
        if emotion_gradient is None:
            combined = adversarial_gradient
        elif adversarial_gradient is None:
            combined = emotion_gradient
        else:
            combined = emotion_gradient + adversarial_gradient
        _assign(parameter, combined)
    for parameter, gradient in zip(emotion_head, emotion_head_gradients, strict=True):
        _assign(parameter, gradient)
    for parameter, gradient in zip(nuisance_head, nuisance_head_gradients, strict=True):
        _assign(parameter, gradient)
    return diagnostics

