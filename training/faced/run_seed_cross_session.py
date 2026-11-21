#!/usr/bin/env python3
"""Trial-safe cross-session SEED model evidence pipeline.

The protocol freezes ``shot`` as a window, not a trial: support/validation
windows are sampled from whole trials and a trial never crosses roles within
one split. All RG-TPC hyperparameters and prototypes use support/validation
only, and test labels are never used for tuning.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from seed_pipeline import (
    CLASSES,
    METHODS,
    SplitData,
    class_prototypes,
    metrics_for_predictions,
    parse_float_list,
    parse_id_list,
    softmax,
    split_hash,
    trial_prediction_rows,
    trial_roles,
    tune_rgtpc,
    window_split,

)
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed-dir", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--subjects", default="all")
    parser.add_argument("--target-sessions", default="2,3")
    parser.add_argument("--folds", default="1,2,3,4,5")
    parser.add_argument("--seeds", default="20260804")
    parser.add_argument("--models", default="conformer,post_mamba")
    parser.add_argument("--shots-per-class", type=int, default=20)
    parser.add_argument("--val-shots-per-class", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--sampling-rate", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--tau-grid", default="0.45,0.55,0.65,0.75")
    parser.add_argument("--alpha-grid", default="0,0.25,0.5,0.75,1.0,1.5")
    parser.add_argument("--split-seed", type=int, default=20260804)
    parser.add_argument("--transformer-depth", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=10)
    parser.add_argument("--adapter-depth", type=int, default=2)
    parser.add_argument("--adapter-init-scale", type=float, default=0.05)
    parser.add_argument(
        "--temporal-backbone",
        choices=["transformer", "long_bimamba", "multiscale_tcn", "hybrid_long"],
        default="transformer",
        help=(
            "Temporal encoder. The long-sequence variants preserve a higher-resolution "
            "time axis before sequence modeling; transformer reproduces the frozen baseline."
        ),
    )
    parser.add_argument(
        "--temporal-depth",
        type=int,
        default=3,
        help="Number of high-resolution long-sequence blocks.",
    )
    parser.add_argument(
        "--temporal-downsample",
        type=int,
        default=2,
        help="Temporal downsampling used by high-resolution sequence embeddings.",
    )
    parser.add_argument(
        "--temporal-kernels",
        default="3,7,15",
        help="Three odd kernel sizes used by the multi-scale temporal block.",
    )
    parser.add_argument(
        "--token-pooling",
        choices=["mean_max", "mean"],
        default="mean_max",
        help="Frozen temporal pooling rule for high-resolution sequence backbones.",
    )
    parser.add_argument("--mamba-d-state", type=int, default=16)
    parser.add_argument("--mamba-d-conv", type=int, default=4)
    parser.add_argument("--mamba-expand", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--clip-robust-z", type=float, default=20.0)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu", "mps"], default="auto")
    parser.add_argument("--de-features", action="store_true",
                        help="Use DE (differential entropy) features as dual-path spectral input.")
    parser.add_argument("--dann", action="store_true",
                        help="Enable Domain Adversarial training (subject-invariant features).")
    parser.add_argument("--dann-lambda", type=float, default=0.1,
                        help="Gradient reversal weight for DANN (default: 0.1).")
    parser.add_argument("--contrastive", action="store_true",
                        help="Enable supervised contrastive loss during source training.")
    parser.add_argument("--contrastive-alpha", type=float, default=0.5,
                        help="Contrastive loss weight (default: 0.5).")
    parser.add_argument("--contrastive-temperature", type=float, default=0.07,
                        help="Temperature for contrastive loss (default: 0.07).")
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="Cross-entropy label smoothing used in source and calibration training.")
    parser.add_argument("--calibration-augment", action="store_true",
                        help="Apply deterministic label-preserving augmentation to support windows only.")
    parser.add_argument("--augment-noise-std", type=float, default=0.01)
    parser.add_argument("--augment-amplitude", type=float, default=0.10)
    parser.add_argument("--augment-channel-dropout", type=float, default=0.10)
    parser.add_argument(
        "--calibration-strategy",
        choices=["head_ln", "last_block"],
        default="head_ln",
        help="Parameters adapted on target support windows.",
    )
    parser.add_argument(
        "--calibration-learning-rate",
        type=float,
        default=0.0,
        help="Target-calibration learning rate; 0 reuses --learning-rate.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def supcon_loss(features, labels, temperature=0.07):
    """Supervised Contrastive Loss (SupCon)."""
    features = F.normalize(features, dim=1)
    sim = (features @ features.T) / temperature
    labels = labels.unsqueeze(1)
    pos_mask = (labels == labels.T).float()
    pos_mask.fill_diagonal_(0)
    sim_max = sim.max(dim=1, keepdim=True)[0].detach()
    sim = sim - sim_max
    exp_sim = torch.exp(sim)
    num = (exp_sim * pos_mask).sum(dim=1)
    den = (exp_sim * (torch.ones_like(pos_mask) - torch.eye(len(features), device=features.device))).sum(dim=1)
    loss = -torch.log((num + 1e-12) / (den + 1e-12))
    pos_count = pos_mask.sum(dim=1).clamp(min=1)
    return (loss * (pos_count > 0).float()).sum() / pos_count.sum()

def resolve_device(value: str) -> torch.device:
    if value == "cpu":
        return torch.device("cpu")
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    if value == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but unavailable")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
class GradientReversalLayer(torch.autograd.Function):
    """Gradient reversal for domain adversarial training."""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class DomainClassifier(nn.Module):
    """Predicts subject ID from Conformer features for adversarial training."""
    def __init__(self, feature_dim: int, n_subjects: int = 15, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_subjects),
        )

    def forward(self, x):
        return self.net(x)


class SpectralEncoder(nn.Module):
    """Lightweight encoder for DE features: (batch, 62, 5) -> (batch, 40)."""
    def __init__(self, n_channels=62, n_bands=5, emb_size=40, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, emb_size, kernel_size=1),
            nn.BatchNorm1d(emb_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(emb_size, emb_size, kernel_size=n_bands),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(start_dim=1),
        )

    def forward(self, de):
        return self.net(de)


class PatchEmbedding(nn.Module):
    def __init__(self, n_channels: int, emb_size: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, emb_size, (1, 25)),
            nn.Conv2d(emb_size, emb_size, (n_channels, 1)),
            nn.BatchNorm2d(emb_size),
            nn.ELU(),
            nn.AvgPool2d((1, 25), (1, 5)),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.net(x)
        return x.flatten(start_dim=2).transpose(1, 2)


class FineTemporalEmbedding(nn.Module):
    """Spatially mix EEG channels while retaining a dense temporal token stream."""

    def __init__(
        self,
        n_channels: int,
        emb_size: int,
        dropout: float,
        downsample: int,
    ):
        super().__init__()
        if downsample < 1:
            raise ValueError("temporal_downsample must be >= 1")
        pooling: nn.Module
        if downsample == 1:
            pooling = nn.Identity()
        else:
            pooling = nn.AvgPool2d((1, downsample), (1, downsample))
        self.net = nn.Sequential(
            nn.Conv2d(1, emb_size, (n_channels, 1), bias=False),
            nn.Conv2d(
                emb_size,
                emb_size,
                (1, 15),
                padding=(0, 7),
                groups=emb_size,
                bias=False,
            ),
            nn.BatchNorm2d(emb_size),
            nn.ELU(),
            pooling,
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.net(x)
        return x.squeeze(2).transpose(1, 2)


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor) -> Tensor:
        queries = self.queries(x).view(x.shape[0], x.shape[1], self.num_heads, -1).transpose(1, 2)
        keys = self.keys(x).view(x.shape[0], x.shape[1], self.num_heads, -1).transpose(1, 2)
        values = self.values(x).view(x.shape[0], x.shape[1], self.num_heads, -1).transpose(1, 2)
        energy = torch.einsum("bhqd,bhkd->bhqk", queries, keys)
        attention = F.softmax(energy / math.sqrt(self.emb_size), dim=-1)
        output = torch.einsum("bhqk,bhkd->bhqd", self.att_drop(attention), values)
        return self.projection(output.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], -1))


class ResidualAdd(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return x + self.fn(x)


class FeedForward(nn.Sequential):
    def __init__(self, emb_size: int, dropout: float):
        super().__init__(
            nn.Linear(emb_size, 4 * emb_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * emb_size, emb_size),
        )


class TransformerBlock(nn.Sequential):
    def __init__(self, emb_size: int, num_heads: int, dropout: float):
        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    MultiHeadAttention(emb_size, num_heads, dropout),
                    nn.Dropout(dropout),
                )
            ),
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    FeedForward(emb_size, dropout),
                    nn.Dropout(dropout),
                )
            ),
        )


class MambaMixer(nn.Module):
    def __init__(self, emb_size: int, d_state: int, d_conv: int, expand: int):
        super().__init__()
        from mamba_ssm import Mamba

        self.forward_mamba = Mamba(d_model=emb_size, d_state=d_state, d_conv=d_conv, expand=expand)
        self.backward_mamba = Mamba(d_model=emb_size, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x: Tensor) -> Tensor:
        forward = self.forward_mamba(x)
        backward = torch.flip(self.backward_mamba(torch.flip(x, dims=[1])), dims=[1])
        return 0.5 * (forward + backward)


class PostMambaBlock(nn.Module):
    def __init__(
        self,
        emb_size: int,
        dropout: float,
        d_state: int,
        d_conv: int,
        expand: int,
        init_scale: float,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(emb_size)
        self.mamba = MambaMixer(emb_size, d_state, d_conv, expand)
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.scale * self.dropout(self.mamba(self.norm(x)))


class MultiScaleTemporalBlock(nn.Module):
    """Residual multi-kernel temporal block for short and long EEG patterns."""

    def __init__(self, emb_size: int, dropout: float, kernels: tuple[int, int, int] = (3, 7, 15)):
        super().__init__()
        if len(kernels) != 3 or any(kernel < 1 or kernel % 2 == 0 for kernel in kernels):
            raise ValueError("temporal_kernels must contain exactly three positive odd integers")
        self.norm = nn.LayerNorm(emb_size)
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(
                    emb_size,
                    emb_size,
                    kernel_size=kernel,
                    padding=kernel // 2,
                    groups=emb_size,
                    bias=False,
                )
                for kernel in kernels
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv1d(3 * emb_size, emb_size, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(emb_size),
            FeedForward(emb_size, dropout),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        normalized = self.norm(x).transpose(1, 2)
        temporal = self.fuse(torch.cat([branch(normalized) for branch in self.branches], dim=1))
        x = x + temporal.transpose(1, 2)
        return x + self.feed_forward(x)


class EmotionNet(nn.Module):
    def __init__(
        self,
        args: argparse.Namespace,
        model_name: str,
        n_channels: int,
        n_samples: int,
        n_classes: int,
        use_de_features: bool = False,
    ):
        super().__init__()
        emb_size = 40
        self.temporal_backbone = args.temporal_backbone
        self.uses_token_pooling = args.temporal_backbone != "transformer"
        self.token_pooling = getattr(args, "token_pooling", "mean_max")
        temporal_kernels = tuple(
            int(item.strip())
            for item in str(getattr(args, "temporal_kernels", "3,7,15")).split(",")
            if item.strip()
        )
        if self.uses_token_pooling:
            self.patch = FineTemporalEmbedding(
                n_channels,
                emb_size,
                args.dropout,
                args.temporal_downsample,
            )
            blocks: list[nn.Module] = []
            for _ in range(args.temporal_depth):
                if args.temporal_backbone in {"multiscale_tcn", "hybrid_long"}:
                    blocks.append(MultiScaleTemporalBlock(emb_size, args.dropout, temporal_kernels))
                if args.temporal_backbone in {"long_bimamba", "hybrid_long"}:
                    blocks.append(
                        PostMambaBlock(
                            emb_size,
                            args.dropout,
                            args.mamba_d_state,
                            args.mamba_d_conv,
                            args.mamba_expand,
                            1.0,
                        )
                    )
            self.transformer = nn.Sequential(*blocks)
            self.adapter = nn.Identity()
            self.feature_norm = nn.LayerNorm(emb_size)
        else:
            self.patch = PatchEmbedding(n_channels, emb_size, args.dropout)
            self.transformer = nn.Sequential(
                *[
                    TransformerBlock(emb_size, args.num_heads, args.dropout)
                    for _ in range(args.transformer_depth)
                ]
            )
            if model_name == "post_mamba":
                self.adapter = nn.Sequential(
                    *[
                        PostMambaBlock(
                            emb_size,
                            args.dropout,
                            args.mamba_d_state,
                            args.mamba_d_conv,
                            args.mamba_expand,
                            args.adapter_init_scale,
                        )
                        for _ in range(args.adapter_depth)
                    ]
                )
            else:
                self.adapter = nn.Identity()
        self.use_de_features = use_de_features
        if use_de_features:
            self.spectral_encoder = SpectralEncoder(n_channels, 5, 40, args.dropout)
            self.de_head = nn.Linear(40, n_classes)
        self.use_dann = getattr(args, 'dann', False)
        if self.uses_token_pooling:
            if self.token_pooling == "mean_max":
                classifier_input = 2 * emb_size
            elif self.token_pooling == "mean":
                classifier_input = emb_size
            else:
                raise ValueError(f"Unsupported token_pooling: {self.token_pooling}")
        else:
            token_count = ((n_samples - 25 + 1) - 25) // 5 + 1
            if token_count <= 0:
                raise ValueError(f"Input has too few samples: {n_samples}")
            classifier_input = token_count * emb_size
        if self.use_dann:
            self.domain_classifier = DomainClassifier(classifier_input, 15, args.dropout)
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input, 256),
            nn.ELU(),
            nn.Dropout(args.dropout),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes),
        )

    def encode_tokens(self, x: Tensor) -> Tensor:
        return self.adapter(self.transformer(self.patch(x)))

    def pooled_features(self, tokens: Tensor) -> Tensor:
        if self.uses_token_pooling:
            tokens = self.feature_norm(tokens)
            if self.token_pooling == "mean_max":
                return torch.cat([tokens.mean(dim=1), tokens.amax(dim=1)], dim=1)
            return tokens.mean(dim=1)
        return tokens.flatten(start_dim=1)

    def forward(self, x: Tensor, de: Tensor | None = None) -> Tensor:
        features = self.pooled_features(self.encode_tokens(x))
        logits = self.classifier(features)
        if self.use_de_features and de is not None:
            de_emb = self.spectral_encoder(de)
            de_logits = self.de_head(de_emb)
            logits = logits + de_logits
        return logits

    def extract_features(self, x: Tensor) -> Tensor:
        """Extract pre-classifier features for contrastive learning."""
        return self.pooled_features(self.encode_tokens(x))

    def domain_forward(self, x: Tensor, alpha: float = 1.0) -> Tensor:
        """Forward pass with domain prediction for adversarial training."""
        features = self.pooled_features(self.encode_tokens(x))
        reversed_features = GradientReversalLayer.apply(features, alpha)
        return self.domain_classifier(reversed_features)


class MaxNormLinear(nn.Linear):
    """Linear layer with the EEGNet output-weight max-norm constraint."""

    def __init__(self, in_features: int, out_features: int, max_norm: float = 0.25):
        super().__init__(in_features, out_features)
        self.max_norm = float(max_norm)

    def forward(self, x: Tensor) -> Tensor:
        norm = self.weight.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
        constrained = self.weight * (norm.clamp(max=self.max_norm) / norm)
        return F.linear(x, constrained, self.bias)


class EEGNetBaseline(nn.Module):
    """EEGNet-8,2 baseline adapted to the requested sampling rate."""

    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        n_classes: int,
        sampling_rate: int = 200,
    ):
        super().__init__()
        temporal_kernel = sampling_rate // 2
        separable_kernel = sampling_rate // 8
        f1, depth_multiplier, f2 = 8, 2, 16
        self.use_de_features = False
        self.use_dann = False
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                1,
                f1,
                kernel_size=(1, temporal_kernel),
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(f1),
            nn.Conv2d(
                f1,
                f1 * depth_multiplier,
                kernel_size=(n_channels, 1),
                groups=f1,
                bias=False,
            ),
            nn.BatchNorm2d(f1 * depth_multiplier),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.5),
            nn.Conv2d(
                f1 * depth_multiplier,
                f1 * depth_multiplier,
                kernel_size=(1, separable_kernel),
                padding="same",
                groups=f1 * depth_multiplier,
                bias=False,
            ),
            nn.Conv2d(f1 * depth_multiplier, f2, kernel_size=1, bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.5),
        )
        feature_dim = self._feature_dim(n_channels, n_samples)
        self.classifier = MaxNormLinear(feature_dim, n_classes, max_norm=0.25)

    def _feature_dim(self, n_channels: int, n_samples: int) -> int:
        was_training = self.feature_extractor.training
        self.feature_extractor.eval()
        with torch.no_grad():
            output = self.feature_extractor(torch.zeros(1, 1, n_channels, n_samples))
        self.feature_extractor.train(was_training)
        return int(output[0].numel())

    def extract_features(self, x: Tensor) -> Tensor:
        return self.feature_extractor(x).flatten(start_dim=1)

    def forward(self, x: Tensor, de: Tensor | None = None) -> Tensor:
        del de
        return self.classifier(self.extract_features(x))


class TSceptionBaseline(nn.Module):
    """Official TSception topology with sampling-rate-scaled windows."""

    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        n_classes: int,
        sampling_rate: int = 200,
    ):
        super().__init__()
        number_filter_temp = 9
        number_filter_spat = 6
        self.use_de_features = False
        self.use_dann = False
        self.temporal_branches = nn.ModuleList(
            [
                self._conv_block(
                    1,
                    number_filter_temp,
                    (1, int(window_seconds * sampling_rate)),
                    (1, 1),
                    8,
                )
                for window_seconds in (0.5, 0.25, 0.125)
            ]
        )
        self.temporal_norm = nn.BatchNorm2d(number_filter_temp)
        self.spatial_global = self._conv_block(
            number_filter_temp,
            number_filter_spat,
            (n_channels, 1),
            (1, 1),
            2,
        )
        half_channels = n_channels // 2
        self.spatial_hemisphere = self._conv_block(
            number_filter_temp,
            number_filter_spat,
            (half_channels, 1),
            (half_channels, 1),
            2,
        )
        self.spatial_norm = nn.BatchNorm2d(number_filter_spat)
        feature_dim = self._feature_dim(n_channels, n_samples)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_classes),
        )

    @staticmethod
    def _conv_block(
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        pool: int,
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(1, pool), stride=(1, pool)),
        )

    def _spatial_features(self, x: Tensor) -> Tensor:
        temporal = torch.cat([branch(x) for branch in self.temporal_branches], dim=-1)
        temporal = self.temporal_norm(temporal)
        global_features = self.spatial_global(temporal)
        hemisphere_features = self.spatial_hemisphere(temporal)
        return self.spatial_norm(
            torch.cat([global_features, hemisphere_features], dim=2)
        )

    def _feature_dim(self, n_channels: int, n_samples: int) -> int:
        was_training = self.training
        self.eval()
        with torch.no_grad():
            output = self._spatial_features(torch.zeros(1, 1, n_channels, n_samples))
        self.train(was_training)
        return int(output[0].numel())

    def extract_features(self, x: Tensor) -> Tensor:
        return self._spatial_features(x).flatten(start_dim=1)

    def forward(self, x: Tensor, de: Tensor | None = None) -> Tensor:
        del de
        return self.classifier(self.extract_features(x))


def build_model(
    args: argparse.Namespace,
    model_name: str,
    n_channels: int,
    n_samples: int,
    n_classes: int,
) -> nn.Module:
    sampling_rate = int(getattr(args, "sampling_rate", 200))
    if model_name == "eegnet":
        return EEGNetBaseline(n_channels, n_samples, n_classes, sampling_rate)
    if model_name == "tsception":
        return TSceptionBaseline(n_channels, n_samples, n_classes, sampling_rate)
    return EmotionNet(args, model_name, n_channels, n_samples, n_classes, args.de_features)


def fit_robust_scaler(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(x, axis=(0, 2), keepdims=True)
    mad = np.median(np.abs(x - center), axis=(0, 2), keepdims=True)
    scale = 1.4826 * mad
    scale = np.where(scale < 1e-6, 1.0, scale)
    return center.astype(np.float32), scale.astype(np.float32)


def transform(
    x: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    clip: float,
) -> np.ndarray:
    output = (x - center) / scale
    return np.clip(output, -clip, clip).astype(np.float32)[:, None, :, :]


def load_session(preprocessed_dir: Path, subject: int, session: int, de_features: bool = False) -> dict[str, np.ndarray]:
    path = preprocessed_dir / f"subject{subject:02d}_session{session}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as data:
        result = {
            "x": data["x"].astype(np.float32),
            "y": data["y"].astype(np.int64),
            "trial_id": data["trial_id"].astype(np.int64),
            "window_id": data["window_id"].astype(np.int64),
        }
        if de_features and "de_features" in data:
            result["de"] = data["de_features"].astype(np.float32)
        return result


def trials_by_class(arrays: dict[str, np.ndarray]) -> dict[int, list[int]]:
    output: dict[int, list[int]] = {class_id: [] for class_id in CLASSES}
    for trial in sorted(np.unique(arrays["trial_id"]).tolist()):
        mask = arrays["trial_id"] == trial
        label = int(arrays["y"][mask][0])
        output.setdefault(label, []).append(trial)
    return output


@torch.no_grad()
def collect_logits(
    model: nn.Module,
    x: np.ndarray,
    device: torch.device,
    batch_size: int,
    de: np.ndarray | None = None,
) -> np.ndarray:
    model.eval()
    outputs = []
    for start in range(0, len(x), batch_size):
        batch = torch.from_numpy(x[start : start + batch_size]).to(device)
        if de is not None:
            batch_de = torch.from_numpy(de[start : start + batch_size]).to(device)
            outputs.append(model(batch, batch_de).detach().cpu().numpy())
        else:
            outputs.append(model(batch).detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            batch = tuple(item.to(device) for item in batch)
            if len(batch) == 3:
                logits = model(batch[0], batch[2])
            else:
                logits = model(batch[0])
            total_loss += F.cross_entropy(logits, batch[1], reduction="sum").item()
            correct += int((logits.argmax(dim=1) == batch[1]).sum().item())
            total += len(batch[1])
    return total_loss / max(1, total), correct / max(1, total)


def train_model(
    model: nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    train_de: np.ndarray | None,
    val_de: np.ndarray | None,
    train_domain: np.ndarray | None,
    val_domain: np.ndarray | None,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
    *,
    calibration: bool = False,
) -> tuple[nn.Module, dict[str, object]]:
    set_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    train_data = [torch.from_numpy(train_x), torch.from_numpy(train_y)]
    if train_de is not None:
        train_data.append(torch.from_numpy(train_de))
    if train_domain is not None:
        train_data.append(torch.from_numpy(train_domain))
    val_data = [torch.from_numpy(val_x), torch.from_numpy(val_y)]
    if val_de is not None:
        val_data.append(torch.from_numpy(val_de))
    if val_domain is not None:
        val_data.append(torch.from_numpy(val_domain))
    train_loader = DataLoader(
        TensorDataset(*train_data),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(
        TensorDataset(*val_data),
        batch_size=args.eval_batch_size,
        shuffle=False,
    )
    if calibration:
        model.eval()
        model.classifier.train()
    else:
        model.train()
    learning_rate = (
        args.calibration_learning_rate
        if calibration and args.calibration_learning_rate > 0
        else args.learning_rate
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=args.weight_decay,
    )
    best_state = copy.deepcopy(model.state_dict())
    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train() if not calibration else None
        if calibration:
            model.eval()
            model.classifier.train()
        for batch_index, batch in enumerate(train_loader, start=1):
            batch = tuple(item.to(device) for item in batch)
            optimizer.zero_grad(set_to_none=True)
            # Determine the positional args: x=0, y=1, de?=2, domain?=last
            x, y = batch[0], batch[1]
            de_tensor = None
            domain_tensor = None
            idx = 2
            if train_de is not None:
                de_tensor = batch[idx]; idx += 1
            if train_domain is not None:
                domain_tensor = batch[idx]; idx += 1
            model_x = x
            if calibration and args.calibration_augment:
                amplitude = 1.0 + (
                    2.0 * torch.rand((len(x), 1, 1, 1), device=x.device) - 1.0
                ) * args.augment_amplitude
                noise = torch.randn_like(x) * args.augment_noise_std
                channel_keep = (
                    torch.rand((len(x), 1, x.shape[2], 1), device=x.device)
                    >= args.augment_channel_dropout
                ).to(x.dtype)
                model_x = (x * amplitude + noise) * channel_keep
            # Emotion loss
            if de_tensor is not None:
                logits = model(model_x, de_tensor)
            else:
                logits = model(model_x)
            loss = F.cross_entropy(logits, y, label_smoothing=args.label_smoothing)
            # Domain adversarial loss (DANN)
            if domain_tensor is not None and model.use_dann:
                dann_lambda = getattr(args, 'dann_lambda', 0.1)
                domain_logits = model.domain_forward(x, dann_lambda)
                loss = loss + F.cross_entropy(domain_logits, domain_tensor)
            # Supervised contrastive loss (only during non-calibration source training)
            use_contrastive = getattr(args, 'contrastive', False)
            if use_contrastive and not calibration:
                features = model.extract_features(x)
                c_alpha = getattr(args, 'contrastive_alpha', 0.5)
                c_temp = getattr(args, 'contrastive_temperature', 0.07)
                loss = loss + c_alpha * supcon_loss(features, y, c_temp)
            loss.backward()
            optimizer.step()
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break
        val_loss, val_acc = evaluate(model, val_loader, device)
        if val_acc > best_val_acc + 1e-8 or (
            abs(val_acc - best_val_acc) <= 1e-8 and val_loss < best_val_loss
        ):
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "trainable_params": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }


def freeze_for_calibration(model: nn.Module, args: argparse.Namespace) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for module in model.modules():
        if isinstance(module, nn.LayerNorm):
            for parameter in module.parameters(recurse=False):
                parameter.requires_grad = True
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    if model.use_de_features:
        for parameter in model.de_head.parameters():
            parameter.requires_grad = True
    if args.calibration_strategy == "last_block":
        for parameter in model.transformer[-1].parameters():
            parameter.requires_grad = True


def source_roles(
    subject: int,
    trials_by_class: dict[int, list[int]],
    split_seed: int,
) -> dict[int, str]:
    roles = trial_roles(subject, 1, 0, trials_by_class, split_seed)
    return {
        trial_id: ("validation" if role == "validation" else "train")
        for trial_id, role in roles.items()
    }


def source_train(
    args: argparse.Namespace,
    preprocessed_dir: Path,
    subject: int,
    model_name: str,
    seed: int,
    device: torch.device,
    bundle_dir: Path,
) -> tuple[nn.Module, np.ndarray, np.ndarray, dict[str, object]]:
    session1 = load_session(preprocessed_dir, subject, 1, args.de_features)
    use_dann = getattr(args, 'dann', False)
    classes = trials_by_class(session1)
    roles = source_roles(subject, classes, args.split_seed)
    train_mask = np.isin(session1["trial_id"], [t for t, role in roles.items() if role == "train"])
    val_mask = np.isin(session1["trial_id"], [t for t, role in roles.items() if role == "validation"])
    center, scale = fit_robust_scaler(session1["x"][train_mask])
    train_x = transform(session1["x"][train_mask], center, scale, args.clip_robust_z)
    train_y = session1["y"][train_mask]
    train_de = session1["de"][train_mask] if "de" in session1 else None
    train_domain = np.full(len(train_y), subject - 1, dtype=np.int64) if use_dann else None
    val_x = transform(session1["x"][val_mask], center, scale, args.clip_robust_z)
    val_y = session1["y"][val_mask]
    val_de = session1["de"][val_mask] if "de" in session1 else None
    val_domain = np.full(len(val_y), subject - 1, dtype=np.int64) if use_dann else None
    model = build_model(args, model_name, session1["x"].shape[1], session1["x"].shape[2], 3)
    model = model.to(device)
    model, info = train_model(
        model,
        train_x,
        train_y,
        val_x,
        val_y,
        train_de,
        val_de,
        train_domain,
        val_domain,
        args,
        device,
        seed + (1 if model_name == "post_mamba" else 0),
    )
    checkpoint_dir = bundle_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / f"source_{model_name}_s{subject:02d}_seed{seed}.pth"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "center": center,
            "scale": scale,
            "subject": subject,
            "model": model_name,
            "seed": seed,
            "args": vars(args),
        },
        checkpoint,
    )
    return model, center, scale, {**info, "source_checkpoint": checkpoint}


def target_split_data(
    args: argparse.Namespace,
    preprocessed_dir: Path,
    subject: int,
    session: int,
    fold: int,
    center: np.ndarray,
    scale: np.ndarray,
) -> tuple[dict[int, str], dict[str, SplitData], dict[int, str], dict[int, int]]:
    arrays = load_session(preprocessed_dir, subject, session, args.de_features)
    classes = trials_by_class(arrays)
    roles = trial_roles(subject, session, fold, classes, args.split_seed)
    split_data = window_split(
        arrays["x"],
        arrays["y"],
        arrays["trial_id"],
        arrays["window_id"],
        roles,
        args.shots_per_class,
        args.val_shots_per_class,
        de=arrays.get("de"),
    )
    transformed = {
        name: SplitData(
            x=transform(data.x, center, scale, args.clip_robust_z),
            y=data.y,
            trial_id=data.trial_id,
            window_id=data.window_id,
            de=data.de,
        )
        for name, data in split_data.items()
    }
    trial_labels = {
        trial: int(arrays["y"][arrays["trial_id"] == trial][0])
        for trial in np.unique(arrays["trial_id"]).tolist()
    }
    return roles, transformed, classes, trial_labels


def append_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def dedupe_rows(
    rows: list[dict[str, object]],
    key_fields: list[str],
) -> list[dict[str, object]]:
    seen: set[tuple[str, ...]] = set()
    output = []
    for row in rows:
        key = tuple(str(row[field]) for field in key_fields)
        if key in seen:
            continue
        seen.add(key)
        output.append(row)
    return output


def window_rows(
    args: argparse.Namespace,
    subject: int,
    session: int,
    fold: int,
    seed: int,
    model_name: str,
    split_name: str,
    logits: np.ndarray,
    split_data: SplitData,
) -> list[dict[str, object]]:
    probabilities = softmax(logits)
    rows = []
    for index in range(len(split_data.y)):
        row = {
            "subject": subject,
            "target_session": session,
            "fold": fold,
            "seed": seed,
            "model": model_name,
            "trial_id": int(split_data.trial_id[index]),
            "window_id": int(split_data.window_id[index]),
            "split": split_name,
            "y_true": int(split_data.y[index]),
            **{f"logit_{i}": float(logits[index, i]) for i in range(3)},
            **{f"probability_{i}": float(probabilities[index, i]) for i in range(3)},
        }
        rows.append(row)
    return rows


def write_run_outputs(
    args: argparse.Namespace,
    bundle_dir: Path,
    subject: int,
    session: int,
    fold: int,
    seed: int,
    model_name: str,
    split_data: dict[str, SplitData],
    model: nn.Module,
    device: torch.device,
    checkpoint_rel: Path,
) -> None:
    logits = {
        split_name: collect_logits(
            model,
            data.x,
            device,
            args.eval_batch_size,
            de=data.de,
        )
        for split_name, data in split_data.items()
    }
    for split_name in ("support", "validation", "test"):
        append_rows(
            bundle_dir / "window_predictions.csv",
            window_rows(
                args,
                subject,
                session,
                fold,
                seed,
                model_name,
                split_name,
                logits[split_name],
                split_data[split_name],
            ),
        )
    test_data = split_data["test"]
    trial_ids = sorted(np.unique(test_data.trial_id).tolist())
    trial_logits = [
        logits["test"][test_data.trial_id == trial_id] for trial_id in trial_ids
    ]
    trial_labels = np.asarray(
        [int(test_data.y[test_data.trial_id == trial_id][0]) for trial_id in trial_ids],
        dtype=np.int64,
    )
    support_logits = logits["support"]
    validation_logits = logits["validation"]
    support_prototypes = class_prototypes(
        softmax(support_logits), split_data["support"].y, 3
    )
    alpha, tau, val_rgtpc_acc = tune_rgtpc(
        validation_logits,
        split_data["validation"].y,
        support_prototypes,
        args.alpha_grid,
        args.tau_grid,
    )
    final_logits = np.concatenate([support_logits, validation_logits], axis=0)
    final_labels = np.concatenate(
        [split_data["support"].y, split_data["validation"].y], axis=0
    )
    prototypes = class_prototypes(softmax(final_logits), final_labels, 3)
    rows = trial_prediction_rows(
        trial_logits,
        trial_ids,
        trial_labels,
        prototypes,
        alpha,
        tau,
    )
    for row in rows:
        row.update(
            {
                "subject": subject,
                "target_session": session,
                "fold": fold,
                "seed": seed,
                "model": model_name,
                "split": "test",
                "val_rgtpc_acc": val_rgtpc_acc,
            }
        )
    append_rows(bundle_dir / "trial_predictions.csv", rows)
    preds_by_method: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for row in rows:
        preds_by_method[row["aggregation_method"]].append((int(row["y_true"]), int(row["y_pred"])))
    metric_rows = []
    for method in METHODS:
        pairs = preds_by_method[method]
        y_true = np.asarray([pair[0] for pair in pairs], dtype=np.int64)
        y_pred = np.asarray([pair[1] for pair in pairs], dtype=np.int64)
        metrics = metrics_for_predictions(y_true, y_pred)
        metric_rows.append(
            {
                "subject": subject,
                "target_session": session,
                "fold": fold,
                "seed": seed,
                "model": model_name,
                "aggregation_method": method,
                "n_test_trials": len(trial_ids),
                **metrics,
                "checkpoint_path": checkpoint_rel.as_posix(),
            }
        )
    append_rows(bundle_dir / "metrics.csv", metric_rows)


def completed_run_keys(metrics_path: Path) -> set[tuple[int, int, int, str, int]]:
    if not metrics_path.exists():
        return set()
    counts: dict[tuple[int, int, int, str, int], set[str]] = defaultdict(set)
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (
                int(row["subject"]),
                int(row["target_session"]),
                int(row["fold"]),
                row["model"],
                int(row["seed"]),
            )
            counts[key].add(row["aggregation_method"])
    return {key for key, methods in counts.items() if set(METHODS).issubset(methods)}


def existing_split_keys(splits_path: Path) -> set[tuple[int, int, int, int]]:
    if not splits_path.exists():
        return set()
    keys = set()
    with splits_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            keys.add(
                (
                    int(row["subject"]),
                    int(row["target_session"]),
                    int(row["fold"]),
                    int(row["seed"]),
                )
            )
    return keys


def existing_fold_keys(folds_path: Path) -> set[tuple[int, int, int, int]]:
    if not folds_path.exists():
        return set()
    keys = set()
    with folds_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            keys.add(
                (
                    int(row["subject"]),
                    int(row["target_session"]),
                    int(row["fold"]),
                    int(row["seed"]),
                )
            )
    return keys


def write_manifest(
    bundle_dir: Path,
    args: argparse.Namespace,
    subjects: list[int],
    target_sessions: list[int],
    folds: list[int],
    seeds: list[int],
    models: list[str],
) -> None:
    manifest = {
        "schema_version": "2.0",
        "dataset": "SEED",
        "protocol": "cross_session_few_shot_trial_level",
        "shot_unit": "window",
        "shots_per_class": args.shots_per_class,
        "val_shots_per_class": args.val_shots_per_class,
        "support_trials_per_class": 1,
        "validation_trials_per_class": 1,
        "test_trials_per_class": 3,
        "source_session": 1,
        "subjects": subjects,
        "target_sessions": target_sessions,
        "folds": folds,
        "seeds": seeds,
        "models": models,
        "classes": CLASSES,
        "aggregation_methods": list(METHODS),
        "selection_rule": "validation_only",
        "test_labels_used_for_tuning": False,
        "split_seed": args.split_seed,
        "label_map": {"-1": 0, "0": 1, "1": 2},
        "window_size": 200,
        "stride": 200,
        "de_features": bool(args.de_features),
        "dann": bool(args.dann),
        "contrastive": bool(args.contrastive),
        "label_smoothing": float(args.label_smoothing),
        "calibration_augment": bool(args.calibration_augment),
        "calibration_strategy": args.calibration_strategy,
        "calibration_learning_rate": float(args.calibration_learning_rate),
        "temporal_backbone": args.temporal_backbone,
        "temporal_depth": int(args.temporal_depth),
        "temporal_downsample": int(args.temporal_downsample),
        "temporal_kernels": str(args.temporal_kernels),
        "token_pooling": str(args.token_pooling),
    }
    if models == ["eegnet"]:
        manifest["model_configuration"] = {
            "family": "EEGNet-8,2",
            "sampling_rate_hz": 200,
            "f1": 8,
            "depth_multiplier": 2,
            "f2": 16,
            "temporal_kernel_samples": 100,
            "separable_kernel_samples": 25,
            "dropout": 0.5,
            "target_calibration": "classifier_only",
        }
    elif models == ["tsception"]:
        manifest["model_configuration"] = {
            "family": "TSception",
            "sampling_rate_hz": 200,
            "temporal_windows_seconds": [0.5, 0.25, 0.125],
            "temporal_filters": 9,
            "spatial_filters": 6,
            "hidden_units": 128,
            "dropout": 0.2,
            "target_calibration": "classifier_only",
        }
    (bundle_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_checksums(bundle_dir: Path) -> None:
    lines = []
    for path in sorted(bundle_dir.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            lines.append(f"{sha256(path)}  {path.relative_to(bundle_dir).as_posix()}")
    (bundle_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_environment(bundle_dir: Path) -> None:
    lines = [
        f"python={sys.executable}",
        f"python_version={sys.version.split()[0]}",
        f"torch={torch.__version__}",
        f"numpy={np.__version__}",
    ]
    try:
        import scipy

        lines.append(f"scipy={scipy.__version__}")
    except Exception:
        pass
    try:
        import mamba_ssm

        lines.append(f"mamba_ssm={mamba_ssm.__version__}")
    except Exception:
        pass
    (bundle_dir / "environment.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(bundle_dir: Path) -> None:
    metrics_path = bundle_dir / "metrics.csv"
    if not metrics_path.exists():
        return
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    lines = [
        "# SEED trial-safe cross-session smoke/evidence bundle",
        "",
        f"- Completed metric rows: {len(rows)}",
        "",
        "| Model | Method | Runs | Trial acc | BAcc | Macro F1 | Kappa |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model_name in sorted({row["model"] for row in rows}):
        for method in METHODS:
            selected = [row for row in rows if row["model"] == model_name and row["aggregation_method"] == method]
            if not selected:
                continue
            mean = {
                key: float(np.mean([float(row[key]) for row in selected]))
                for key in ("accuracy", "balanced_accuracy", "macro_f1", "kappa")
            }
            lines.append(
                f"| {model_name} | {method} | {len(selected)} | "
                f"{mean['accuracy']:.4f} | {mean['balanced_accuracy']:.4f} | "
                f"{mean['macro_f1']:.4f} | {mean['kappa']:.4f} |"
            )
    (bundle_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.epochs = min(args.epochs, 2)
        args.patience = max(1, min(args.patience, 2))
        args.max_train_batches = args.max_train_batches or 2
    subjects = parse_id_list(args.subjects, 15)
    target_sessions = parse_id_list(args.target_sessions, 3)
    folds = parse_id_list(args.folds, 5)
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    supported_models = {"conformer", "post_mamba", "eegnet", "tsception"}
    if not models or set(models) - supported_models:
        raise ValueError(f"Unsupported models: {models}")
    if args.temporal_backbone != "transformer" and models != ["conformer"]:
        raise ValueError(
            "High-resolution temporal backbones are single-model Conformer variants; "
            "use --models conformer."
        )
    baseline_models = {"eegnet", "tsception"}
    if set(models) & baseline_models:
        incompatible = (
            args.de_features
            or args.dann
            or args.contrastive
            or args.calibration_augment
            or args.label_smoothing != 0.0
            or args.calibration_strategy != "head_ln"
            or args.temporal_backbone != "transformer"
        )
        if incompatible:
            raise ValueError(
                "EEGNet/TSception are frozen raw-EEG baselines and cannot use DE, DANN, "
                "contrastive learning, augmentation, label smoothing, last-block calibration, "
                "or alternate temporal backbones."
            )
    if args.temporal_depth < 1:
        raise ValueError("temporal_depth must be >= 1")
    args.tau_grid = parse_float_list(args.tau_grid)
    args.alpha_grid = parse_float_list(args.alpha_grid)
    device = resolve_device(args.device)
    bundle_dir = args.bundle_dir.resolve()
    if args.overwrite and bundle_dir.exists():
        for path in sorted(bundle_dir.rglob("*"), reverse=True):
            if path.is_file() or path.is_symlink():
                path.unlink()
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "checkpoints").mkdir(exist_ok=True)
    for subject in subjects:
        for session in [1, *target_sessions]:
            load_session(args.preprocessed_dir, subject, session)
    run_keys = {
        (subject, session, fold, model_name, seed)
        for subject in subjects
        for session in target_sessions
        for fold in folds
        for model_name in models
        for seed in seeds
    }
    done = completed_run_keys(bundle_dir / "metrics.csv")
    splits_path = bundle_dir / "support_validation_test_splits.csv"
    folds_path = bundle_dir / "folds.csv"
    known_split_keys = existing_split_keys(splits_path)
    known_fold_keys = existing_fold_keys(folds_path)
    source_cache: dict[tuple[int, str, int], tuple[nn.Module, np.ndarray, np.ndarray]] = {}
    started = time.time()
    for subject in subjects:
        for model_name in models:
            for seed in seeds:
                source = source_train(
                    args,
                    args.preprocessed_dir,
                    subject,
                    model_name,
                    seed,
                    device,
                    bundle_dir,
                )
                source_cache[(subject, model_name, seed)] = (source[0], source[1], source[2])
                for session in target_sessions:
                    for fold in folds:
                        key = (subject, session, fold, model_name, seed)
                        if key in done:
                            print(f"[skip] {key}")
                            continue
                        roles, split_data, _classes, trial_labels = target_split_data(
                            args,
                            args.preprocessed_dir,
                            subject,
                            session,
                            fold,
                            source[1],
                            source[2],
                        )
                        split_key = (subject, session, fold, seed)
                        if split_key not in known_split_keys:
                            append_rows(
                                splits_path,
                                [
                                    {
                                        "subject": subject,
                                        "target_session": session,
                                        "fold": fold,
                                        "seed": seed,
                                        "trial_id": trial_id,
                                        "split": role,
                                        "y_true": trial_labels[trial_id],
                                    }
                                    for trial_id, role in sorted(roles.items())
                                ],
                            )
                            known_split_keys.add(split_key)
                        if split_key not in known_fold_keys:
                            append_rows(
                                folds_path,
                                [
                                    {
                                        "subject": subject,
                                        "target_session": session,
                                        "fold": fold,
                                        "seed": seed,
                                        "split_hash": split_hash(roles),
                                    }
                                ],
                            )
                            known_fold_keys.add(split_key)
                        model = copy.deepcopy(source[0])
                        freeze_for_calibration(model, args)
                        model, _info = train_model(
                            model,
                            split_data["support"].x,
                            split_data["support"].y,
                            split_data["validation"].x,
                            split_data["validation"].y,
                            split_data["support"].de,
                            split_data["validation"].de,
                            None,
                            None,
                            args,
                            device,
                            seed + fold * 1009 + (2 if model_name == "post_mamba" else 0),
                            calibration=True,
                        )
                        checkpoint_rel = (
                            Path("checkpoints")
                            / f"{model_name}_s{subject:02d}_sess{session}_fold{fold}_seed{seed}.pth"
                        )
                        torch.save(
                            {
                                "state_dict": model.state_dict(),
                                "subject": subject,
                                "target_session": session,
                                "fold": fold,
                                "model": model_name,
                                "seed": seed,
                                "args": vars(args),
                            },
                            bundle_dir / checkpoint_rel,
                        )
                        write_run_outputs(
                            args,
                            bundle_dir,
                            subject,
                            session,
                            fold,
                            seed,
                            model_name,
                            split_data,
                            model,
                            device,
                            checkpoint_rel,
                        )
                        print(
                            f"[done] {model_name} s{subject} session{session} "
                            f"fold{fold} seed{seed}",
                            flush=True,
                        )
    write_manifest(bundle_dir, args, subjects, target_sessions, folds, seeds, models)
    (bundle_dir / "commands.txt").write_text(
        " ".join([sys.executable, *sys.argv]) + "\n", encoding="utf-8"
    )
    write_environment(bundle_dir)
    write_report(bundle_dir)
    write_checksums(bundle_dir)
    print(f"elapsed_sec={time.time() - started:.1f}")
    print(bundle_dir)


if __name__ == "__main__":
    main()
