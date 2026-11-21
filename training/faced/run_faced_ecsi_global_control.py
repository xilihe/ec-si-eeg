#!/usr/bin/env python3
"""R1 matched global-video-adversary development control for FACED EC-SI.

This runner is intentionally separate from the frozen FACED v2 evidence runner.
It reuses the same train-only video heads for conditional and global adversarial
objectives so that model parameters, initialization, and inference are matched.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from run_faced_cross_subject import (
    N_CHANNELS,
    N_CLASSES,
    N_SAMPLES,
    aggregate_trial_logits,
    fit_de_scaler,
    fit_streaming_channel_scaler,
    load_mapping,
    load_subject_group,
    metric_values,
    normalize_de_in_place,
    normalize_in_place,
    split_subjects,
)
from run_seed_cross_session import (
    FineTemporalEmbedding,
    GradientReversalLayer,
    MultiScaleTemporalBlock,
    SpectralEncoder,
    resolve_device,
    set_seed,
)
from ecsi_gradient_guard import guarded_adversarial_backward
from ecsi_conditional_hsic import conditional_hsic_loss
from ecsi_hierarchical_rex import hierarchical_rex_penalty
from ecsi_dual_video_mixstyle import (
    emotion_conditioned_dual_video_mixstyle,
    stop_gradient_style_consistency_kl,
)
from ecsi_same_basin_soup import (
    FIXED_LEFT_WEIGHT,
    fixed_equal_weight_state_dict,
    state_dicts_identical,
)


MODEL_VARIANTS = (
    "ecsi_ce",
    "ecsi_adv",
    "ecsi_global_adv",
    "ecsi_contrastive",
    "ecsi_full",
    "ecsi_warm",
    "ecsi_guard",
    "ecsi_warm_guard",
    "ecsi_chsic_kd",
    "ecsi_hrex_adv",
    "ecsi_dvms_adv_v1",
    "ecsi_dvsc_adv_v1",
    "ecsi_dv_soup_v1",
)
DEVELOPMENT_MODEL_IDS = (
    "ecsi_ce",
    "ecsi_adv",
    "ecsi_warm",
    "ecsi_guard",
    "ecsi_warm_guard_l025",
    "ecsi_warm_guard_l050",
    "ecsi_warm_guard_l100",
    "ecsi_chsic_kd",
    "ecsi_hrex_adv",
    "ecsi_dvms_adv_v1",
    "ecsi_dvsc_adv_v1",
    "ecsi_dv_soup_v1",
)
WARM_GUARD_LAMBDAS = {
    "ecsi_warm_guard_l025": 0.25,
    "ecsi_warm_guard_l050": 0.5,
    "ecsi_warm_guard_l100": 1.0,
}
TRAIN_VIDEO_COUNTS = (2, 2, 2, 2, 3, 2, 2, 2, 2)
GLOBAL_LOSS_SCALE = float(np.mean(np.log(TRAIN_VIDEO_COUNTS)) / np.log(sum(TRAIN_VIDEO_COUNTS)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--mapping-csv", type=Path, required=True)
    parser.add_argument("--stimulus-splits-json", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument(
        "--models",
        default="ecsi_ce,ecsi_adv,ecsi_global_adv,ecsi_contrastive,ecsi_full",
    )
    parser.add_argument("--model-variant", choices=MODEL_VARIANTS)
    parser.add_argument("--subject-folds", default="1")
    parser.add_argument("--stimulus-folds", default="1,2,3")
    parser.add_argument("--seeds", default="20260812")
    parser.add_argument("--split-seed", type=int, default=20260811)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--seconds-per-trial", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-train-batches", type=int, default=100)
    parser.add_argument("--clip-robust-z", type=float, default=20.0)
    parser.add_argument("--temporal-depth", type=int, default=3)
    parser.add_argument("--temporal-downsample", type=int, default=2)
    parser.add_argument("--temporal-kernels", default="7,7,7")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument("--projection-dim", type=int, default=32)
    parser.add_argument("--adv-lambda", type=float, default=0.1)
    parser.add_argument("--adv-schedule", choices=("constant", "dann"), default="constant")
    parser.add_argument("--warmup-fraction", type=float, default=1.0 / 3.0)
    parser.add_argument(
        "--gradient-guard",
        choices=("auto", "none", "project-conflict"),
        default="auto",
    )
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--contrastive-weight", type=float, default=0.1)
    parser.add_argument("--contrastive-temperature", type=float, default=0.1)
    parser.add_argument("--chsic-weight", type=float, default=0.1)
    parser.add_argument("--kd-weight", type=float, default=1.0)
    parser.add_argument("--kd-temperature", type=float, default=2.0)
    parser.add_argument("--chsic-finetune-epochs", type=int, default=5)
    parser.add_argument("--chsic-finetune-patience", type=int, default=3)
    parser.add_argument("--chsic-learning-rate", type=float, default=1e-4)
    parser.add_argument("--rex-weight", type=float, default=1.0)
    parser.add_argument("--mixstyle-probability", type=float, default=0.5)
    parser.add_argument("--mixstyle-alpha", type=float, default=0.1)
    parser.add_argument("--style-consistency-weight", type=float, default=1.0)
    parser.add_argument("--style-consistency-temperature", type=float, default=2.0)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu", "mps"), default="auto")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--consolidate-only", action="store_true")
    return parser.parse_args()


def int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def resolve_model_spec(token: str, default_lambda: float) -> dict[str, str | float]:
    if token in WARM_GUARD_LAMBDAS:
        return {
            "model_id": token,
            "variant": "ecsi_warm_guard",
            "adv_lambda": WARM_GUARD_LAMBDAS[token],
        }
    if token not in MODEL_VARIANTS:
        raise ValueError(
            f"unknown model {token!r}; expected a base variant or one of {DEVELOPMENT_MODEL_IDS}"
        )
    model_id = token
    if token == "ecsi_warm_guard":
        matches = [
            model_id
            for model_id, value in WARM_GUARD_LAMBDAS.items()
            if abs(value - default_lambda) <= 1e-12
        ]
        if len(matches) != 1:
            raise ValueError("ecsi_warm_guard requires adv_lambda 0.25, 0.5, or 1.0")
        model_id = matches[0]
    return {"model_id": model_id, "variant": token, "adv_lambda": float(default_lambda)}


def resolved_guard_mode(variant: str, requested: str) -> str:
    expected = "project-conflict" if variant in {"ecsi_guard", "ecsi_warm_guard"} else "none"
    if requested != "auto" and requested != expected:
        raise ValueError(
            f"model variant {variant} requires --gradient-guard {expected!r}, got {requested!r}"
        )
    return expected


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv_atomic(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("no rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_stimulus_splits(path: Path) -> dict[int, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS" or payload.get("n_stimulus_folds") != 12:
        raise ValueError("stimulus split manifest is not 12-fold PASS")
    return {int(row["stimulus_fold"]): row for row in payload["folds"]}


def training_video_local_ids(labels_by_trial: np.ndarray, train_videos: list[int]) -> np.ndarray:
    """Map only the videos visible in a stimulus fold to conditional head IDs."""
    local = np.full(len(labels_by_trial), -1, dtype=np.int64)
    for emotion in range(N_CLASSES):
        videos = [video for video in train_videos if labels_by_trial[video] == emotion]
        if len(videos) != TRAIN_VIDEO_COUNTS[emotion]:
            raise ValueError(f"emotion {emotion}: expected {TRAIN_VIDEO_COUNTS[emotion]} train videos")
        for local_id, video in enumerate(videos):
            local[video] = local_id
    return local


def training_video_global_ids(labels_by_trial: np.ndarray, train_videos: list[int]) -> np.ndarray:
    """Map visible videos to one 19-class target aligned with concatenated head logits."""
    conditional = training_video_local_ids(labels_by_trial, train_videos)
    offsets = np.cumsum((0, *TRAIN_VIDEO_COUNTS[:-1]))
    global_ids = np.full(len(labels_by_trial), -1, dtype=np.int64)
    for video in train_videos:
        emotion = int(labels_by_trial[video])
        global_ids[video] = int(offsets[emotion] + conditional[video])
    if sorted(global_ids[global_ids >= 0].tolist()) != list(range(sum(TRAIN_VIDEO_COUNTS))):
        raise AssertionError("global training-video IDs are not a complete 0..18 mapping")
    return global_ids


def subset_videos(group: dict[str, np.ndarray], videos: list[int]) -> dict[str, np.ndarray]:
    mask = np.isin(group["trial"], np.asarray(videos, dtype=np.int16))
    return {key: value[mask] for key, value in group.items()}


class ECSITCN(nn.Module):
    """Mean-pooling TCN + DE representation fusion with train-only auxiliary heads."""

    def __init__(self, args: argparse.Namespace):
        super().__init__()
        emb = 40
        kernels = tuple(int(item.strip()) for item in args.temporal_kernels.split(",") if item.strip())
        self.raw_patch = FineTemporalEmbedding(N_CHANNELS, emb, args.dropout, args.temporal_downsample)
        self.raw_encoder = nn.Sequential(
            *[MultiScaleTemporalBlock(emb, args.dropout, kernels) for _ in range(args.temporal_depth)]
        )
        self.raw_norm = nn.LayerNorm(emb)
        self.de_encoder = SpectralEncoder(N_CHANNELS, 5, emb, args.dropout)
        self.fusion = nn.Sequential(
            nn.Linear(2 * emb, args.feature_dim),
            nn.LayerNorm(args.feature_dim),
            nn.GELU(),
            nn.Dropout(args.dropout),
        )
        self.emotion_head = nn.Linear(args.feature_dim, N_CLASSES)
        self.projection_head = nn.Sequential(
            nn.Linear(args.feature_dim, args.feature_dim),
            nn.GELU(),
            nn.Linear(args.feature_dim, args.projection_dim),
        )
        self.video_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(args.feature_dim, 32),
                    nn.GELU(),
                    nn.Dropout(0.2),
                    nn.Linear(32, count),
                )
                for count in TRAIN_VIDEO_COUNTS
            ]
        )

    def encode(self, x: Tensor, de: Tensor) -> Tensor:
        raw_tokens = self.raw_encoder(self.raw_patch(x))
        raw = self.raw_norm(raw_tokens).mean(dim=1)
        spectral = self.de_encoder(de)
        return self.fusion(torch.cat([raw, spectral], dim=1))

    def forward(self, x: Tensor, de: Tensor) -> Tensor:
        return self.emotion_head(self.encode(x, de))

    def project(self, features: Tensor) -> Tensor:
        return F.normalize(self.projection_head(features), dim=1)

    def encoder_parameters(self):
        for module in (
            self.raw_patch,
            self.raw_encoder,
            self.raw_norm,
            self.de_encoder,
            self.fusion,
        ):
            yield from module.parameters()

    def conditional_video_identity_loss(
        self, features: Tensor, emotion: Tensor, local_video: Tensor
    ) -> Tensor:
        losses = []
        for class_id, head in enumerate(self.video_heads):
            mask = emotion == class_id
            if mask.any():
                losses.append(F.cross_entropy(head(features[mask]), local_video[mask]))
        if not losses:
            raise ValueError("conditional identity loss received an empty batch")
        return torch.stack(losses).mean()

    def conditional_video_loss(
        self, features: Tensor, emotion: Tensor, local_video: Tensor, reversal_lambda: float
    ) -> Tensor:
        reversed_features = GradientReversalLayer.apply(features, reversal_lambda)
        return self.conditional_video_identity_loss(reversed_features, emotion, local_video)

    def global_video_loss(
        self, features: Tensor, global_video: Tensor, reversal_lambda: float
    ) -> Tensor:
        """Single-softmax nuisance control over the 19 videos visible in training."""
        reversed_features = GradientReversalLayer.apply(features, reversal_lambda)
        logits = torch.cat([head(reversed_features) for head in self.video_heads], dim=1)
        return F.cross_entropy(logits, global_video) * GLOBAL_LOSS_SCALE

    def inference_parameter_count(self) -> int:
        excluded = {
            id(p)
            for m in (self.projection_head, self.video_heads)
            for p in m.parameters()
        }
        return sum(p.numel() for p in self.parameters() if id(p) not in excluded)


def cross_video_supcon_loss(
    projected: Tensor,
    emotion: Tensor,
    video: Tensor,
    subject: Tensor,
    temperature: float,
) -> Tensor:
    similarity = projected @ projected.T / temperature
    eye = torch.eye(len(projected), dtype=torch.bool, device=projected.device)
    positive = (
        (emotion[:, None] == emotion[None, :])
        & (video[:, None] != video[None, :])
        & (subject[:, None] != subject[None, :])
        & ~eye
    )
    valid_anchor = positive.any(dim=1)
    if not valid_anchor.any():
        return projected.sum() * 0.0
    negative = emotion[:, None] != emotion[None, :]
    eligible = positive | negative
    logits = similarity.masked_fill(~eligible, -torch.inf)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_count = positive.sum(dim=1).clamp_min(1)
    mean_positive_log_prob = (log_prob.masked_fill(~positive, 0.0).sum(dim=1) / positive_count)
    return -mean_positive_log_prob[valid_anchor].mean()


@torch.no_grad()
def collect_outputs(
    model: ECSITCN, x: np.ndarray, de: np.ndarray, device: torch.device, batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits, features = [], []
    for start in range(0, len(x), batch_size):
        bx = torch.from_numpy(x[start : start + batch_size]).to(device)
        bd = torch.from_numpy(de[start : start + batch_size]).to(device)
        z = model.encode(bx, bd)
        logits.append(model.emotion_head(z).cpu().numpy())
        features.append(z.cpu().numpy())
    return np.concatenate(logits), np.concatenate(features)


def trial_aggregate(group: dict[str, np.ndarray], values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keys = np.column_stack([group["subject"], group["trial"]])
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    output = np.empty((len(unique), values.shape[1]), dtype=np.float32)
    for index in range(len(unique)):
        output[index] = values[inverse == index].mean(axis=0)
    return output, unique[:, 0].astype(np.int16), unique[:, 1].astype(np.int16)


def conditional_video_probe(
    train_group: dict[str, np.ndarray],
    train_features: np.ndarray,
    evaluation_group: dict[str, np.ndarray],
    evaluation_features: np.ndarray,
    labels_by_video: np.ndarray,
) -> tuple[float, float, list[dict]]:
    """Fresh nearest-centroid probe fitted on train and tested on unseen subjects."""
    train_trial, _, train_video = trial_aggregate(train_group, train_features)
    evaluation_trial, evaluation_subject, evaluation_video = trial_aggregate(
        evaluation_group, evaluation_features
    )
    mean = train_trial.mean(axis=0)
    scale = train_trial.std(axis=0)
    scale[scale < 1e-8] = 1.0
    train_z = (train_trial - mean) / scale
    evaluation_z = (evaluation_trial - mean) / scale
    subject_emotion_rows: list[dict] = []
    for emotion in range(N_CLASSES):
        candidate_videos = sorted(np.unique(train_video[labels_by_video[train_video] == emotion]).tolist())
        centroids = np.stack([train_z[train_video == video].mean(axis=0) for video in candidate_videos])
        mask = labels_by_video[evaluation_video] == emotion
        distances = ((evaluation_z[mask, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        predictions = np.asarray(candidate_videos)[np.argmin(distances, axis=1)]
        emotion_subject = evaluation_subject[mask]
        emotion_truth = evaluation_video[mask]
        for subject_id in sorted(np.unique(emotion_subject).tolist()):
            subject_mask = emotion_subject == subject_id
            subject_emotion_rows.append({
                "subject": int(subject_id),
                "emotion": emotion,
                "videos": len(candidate_videos),
                "samples": int(subject_mask.sum()),
                "accuracy": float(np.mean(predictions[subject_mask] == emotion_truth[subject_mask])),
                "chance": float(1.0 / len(candidate_videos)),
            })
    return (
        float(np.mean([row["accuracy"] for row in subject_emotion_rows])),
        float(np.mean([row["chance"] for row in subject_emotion_rows])),
        subject_emotion_rows,
    )


def validation_score(
    model: ECSITCN, val: dict[str, np.ndarray], val_x: np.ndarray, val_de: np.ndarray,
    device: torch.device, batch_size: int
) -> tuple[float, float]:
    logits, _ = collect_outputs(model, val_x, val_de, device, batch_size)
    trials = aggregate_trial_logits(val, logits)
    metrics = metric_values(trials["y"], trials["logits"])
    probabilities = torch.from_numpy(logits)
    labels = torch.from_numpy(val["y"])
    loss = float(F.cross_entropy(probabilities, labels).item())
    return float(metrics["balanced_accuracy"]), loss


def train_variant(
    model: ECSITCN,
    variant: str,
    adv_lambda: float,
    guard_mode: str,
    train: dict[str, np.ndarray],
    val: dict[str, np.ndarray],
    train_x: np.ndarray,
    val_x: np.ndarray,
    train_de: np.ndarray,
    val_de: np.ndarray,
    train_local_by_video: np.ndarray,
    train_global_by_video: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[ECSITCN, dict]:
    set_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    mixstyle_rng = np.random.default_rng(seed + 32452843)
    local_video = train_local_by_video[train["trial"]]
    global_video = train_global_by_video[train["trial"]]
    if np.any(local_video < 0):
        raise AssertionError("training data contains a video without a conditional-head label")
    if np.any(global_video < 0):
        raise AssertionError("training data contains a video without a global-head label")
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_x), torch.from_numpy(train["y"]), torch.from_numpy(train_de),
            torch.from_numpy(train["subject"].astype(np.int64)),
            torch.from_numpy(train["trial"].astype(np.int64)),
            torch.from_numpy(local_video.astype(np.int64)),
            torch.from_numpy(global_video.astype(np.int64)),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_state = copy.deepcopy(model.state_dict())
    best_score, best_loss, best_epoch, stale = -1.0, float("inf"), 0, 0
    history = []
    use_conditional_adv = variant in {
        "ecsi_adv",
        "ecsi_full",
        "ecsi_warm",
        "ecsi_guard",
        "ecsi_warm_guard",
        "ecsi_hrex_adv",
        "ecsi_dvms_adv_v1",
        "ecsi_dvsc_adv_v1",
    }
    use_global_adv = variant == "ecsi_global_adv"
    use_contrastive = variant in {"ecsi_contrastive", "ecsi_full"}
    use_warmup = variant in {"ecsi_warm", "ecsi_warm_guard"}
    warmup_epochs = math.ceil(args.epochs * args.warmup_fraction) if use_warmup else 0
    if use_warmup and warmup_epochs >= args.epochs:
        raise ValueError("warm variants require at least one post-warmup epoch")
    effective_batches = min(len(loader), args.max_train_batches) if args.max_train_batches else len(loader)
    adversarial_epochs = max(1, args.epochs - warmup_epochs)
    total_steps = max(1, adversarial_epochs * effective_batches)
    total_guard_batches = 0
    total_guard_conflicts = 0
    total_mixstyle_batches = 0
    total_mixed_video_groups = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = np.zeros(8, dtype=float)
        batches = 0
        epoch_guard_batches = 0
        epoch_guard_conflicts = 0
        epoch_guard_dot_before = 0.0
        epoch_guard_dot_after = 0.0
        epoch_mixstyle_batches = 0
        epoch_mixed_video_groups = 0
        adversary_active = use_conditional_adv and epoch > warmup_epochs
        training_stage = "warmup_ce" if use_warmup and not adversary_active else (
            "conditional_adversarial" if adversary_active else "task_training"
        )
        for batch_index, (x, y, de, subject, video, local, global_id) in enumerate(loader, start=1):
            x, y, de = x.to(device), y.to(device), de.to(device)
            subject, video, local = subject.to(device), video.to(device), local.to(device)
            global_id = global_id.to(device)
            optimizer.zero_grad(set_to_none=True)
            features = model.encode(x, de)
            style_consistency_loss = features.sum() * 0.0
            if variant == "ecsi_dvms_adv_v1":
                features, mixstyle_audit = emotion_conditioned_dual_video_mixstyle(
                    features,
                    y,
                    video,
                    mixstyle_rng,
                    probability=args.mixstyle_probability,
                    alpha=args.mixstyle_alpha,
                )
                epoch_mixstyle_batches += int(mixstyle_audit["applied"])
                epoch_mixed_video_groups += int(mixstyle_audit["mixed_video_groups"])
            elif variant == "ecsi_dvsc_adv_v1":
                style_features, mixstyle_audit = emotion_conditioned_dual_video_mixstyle(
                    features,
                    y,
                    video,
                    mixstyle_rng,
                    probability=args.mixstyle_probability,
                    alpha=args.mixstyle_alpha,
                )
                epoch_mixstyle_batches += int(mixstyle_audit["applied"])
                epoch_mixed_video_groups += int(mixstyle_audit["mixed_video_groups"])
                if mixstyle_audit["applied"]:
                    style_consistency_loss = stop_gradient_style_consistency_kl(
                        model.emotion_head(style_features),
                        model.emotion_head(features),
                        args.style_consistency_temperature,
                    )
            per_sample_emotion_loss = F.cross_entropy(
                model.emotion_head(features), y, reduction="none"
            )
            emotion_loss = per_sample_emotion_loss.mean()
            adversarial_epoch = max(0, epoch - warmup_epochs - 1)
            progress = min(1.0, (adversarial_epoch * effective_batches + batch_index) / total_steps)
            reversal_lambda = adv_lambda
            if args.adv_schedule == "dann":
                reversal_lambda *= 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0
            guard_active = adversary_active and guard_mode == "project-conflict"
            if guard_active:
                conditional_adversarial_loss = model.conditional_video_identity_loss(
                    features, y, local
                )
            elif adversary_active:
                conditional_adversarial_loss = model.conditional_video_loss(
                    features, y, local, reversal_lambda
                )
            else:
                conditional_adversarial_loss = features.sum() * 0.0
            global_adversarial_loss = (
                model.global_video_loss(features, global_id, reversal_lambda)
                if use_global_adv else features.sum() * 0.0
            )
            contrastive_loss = (
                cross_video_supcon_loss(model.project(features), y, video, subject, args.contrastive_temperature)
                if use_contrastive else features.sum() * 0.0
            )
            if variant == "ecsi_hrex_adv":
                subject_rex_loss, stimulus_rex_loss = hierarchical_rex_penalty(
                    per_sample_emotion_loss, y, video, subject
                )
            else:
                subject_rex_loss = features.sum() * 0.0
                stimulus_rex_loss = features.sum() * 0.0
            loss = (
                emotion_loss + conditional_adversarial_loss + global_adversarial_loss
                + args.contrastive_weight * contrastive_loss
                + args.rex_weight * (subject_rex_loss + stimulus_rex_loss)
                + args.style_consistency_weight * style_consistency_loss
            )
            if guard_active:
                diagnostics = guarded_adversarial_backward(
                    emotion_loss,
                    conditional_adversarial_loss,
                    encoder_parameters=model.encoder_parameters(),
                    emotion_head_parameters=model.emotion_head.parameters(),
                    nuisance_head_parameters=model.video_heads.parameters(),
                    adversarial_strength=reversal_lambda,
                )
                epoch_guard_batches += 1
                epoch_guard_conflicts += int(diagnostics.conflict)
                epoch_guard_dot_before += diagnostics.dot_before
                epoch_guard_dot_after += diagnostics.dot_after
            else:
                loss.backward()
            optimizer.step()
            totals += [
                float(loss.item()), float(emotion_loss.item()),
                float(conditional_adversarial_loss.item()), float(global_adversarial_loss.item()),
                float(contrastive_loss.item()),
                float(subject_rex_loss.item()), float(stimulus_rex_loss.item()),
                float(style_consistency_loss.item()),
            ]
            batches += 1
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break
        score, val_loss = validation_score(model, val, val_x, val_de, device, args.eval_batch_size)
        history.append({
            "epoch": epoch, "train_total_loss": totals[0] / batches,
            "train_emotion_loss": totals[1] / batches,
            "train_conditional_video_loss": totals[2] / batches,
            "train_global_video_loss": totals[3] / batches,
            "train_cross_video_supcon_loss": totals[4] / batches,
            "train_subject_rex_loss": totals[5] / batches,
            "train_within_emotion_stimulus_rex_loss": totals[6] / batches,
            "train_style_consistency_loss": totals[7] / batches,
            "training_stage": training_stage,
            "gradient_guard_batches": epoch_guard_batches,
            "gradient_guard_conflicts": epoch_guard_conflicts,
            "gradient_guard_conflict_rate": (
                epoch_guard_conflicts / epoch_guard_batches if epoch_guard_batches else 0.0
            ),
            "gradient_guard_mean_dot_before": (
                epoch_guard_dot_before / epoch_guard_batches if epoch_guard_batches else 0.0
            ),
            "gradient_guard_mean_dot_after": (
                epoch_guard_dot_after / epoch_guard_batches if epoch_guard_batches else 0.0
            ),
            "mixstyle_batches": epoch_mixstyle_batches,
            "mixed_video_groups": epoch_mixed_video_groups,
            "val_trial_balanced_accuracy": score, "val_window_cross_entropy": val_loss,
        })
        total_guard_batches += epoch_guard_batches
        total_guard_conflicts += epoch_guard_conflicts
        total_mixstyle_batches += epoch_mixstyle_batches
        total_mixed_video_groups += epoch_mixed_video_groups
        checkpoint_eligible = not use_warmup or epoch > warmup_epochs
        if checkpoint_eligible and (
            score > best_score + 1e-8
            or (abs(score - best_score) <= 1e-8 and val_loss < best_loss)
        ):
            best_score, best_loss, best_epoch = score, val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        elif checkpoint_eligible:
            stale += 1
        if checkpoint_eligible and stale >= args.patience:
            break
    if best_epoch <= warmup_epochs and use_warmup:
        raise RuntimeError("warm variant did not produce an eligible adversarial-stage checkpoint")
    model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "best_val_trial_balanced_accuracy": best_score,
        "best_val_window_cross_entropy": best_loss,
        "warmup_epochs": warmup_epochs,
        "gradient_guard": guard_mode,
        "gradient_guard_batches": total_guard_batches,
        "gradient_guard_conflicts": total_guard_conflicts,
        "gradient_guard_conflict_rate": (
            total_guard_conflicts / total_guard_batches if total_guard_batches else 0.0
        ),
        "mixstyle_probability": args.mixstyle_probability,
        "mixstyle_alpha": args.mixstyle_alpha,
        "mixstyle_batches": total_mixstyle_batches,
        "mixed_video_groups": total_mixed_video_groups,
        "style_consistency_weight": args.style_consistency_weight,
        "style_consistency_temperature": args.style_consistency_temperature,
        "history": history,
        "total_trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "inference_params": model.inference_parameter_count(),
    }


def train_same_basin_soup(
    train: dict[str, np.ndarray],
    val: dict[str, np.ndarray],
    train_x: np.ndarray,
    val_x: np.ndarray,
    train_de: np.ndarray,
    val_de: np.ndarray,
    train_local_by_video: np.ndarray,
    train_global_by_video: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
    adv_lambda: float,
) -> tuple[ECSITCN, dict]:
    """Train both frozen components from one basin and average them 1:1."""

    set_seed(seed)
    dvms = ECSITCN(args).to(device)
    initial_state = copy.deepcopy(dvms.state_dict())
    set_seed(seed)
    dvsc = ECSITCN(args).to(device)
    initialization_identical = state_dicts_identical(initial_state, dvsc.state_dict())
    if not initialization_identical:
        raise AssertionError("same-basin component initializations are not identical")

    component_args = (
        train, val, train_x, val_x, train_de, val_de,
        train_local_by_video, train_global_by_video, args, device, seed,
    )
    dvms, dvms_training = train_variant(
        dvms, "ecsi_dvms_adv_v1", adv_lambda, "none", *component_args
    )
    dvsc, dvsc_training = train_variant(
        dvsc, "ecsi_dvsc_adv_v1", adv_lambda, "none", *component_args
    )
    merged_state = fixed_equal_weight_state_dict(dvms.state_dict(), dvsc.state_dict())
    dvms.load_state_dict(merged_state)
    history = [
        {"soup_component": "ecsi_dvms_adv_v1", **row}
        for row in dvms_training.pop("history")
    ] + [
        {"soup_component": "ecsi_dvsc_adv_v1", **row}
        for row in dvsc_training.pop("history")
    ]
    report = {
        "soup_weight_dvms": FIXED_LEFT_WEIGHT,
        "soup_weight_dvsc": 1.0 - FIXED_LEFT_WEIGHT,
        "soup_initialization_identical": initialization_identical,
        "soup_single_model_inference": True,
        "soup_checkpoint_selected": False,
        "soup_component_dvms_best_epoch": dvms_training["best_epoch"],
        "soup_component_dvsc_best_epoch": dvsc_training["best_epoch"],
        "soup_component_dvms_best_val_trial_balanced_accuracy": dvms_training[
            "best_val_trial_balanced_accuracy"
        ],
        "soup_component_dvsc_best_val_trial_balanced_accuracy": dvsc_training[
            "best_val_trial_balanced_accuracy"
        ],
        "mixstyle_probability": args.mixstyle_probability,
        "mixstyle_alpha": args.mixstyle_alpha,
        "mixstyle_batches": dvms_training["mixstyle_batches"] + dvsc_training["mixstyle_batches"],
        "mixed_video_groups": (
            dvms_training["mixed_video_groups"] + dvsc_training["mixed_video_groups"]
        ),
        "style_consistency_weight": args.style_consistency_weight,
        "style_consistency_temperature": args.style_consistency_temperature,
        "gradient_guard": "none",
        "gradient_guard_batches": 0,
        "gradient_guard_conflicts": 0,
        "gradient_guard_conflict_rate": 0.0,
        "total_trainable_params": sum(p.numel() for p in dvms.parameters() if p.requires_grad),
        "inference_params": dvms.inference_parameter_count(),
        "history": history,
    }
    del dvsc, initial_state, merged_state
    return dvms, report


def fine_tune_chsic_kd(
    model: ECSITCN,
    teacher: ECSITCN,
    train: dict[str, np.ndarray],
    val: dict[str, np.ndarray],
    train_x: np.ndarray,
    val_x: np.ndarray,
    train_de: np.ndarray,
    val_de: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[ECSITCN, dict]:
    set_seed(seed)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    generator = torch.Generator().manual_seed(seed + 104729)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_x),
            torch.from_numpy(train["y"]),
            torch.from_numpy(train_de),
            torch.from_numpy(train["trial"].astype(np.int64)),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.chsic_learning_rate, weight_decay=args.weight_decay
    )
    best_score, best_loss = validation_score(
        model, val, val_x, val_de, device, args.eval_batch_size
    )
    best_epoch, stale = 0, 0
    best_state = copy.deepcopy(model.state_dict())
    history = [
        {
            "epoch": 0,
            "training_stage": "ce_teacher_anchor",
            "train_total_loss": 0.0,
            "train_emotion_loss": 0.0,
            "train_kd_loss": 0.0,
            "train_conditional_hsic_loss": 0.0,
            "val_trial_balanced_accuracy": best_score,
            "val_window_cross_entropy": best_loss,
        }
    ]
    finetune_epochs = 1 if args.smoke else args.chsic_finetune_epochs
    patience = 1 if args.smoke else args.chsic_finetune_patience
    for epoch in range(1, finetune_epochs + 1):
        model.train()
        totals = np.zeros(4, dtype=float)
        batches = 0
        for batch_index, (x, y, de, video) in enumerate(loader, start=1):
            x, y, de, video = x.to(device), y.to(device), de.to(device), video.to(device)
            optimizer.zero_grad(set_to_none=True)
            features = model.encode(x, de)
            student_logits = model.emotion_head(features)
            with torch.no_grad():
                teacher_logits = teacher(x, de)
            emotion_loss = F.cross_entropy(student_logits, y)
            temperature = args.kd_temperature
            kd_loss = F.kl_div(
                F.log_softmax(student_logits / temperature, dim=1),
                F.softmax(teacher_logits / temperature, dim=1),
                reduction="batchmean",
            ) * (temperature * temperature)
            hsic_loss = conditional_hsic_loss(features, y, video)
            loss = emotion_loss + args.kd_weight * kd_loss + args.chsic_weight * hsic_loss
            loss.backward()
            optimizer.step()
            totals += [
                float(loss.item()),
                float(emotion_loss.item()),
                float(kd_loss.item()),
                float(hsic_loss.item()),
            ]
            batches += 1
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break
        score, val_loss = validation_score(
            model, val, val_x, val_de, device, args.eval_batch_size
        )
        history.append(
            {
                "epoch": epoch,
                "training_stage": "conditional_hsic_kd_finetune",
                "train_total_loss": totals[0] / batches,
                "train_emotion_loss": totals[1] / batches,
                "train_kd_loss": totals[2] / batches,
                "train_conditional_hsic_loss": totals[3] / batches,
                "val_trial_balanced_accuracy": score,
                "val_window_cross_entropy": val_loss,
            }
        )
        if score > best_score + 1e-8 or (
            abs(score - best_score) <= 1e-8 and val_loss < best_loss
        ):
            best_score, best_loss, best_epoch = score, val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "best_val_trial_balanced_accuracy": best_score,
        "best_val_window_cross_entropy": best_loss,
        "warmup_epochs": 0,
        "gradient_guard": "none",
        "gradient_guard_batches": 0,
        "gradient_guard_conflicts": 0,
        "gradient_guard_conflict_rate": 0.0,
        "teacher_anchor_eligible": True,
        "chsic_weight": args.chsic_weight,
        "kd_weight": args.kd_weight,
        "kd_temperature": args.kd_temperature,
        "history": history,
        "total_trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "inference_params": model.inference_parameter_count(),
    }


def run_dir(bundle: Path, subject_fold: int, stimulus_fold: int, seed: int, variant: str) -> Path:
    return bundle / "runs" / f"subjectfold{subject_fold}_stimulusfold{stimulus_fold}_seed{seed}_{variant}"


def main() -> None:
    args = parse_args()
    model_tokens = (
        [args.model_variant]
        if args.model_variant
        else [item.strip() for item in args.models.split(",") if item.strip()]
    )
    model_specs = [resolve_model_spec(token, args.adv_lambda) for token in model_tokens]
    model_ids = [str(spec["model_id"]) for spec in model_specs]
    if not model_specs or len(set(model_ids)) != len(model_ids):
        raise ValueError("model specifications must be nonempty and have unique model IDs")
    for spec in model_specs:
        spec["gradient_guard"] = resolved_guard_mode(
            str(spec["variant"]), args.gradient_guard
        )
    subject_folds = int_list(args.subject_folds)
    stimulus_folds = int_list(args.stimulus_folds)
    seeds = int_list(args.seeds)
    if not set(subject_folds) <= set(range(1, 6)) or not set(stimulus_folds) <= set(range(1, 13)):
        raise ValueError("subject folds must be 1..5 and stimulus folds 1..12")
    if args.smoke:
        subject_folds, stimulus_folds, seeds = subject_folds[:1], stimulus_folds[:1], seeds[:1]
        args.seconds_per_trial = min(args.seconds_per_trial, 2)
        has_warm_variant = any(
            spec["variant"] in {"ecsi_warm", "ecsi_warm_guard"} for spec in model_specs
        )
        args.epochs, args.patience, args.max_train_batches = (
            (2, 2, 2) if has_warm_variant else (1, 1, 2)
        )
    if not 1 <= args.seconds_per_trial <= 30:
        raise ValueError("seconds_per_trial must be 1..30")
    if (
        args.adv_lambda < 0
        or args.contrastive_weight < 0
        or args.contrastive_temperature <= 0
        or not 0.0 <= args.warmup_fraction < 1.0
    ):
        raise ValueError("loss weights must be nonnegative and temperature positive")
    if "ecsi_dv_soup_v1" in model_ids and not (
        abs(args.adv_lambda - 1.0) <= 1e-12
        and args.adv_schedule == "dann"
        and abs(args.mixstyle_probability - 0.5) <= 1e-12
        and abs(args.mixstyle_alpha - 0.1) <= 1e-12
        and abs(args.style_consistency_weight - 1.0) <= 1e-12
        and abs(args.style_consistency_temperature - 2.0) <= 1e-12
    ):
        raise ValueError("ecsi_dv_soup_v1 requires the frozen 20260825 hyperparameters")

    data_root = args.data_root.resolve()
    mapping_path = args.mapping_csv.resolve()
    split_path = args.stimulus_splits_json.resolve()
    bundle = args.bundle_dir.resolve()
    bundle.mkdir(parents=True, exist_ok=True)
    labels = load_mapping(mapping_path)
    stimulus_splits = read_stimulus_splits(split_path)
    device = resolve_device(args.device)
    started = time.time()
    input_descriptor = {
        "data_root": str(data_root),
        "mapping_sha256": sha256(mapping_path),
        "stimulus_splits_sha256": sha256(split_path),
        "seconds_per_trial": args.seconds_per_trial,
    }
    configuration = {
        "model_specs": model_specs,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_train_batches": args.max_train_batches,
        "warmup_fraction": args.warmup_fraction,
        "adv_schedule": args.adv_schedule,
        "chsic_weight": args.chsic_weight,
        "kd_weight": args.kd_weight,
        "kd_temperature": args.kd_temperature,
        "chsic_finetune_epochs": args.chsic_finetune_epochs,
        "chsic_finetune_patience": args.chsic_finetune_patience,
        "chsic_learning_rate": args.chsic_learning_rate,
        "rex_weight": args.rex_weight,
        "mixstyle_probability": args.mixstyle_probability,
        "mixstyle_alpha": args.mixstyle_alpha,
        "style_consistency_weight": args.style_consistency_weight,
        "style_consistency_temperature": args.style_consistency_temperature,
        "soup_weight_dvms": FIXED_LEFT_WEIGHT,
        "soup_weight_dvsc": 1.0 - FIXED_LEFT_WEIGHT,
    }
    manifest = {
        "status": "RUNNING", "prototype_stage": "M4_Gate_0_to_2", "smoke": bool(args.smoke),
        "dataset": "FACED", "protocol": "ecsi_dual_gain_post_hoc_extension_v1",
        "models": model_ids, "model_specs": model_specs,
        "subject_folds": subject_folds, "stimulus_folds": stimulus_folds, "seeds": seeds,
        "mapping_csv": str(mapping_path), "mapping_sha256": input_descriptor["mapping_sha256"],
        "stimulus_splits_json": str(split_path),
        "stimulus_splits_sha256": input_descriptor["stimulus_splits_sha256"],
        "input_descriptor_sha256": object_sha256(input_descriptor),
        "configuration_sha256": object_sha256(configuration),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "gradient_guard_sha256": sha256((Path(__file__).resolve().parent / "ecsi_gradient_guard.py")),
        "dual_video_mixstyle_sha256": sha256(
            Path(__file__).resolve().parent / "ecsi_dual_video_mixstyle.py"
        ),
        "same_basin_soup_sha256": sha256(
            Path(__file__).resolve().parent / "ecsi_same_basin_soup.py"
        ),
        "independent_unit": "subject", "nested_repeats": ["stimulus_fold", "training_seed"],
        "primary_prediction_unit": "held_out_video_trial_mean_logit",
        "test_subject_and_video_used_for_fit": False, "seconds_per_trial": args.seconds_per_trial,
        "device": str(device), "adv_lambda_default": args.adv_lambda,
        "adv_schedule": args.adv_schedule, "warmup_fraction": args.warmup_fraction,
        "resume": bool(args.resume),
        "global_video_loss_scale": GLOBAL_LOSS_SCALE,
        "contrastive_weight": args.contrastive_weight, "contrastive_temperature": args.contrastive_temperature,
    }
    if args.consolidate_only:
        metric_rows, subject_rows = [], []
        for subject_fold in subject_folds:
            for stimulus_fold in stimulus_folds:
                for seed in seeds:
                    for spec in model_specs:
                        model_id = str(spec["model_id"])
                        run = run_dir(bundle, subject_fold, stimulus_fold, seed, model_id)
                        completed_manifest_path = run / "run_manifest.json"
                        completed_subject_path = run / "subject_metrics.csv"
                        if not completed_manifest_path.is_file() or not completed_subject_path.is_file():
                            raise FileNotFoundError(f"incomplete run: {run}")
                        completed = json.loads(completed_manifest_path.read_text(encoding="utf-8"))
                        if completed.get("status") != "PASS":
                            raise ValueError(f"non-PASS run: {run}")
                        metric_rows.append(completed["metrics"])
                        with completed_subject_path.open("r", encoding="utf-8", newline="") as handle:
                            subject_rows.extend(list(csv.DictReader(handle)))
        metrics_path = bundle / "metrics.csv"
        subject_path = bundle / "subject_metrics.csv"
        write_csv_atomic(metrics_path, metric_rows)
        write_csv_atomic(subject_path, subject_rows)
        manifest.update({
            "status": "PASS", "consolidate_only": True,
            "expected_runs": len(subject_folds) * len(stimulus_folds) * len(seeds) * len(model_specs),
            "completed_runs": len(metric_rows), "elapsed_seconds": time.time() - started,
            "metrics_sha256": sha256(metrics_path), "subject_metrics_sha256": sha256(subject_path),
        })
        write_json_atomic(bundle / "manifest.json", manifest)
        print(json.dumps({"status": "PASS", "completed_runs": len(metric_rows), "bundle": str(bundle)}), flush=True)
        return
    write_json_atomic(bundle / "manifest.json", manifest)
    metric_rows, subject_rows = [], []

    for subject_fold in subject_folds:
        train_subjects, val_subjects, test_subjects = split_subjects(
            subject_fold - 1, args.split_seed, args.validation_fraction
        )
        all_train = load_subject_group(data_root, train_subjects, labels, args.seconds_per_trial)
        all_val = load_subject_group(data_root, val_subjects, labels, args.seconds_per_trial)
        all_test = load_subject_group(data_root, test_subjects, labels, args.seconds_per_trial)
        for stimulus_fold in stimulus_folds:
            split = stimulus_splits[stimulus_fold]
            train_videos = [int(v) for v in split["train_val_video_indices_zero_based"]]
            test_videos = [int(v) for v in split["test_video_indices_zero_based"]]
            if set(train_videos) & set(test_videos):
                raise AssertionError("stimulus leakage detected")
            train = subset_videos(all_train, train_videos)
            val = subset_videos(all_val, train_videos)
            test = subset_videos(all_test, test_videos)
            probe_test = subset_videos(all_test, train_videos)
            train_local_by_video = training_video_local_ids(labels, train_videos)
            train_global_by_video = training_video_global_ids(labels, train_videos)
            raw_mean, raw_scale = fit_streaming_channel_scaler(train["x"])
            train_x = normalize_in_place(train["x"], raw_mean, raw_scale, args.clip_robust_z)
            val_x = normalize_in_place(val["x"], raw_mean, raw_scale, args.clip_robust_z)
            test_x = normalize_in_place(test["x"], raw_mean, raw_scale, args.clip_robust_z)
            probe_test_x = normalize_in_place(probe_test["x"], raw_mean, raw_scale, args.clip_robust_z)
            train_de = train["de"].astype(np.float32, copy=False)
            val_de = val["de"].astype(np.float32, copy=False)
            test_de = test["de"].astype(np.float32, copy=False)
            de_mean, de_scale = fit_de_scaler(train_de)
            train_de = normalize_de_in_place(train_de, de_mean, de_scale, args.clip_robust_z)
            val_de = normalize_de_in_place(val_de, de_mean, de_scale, args.clip_robust_z)
            test_de = normalize_de_in_place(test_de, de_mean, de_scale, args.clip_robust_z)
            probe_test_de = normalize_de_in_place(
                probe_test["de"].astype(np.float32, copy=False), de_mean, de_scale, args.clip_robust_z
            )

            for seed in seeds:
                for spec in model_specs:
                    model_id = str(spec["model_id"])
                    variant = str(spec["variant"])
                    variant_lambda = float(spec["adv_lambda"])
                    guard_mode = str(spec["gradient_guard"])
                    run = run_dir(bundle, subject_fold, stimulus_fold, seed, model_id)
                    completed_manifest_path = run / "run_manifest.json"
                    completed_subject_path = run / "subject_metrics.csv"
                    if completed_manifest_path.exists() and completed_subject_path.exists():
                        completed = json.loads(completed_manifest_path.read_text(encoding="utf-8"))
                        if completed.get("status") == "PASS" and args.resume:
                            metric_rows.append(completed["metrics"])
                            with completed_subject_path.open("r", encoding="utf-8", newline="") as handle:
                                subject_rows.extend(list(csv.DictReader(handle)))
                            print(
                                f"[skip-complete] sf={subject_fold} vf={stimulus_fold} "
                                f"seed={seed} model={model_id}", flush=True,
                            )
                            continue
                        if completed.get("status") == "PASS" and not args.resume:
                            raise FileExistsError(
                                f"completed run exists and --no-resume forbids overwrite: {run}"
                            )
                    run_started = time.time()
                    if variant == "ecsi_dv_soup_v1":
                        model, training = train_same_basin_soup(
                            train, val, train_x, val_x, train_de, val_de,
                            train_local_by_video, train_global_by_video,
                            args, device, seed, variant_lambda,
                        )
                    else:
                        set_seed(seed)
                        model = ECSITCN(args).to(device)
                        model, training = train_variant(
                            model, "ecsi_ce" if variant == "ecsi_chsic_kd" else variant,
                            variant_lambda, guard_mode,
                            train, val, train_x, val_x, train_de, val_de,
                            train_local_by_video, train_global_by_video, args, device, seed,
                        )
                    if variant == "ecsi_chsic_kd":
                        teacher_training = training
                        teacher = copy.deepcopy(model).to(device)
                        model, training = fine_tune_chsic_kd(
                            model, teacher, train, val, train_x, val_x, train_de, val_de,
                            args, device, seed,
                        )
                        training["teacher_best_epoch"] = teacher_training["best_epoch"]
                        training["teacher_best_val_trial_balanced_accuracy"] = teacher_training[
                            "best_val_trial_balanced_accuracy"
                        ]
                        del teacher
                    train_logits, train_features = collect_outputs(
                        model, train_x, train_de, device, args.eval_batch_size
                    )
                    _, probe_test_features = collect_outputs(
                        model, probe_test_x, probe_test_de, device, args.eval_batch_size
                    )
                    probe_accuracy, probe_chance, probe_rows = conditional_video_probe(
                        train, train_features, probe_test, probe_test_features, labels
                    )
                    logits, features = collect_outputs(model, test_x, test_de, device, args.eval_batch_size)
                    trials = aggregate_trial_logits(test, logits)
                    trial_features, feature_subject, feature_trial = trial_aggregate(test, features)
                    metrics = metric_values(trials["y"], trials["logits"])
                    run.mkdir(parents=True, exist_ok=True)
                    checkpoint_path = run / "best_model_state.pt"
                    checkpoint_temporary = checkpoint_path.with_suffix(".pt.partial")
                    torch.save(model.state_dict(), checkpoint_temporary)
                    checkpoint_temporary.replace(checkpoint_path)
                    prediction_path = run / "predictions.npz"
                    with prediction_path.with_suffix(".npz.partial").open("wb") as handle:
                        np.savez_compressed(
                            handle, y=test["y"], logits=logits.astype(np.float32), features=features.astype(np.float32),
                            subject=test["subject"], trial=test["trial"], second=test["second"],
                            trial_y=trials["y"], trial_logits=trials["logits"], trial_subject=trials["subject"],
                            trial_index=trials["trial"], trial_features=trial_features,
                            feature_subject=feature_subject, feature_trial=feature_trial,
                        )
                    prediction_path.with_suffix(".npz.partial").replace(prediction_path)
                    history_path = run / "training_history.csv"
                    write_csv_atomic(history_path, training.pop("history"))
                    probe_path = run / "conditional_video_probe.csv"
                    write_csv_atomic(probe_path, probe_rows)
                    representation_path = run / "train_probe_trial_representations.npz"
                    train_trial_features, train_feature_subject, train_feature_trial = trial_aggregate(train, train_features)
                    probe_trial_features, probe_feature_subject, probe_feature_trial = trial_aggregate(
                        probe_test, probe_test_features
                    )
                    temporary_representation = representation_path.with_suffix(".npz.partial")
                    with temporary_representation.open("wb") as handle:
                        np.savez_compressed(
                            handle,
                            train_trial_features=train_trial_features,
                            train_subject=train_feature_subject,
                            train_video=train_feature_trial,
                            probe_trial_features=probe_trial_features,
                            probe_subject=probe_feature_subject,
                            probe_video=probe_feature_trial,
                        )
                    temporary_representation.replace(representation_path)
                    row = {
                        "dataset": "FACED", "subject_fold": subject_fold,
                        "stimulus_fold": stimulus_fold, "seed": seed, "model": model_id,
                        "model_variant": variant, "adv_lambda": variant_lambda,
                        "training_stage": "post_hoc_development_or_extension",
                        "configuration_sha256": manifest["configuration_sha256"],
                        "input_descriptor_sha256": manifest["input_descriptor_sha256"],
                        **metrics, **training,
                        "conditional_video_probe_accuracy": probe_accuracy,
                        "conditional_video_probe_chance": probe_chance,
                        "test_subjects": len(test_subjects), "test_videos": len(test_videos),
                        "test_trials": len(trials["y"]), "test_windows": len(test["y"]),
                        "duration_seconds": time.time() - run_started,
                    }
                    metric_rows.append(row)
                    probe_by_subject = {
                        int(subject_id): float(
                            np.mean(
                                [
                                    item["accuracy"]
                                    for item in probe_rows
                                    if int(item["subject"]) == int(subject_id)
                                ]
                            )
                        )
                        for subject_id in test_subjects
                    }
                    probe_chance_by_subject = {
                        int(subject_id): float(
                            np.mean(
                                [
                                    item["chance"]
                                    for item in probe_rows
                                    if int(item["subject"]) == int(subject_id)
                                ]
                            )
                        )
                        for subject_id in test_subjects
                    }
                    per_subject = []
                    for subject_id in test_subjects:
                        mask = trials["subject"] == subject_id
                        sm = metric_values(trials["y"][mask], trials["logits"][mask])
                        srow = {
                            "dataset": "FACED", "subject_fold": subject_fold,
                            "stimulus_fold": stimulus_fold, "seed": seed,
                            "model": model_id, "model_variant": variant,
                            "adv_lambda": variant_lambda, "subject": int(subject_id), **sm,
                            "conditional_video_probe_accuracy": probe_by_subject[int(subject_id)],
                            "conditional_video_probe_chance": probe_chance_by_subject[int(subject_id)],
                            "trials": int(mask.sum()),
                        }
                        per_subject.append(srow)
                        subject_rows.append(srow)
                    subject_path = run / "subject_metrics.csv"
                    write_csv_atomic(subject_path, per_subject)
                    run_manifest = {
                        "status": "PASS", "subject_fold": subject_fold, "stimulus_fold": stimulus_fold,
                        "seed": seed, "model": model_id, "model_variant": variant,
                        "adv_lambda": variant_lambda, "gradient_guard": guard_mode,
                        "warmup_fraction": args.warmup_fraction,
                        "train_subject_ids": train_subjects.tolist(), "val_subject_ids": val_subjects.tolist(),
                        "test_subject_ids": test_subjects.tolist(), "train_val_video_ids": train_videos,
                        "test_video_ids": test_videos, "subject_disjoint": True, "stimulus_disjoint": True,
                        "normalization_fit": "train_subjects_and_train_videos_only",
                        "conditional_video_probe_fit_subject_role": "train",
                        "conditional_video_probe_evaluation_subject_role": "test",
                        "test_subject_and_video_used_for_fit": False, "metrics": row,
                        "prediction_sha256": sha256(prediction_path), "subject_metrics_sha256": sha256(subject_path),
                        "checkpoint_sha256": sha256(checkpoint_path),
                        "training_history_sha256": sha256(history_path),
                        "conditional_video_probe_sha256": sha256(probe_path),
                        "train_probe_trial_representations_sha256": sha256(representation_path),
                    }
                    write_json_atomic(run / "run_manifest.json", run_manifest)
                    print(
                        f"[done] sf={subject_fold} vf={stimulus_fold} seed={seed} model={model_id} "
                        f"acc={metrics['accuracy']:.4f} bacc={metrics['balanced_accuracy']:.4f} "
                        f"probe={probe_accuracy:.4f}", flush=True,
                    )
                    del model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

    metrics_path = bundle / "metrics.csv"
    subject_path = bundle / "subject_metrics.csv"
    write_csv_atomic(metrics_path, metric_rows)
    write_csv_atomic(subject_path, subject_rows)
    manifest.update({
        "status": "PASS", "expected_runs": len(subject_folds) * len(stimulus_folds) * len(seeds) * len(model_specs),
        "completed_runs": len(metric_rows), "elapsed_seconds": time.time() - started,
        "metrics_sha256": sha256(metrics_path), "subject_metrics_sha256": sha256(subject_path),
    })
    write_json_atomic(bundle / "manifest.json", manifest)
    print(json.dumps({"status": "PASS", "completed_runs": len(metric_rows), "bundle": str(bundle)}), flush=True)


if __name__ == "__main__":
    main()
