#!/usr/bin/env python3
"""SEED replication of EC-SI under joint unseen-subject/unseen-stimulus evaluation.

The frozen protocol is documented in:
docs/analysis_plans/20260817_seed_ecsi_joint_subject_stimulus_external_plan_v1.md

The runner intentionally uses only DE features and writes compact, trial-level
artifacts. It never saves window-level predictions or model checkpoints.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

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


N_SUBJECTS = 15
N_SESSIONS = 3
N_STIMULI = 15
N_CLASSES = 3
TRAIN_STIMULI_PER_CLASS = 4
MODELS = (
    "ecsi_ce",
    "ecsi_adv",
    "ecsi_global_adv",
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
GLOBAL_LOSS_SCALE = float(math.log(TRAIN_STIMULI_PER_CLASS) / math.log(12))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--de-sidecar-dir",
        type=Path,
        help=(
            "Optional directory containing matching NPZ files with a de_features "
            "array. Used when the primary preprocessed files predate DE export."
        ),
    )
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--models", default="ecsi_ce,ecsi_adv,ecsi_global_adv")
    parser.add_argument("--model-variant", choices=MODELS)
    parser.add_argument("--subject-folds", default="1")
    parser.add_argument("--stimulus-folds", default="1")
    parser.add_argument("--seeds", default="20260817")
    parser.add_argument("--split-seed", type=int, default=20260817)
    parser.add_argument("--stimulus-seed", type=int, default=20260818)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-train-batches", type=int, default=80)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--feature-dim", type=int, default=64)
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
    parser.add_argument("--adv-lambda", type=float, default=1.0)
    parser.add_argument("--warmup-fraction", type=float, default=1.0 / 3.0)
    parser.add_argument(
        "--gradient-guard",
        choices=("auto", "none", "project-conflict"),
        default="auto",
    )
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--clip-z", type=float, default=20.0)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--export-representations",
        action="store_true",
        help=(
            "Save the best checkpoint plus trial-level train/probe/test representations. "
            "This is intended for frozen, train-only post-hoc projection experiments."
        ),
    )
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
    if token not in MODELS:
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
        raise ValueError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@dataclass
class Corpus:
    de: np.ndarray
    y: np.ndarray
    stimulus: np.ndarray
    subject: np.ndarray
    session: np.ndarray
    window: np.ndarray
    labels_by_stimulus: np.ndarray
    source_files: list[str]

    def subset(self, subjects: list[int] | np.ndarray, stimuli: list[int] | np.ndarray) -> dict[str, np.ndarray]:
        mask = np.isin(self.subject, np.asarray(subjects)) & np.isin(
            self.stimulus, np.asarray(stimuli)
        )
        return {
            "de": self.de[mask].copy(),
            "y": self.y[mask].copy(),
            "stimulus": self.stimulus[mask].copy(),
            "subject": self.subject[mask].copy(),
            "session": self.session[mask].copy(),
            "window": self.window[mask].copy(),
        }


def metadata_sha256(arrays: dict[str, np.ndarray]) -> str:
    """Hash the row identity without depending on NPZ container metadata."""
    digest = hashlib.sha256()
    for key in ("y", "trial_id", "subject", "session", "window_id"):
        value = np.ascontiguousarray(arrays[key])
        digest.update(key.encode("utf-8"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def load_corpus(data_dir: Path, de_sidecar_dir: Path | None = None) -> Corpus:
    expected = [
        data_dir / f"subject{subject:02d}_session{session}.npz"
        for subject in range(1, N_SUBJECTS + 1)
        for session in range(1, N_SESSIONS + 1)
    ]
    missing = [str(path) for path in expected if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} SEED files: {missing[:3]}")
    arrays: dict[str, list[np.ndarray]] = {
        key: [] for key in ("de", "y", "stimulus", "subject", "session", "window")
    }
    reference_mapping: np.ndarray | None = None
    source_files: list[str] = []
    for path in expected:
        with np.load(path, allow_pickle=False) as data:
            required = {"y", "trial_id", "subject", "session", "window_id"}
            if not required <= set(data.files):
                raise ValueError(f"{path} lacks {sorted(required - set(data.files))}")
            metadata = {key: np.asarray(data[key]) for key in sorted(required)}
            if "de_features" in data.files:
                de = data["de_features"].astype(np.float32)
                de_source = path
            else:
                if de_sidecar_dir is None:
                    raise ValueError(
                        f"{path} lacks de_features and --de-sidecar-dir was not supplied"
                    )
                de_source = de_sidecar_dir / path.name
                if not de_source.is_file():
                    raise FileNotFoundError(f"missing DE sidecar: {de_source}")
                with np.load(de_source, allow_pickle=False) as sidecar:
                    if "de_features" not in sidecar.files:
                        raise ValueError(f"{de_source} lacks de_features")
                    de = sidecar["de_features"].astype(np.float32)
                    if "metadata_sha256" not in sidecar.files:
                        raise ValueError(f"{de_source} lacks metadata_sha256")
                    recorded = str(sidecar["metadata_sha256"].item())
                observed = metadata_sha256(metadata)
                if recorded != observed:
                    raise ValueError(
                        f"DE sidecar row identity mismatch for {path}: "
                        f"recorded={recorded} observed={observed}"
                    )
            y = metadata["y"].astype(np.int64)
            stimulus = metadata["trial_id"].astype(np.int16) - 1
            subject = metadata["subject"].astype(np.int16)
            session = metadata["session"].astype(np.int8)
            window = metadata["window_id"].astype(np.int32)
        source_files.append(str(path.resolve()))
        if de_source != path:
            source_files.append(str(de_source.resolve()))
        if de.ndim != 3 or de.shape[1:] != (62, 5):
            raise ValueError(f"unexpected DE shape in {path}: {de.shape}")
        if not (len(de) == len(y) == len(stimulus) == len(subject) == len(session) == len(window)):
            raise ValueError(f"row mismatch in {path}")
        mapping = np.full(N_STIMULI, -1, dtype=np.int64)
        for item in range(N_STIMULI):
            labels = np.unique(y[stimulus == item])
            if len(labels) != 1:
                raise ValueError(f"stimulus {item + 1} has labels {labels.tolist()} in {path}")
            mapping[item] = int(labels[0])
        if reference_mapping is None:
            reference_mapping = mapping
        elif not np.array_equal(reference_mapping, mapping):
            raise ValueError(f"stimulus-label mapping differs in {path}")
        arrays["de"].append(de)
        arrays["y"].append(y)
        arrays["stimulus"].append(stimulus)
        arrays["subject"].append(subject)
        arrays["session"].append(session)
        arrays["window"].append(window)
    assert reference_mapping is not None
    class_counts = np.bincount(reference_mapping, minlength=N_CLASSES)
    if class_counts.tolist() != [5, 5, 5]:
        raise ValueError(f"expected five stimuli per class, got {class_counts.tolist()}")
    return Corpus(
        **{key: np.concatenate(value) for key, value in arrays.items()},
        labels_by_stimulus=reference_mapping,
        source_files=source_files,
    )


def make_subject_splits(seed: int) -> dict[int, dict[str, list[int]]]:
    subjects = np.arange(1, N_SUBJECTS + 1)
    permuted = np.random.default_rng(seed).permutation(subjects)
    folds = np.array_split(permuted, 5)
    output: dict[int, dict[str, list[int]]] = {}
    for fold_index, test in enumerate(folds, start=1):
        remaining = np.asarray(sorted(set(subjects.tolist()) - set(test.tolist())))
        ordered = np.random.default_rng(seed + fold_index).permutation(remaining)
        val = ordered[:2]
        train = ordered[2:]
        if set(train) & set(val) or set(train) & set(test) or set(val) & set(test):
            raise AssertionError("subject leakage")
        output[fold_index] = {
            "train": [int(x) for x in train],
            "val": [int(x) for x in val],
            "test": [int(x) for x in test],
        }
    if sorted(x for fold in output.values() for x in fold["test"]) != list(range(1, 16)):
        raise AssertionError("outer test folds do not cover subjects exactly once")
    return output


def make_stimulus_splits(labels: np.ndarray, seed: int) -> dict[int, dict[str, list[int]]]:
    rng = np.random.default_rng(seed)
    by_class = {
        class_id: rng.permutation(np.flatnonzero(labels == class_id)).tolist()
        for class_id in range(N_CLASSES)
    }
    output: dict[int, dict[str, list[int]]] = {}
    all_stimuli = set(range(N_STIMULI))
    for fold_index in range(1, 6):
        test = sorted(int(by_class[class_id][fold_index - 1]) for class_id in range(N_CLASSES))
        train = sorted(all_stimuli - set(test))
        if sorted(int(labels[x]) for x in test) != [0, 1, 2]:
            raise AssertionError("stimulus fold is not class-balanced")
        if set(train) & set(test) or len(train) != 12:
            raise AssertionError("stimulus leakage")
        output[fold_index] = {"train": train, "test": test}
    if sorted(x for fold in output.values() for x in fold["test"]) != list(range(N_STIMULI)):
        raise AssertionError("stimulus folds do not cover stimuli exactly once")
    return output


def fit_scaler(de: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = de.mean(axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
    scale = de.std(axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    return mean, scale


def normalize(de: np.ndarray, mean: np.ndarray, scale: np.ndarray, clip: float) -> np.ndarray:
    np.subtract(de, mean, out=de)
    np.divide(de, scale, out=de)
    np.clip(de, -clip, clip, out=de)
    return de


def training_stimulus_ids(
    labels_by_stimulus: np.ndarray, train_stimuli: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    local = np.full(N_STIMULI, -1, dtype=np.int64)
    global_id = np.full(N_STIMULI, -1, dtype=np.int64)
    counter = 0
    for class_id in range(N_CLASSES):
        candidates = [x for x in train_stimuli if labels_by_stimulus[x] == class_id]
        if len(candidates) != TRAIN_STIMULI_PER_CLASS:
            raise ValueError(f"class {class_id}: expected 4 training stimuli")
        for local_id, stimulus in enumerate(candidates):
            local[stimulus] = local_id
            global_id[stimulus] = counter
            counter += 1
    if sorted(global_id[global_id >= 0].tolist()) != list(range(12)):
        raise AssertionError("global stimulus IDs are incomplete")
    return local, global_id


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, strength: float) -> Tensor:
        ctx.strength = strength
        return x.view_as(x)

    @staticmethod
    def backward(ctx, gradient: Tensor) -> tuple[Tensor, None]:
        return -ctx.strength * gradient, None


class SeedECSI(nn.Module):
    def __init__(self, dropout: float, feature_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(62, 40, kernel_size=1),
            nn.BatchNorm1d(40),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(40, 40, kernel_size=5),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(40, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.emotion_head = nn.Linear(feature_dim, N_CLASSES)
        self.stimulus_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(feature_dim, 32),
                    nn.GELU(),
                    nn.Dropout(0.2),
                    nn.Linear(32, TRAIN_STIMULI_PER_CLASS),
                )
                for _ in range(N_CLASSES)
            ]
        )

    def encode(self, de: Tensor) -> Tensor:
        return self.encoder(de)

    def forward(self, de: Tensor) -> Tensor:
        return self.emotion_head(self.encode(de))

    def encoder_parameters(self):
        yield from self.encoder.parameters()

    def conditional_identity_loss(
        self, features: Tensor, y: Tensor, local_stimulus: Tensor
    ) -> Tensor:
        losses = []
        for class_id, head in enumerate(self.stimulus_heads):
            mask = y == class_id
            if mask.any():
                losses.append(F.cross_entropy(head(features[mask]), local_stimulus[mask]))
        if not losses:
            raise ValueError("conditional identity loss received an empty batch")
        return torch.stack(losses).mean()

    def conditional_loss(
        self, features: Tensor, y: Tensor, local_stimulus: Tensor, strength: float
    ) -> Tensor:
        reversed_features = GradientReversal.apply(features, strength)
        return self.conditional_identity_loss(reversed_features, y, local_stimulus)

    def global_loss(self, features: Tensor, global_stimulus: Tensor, strength: float) -> Tensor:
        reversed_features = GradientReversal.apply(features, strength)
        logits = torch.cat([head(reversed_features) for head in self.stimulus_heads], dim=1)
        return F.cross_entropy(logits, global_stimulus) * GLOBAL_LOSS_SCALE

    def inference_parameter_count(self) -> int:
        excluded = {id(parameter) for head in self.stimulus_heads for parameter in head.parameters()}
        return sum(parameter.numel() for parameter in self.parameters() if id(parameter) not in excluded)


def aggregate_trials(group: dict[str, np.ndarray], values: np.ndarray) -> dict[str, np.ndarray]:
    keys = np.column_stack([group["subject"], group["session"], group["stimulus"]])
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    aggregate = np.empty((len(unique), values.shape[1]), dtype=np.float32)
    labels = np.empty(len(unique), dtype=np.int64)
    for index in range(len(unique)):
        mask = inverse == index
        aggregate[index] = values[mask].mean(axis=0)
        unique_labels = np.unique(group["y"][mask])
        if len(unique_labels) != 1:
            raise AssertionError("trial contains multiple labels")
        labels[index] = int(unique_labels[0])
    return {
        "values": aggregate,
        "y": labels,
        "subject": unique[:, 0].astype(np.int16),
        "session": unique[:, 1].astype(np.int8),
        "stimulus": unique[:, 2].astype(np.int16),
    }


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


def classification_metrics(y: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    predicted = logits.argmax(axis=1)
    recalls = [float(np.mean(predicted[y == c] == c)) for c in range(N_CLASSES) if np.any(y == c)]
    probabilities = softmax(logits)
    return {
        "accuracy": float(np.mean(predicted == y)),
        "balanced_accuracy": float(np.mean(recalls)),
        "log_loss": float(-np.log(np.clip(probabilities[np.arange(len(y)), y], 1e-12, 1.0)).mean()),
    }


@torch.no_grad()
def collect_outputs(
    model: SeedECSI, de: np.ndarray, device: torch.device, batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits: list[np.ndarray] = []
    features: list[np.ndarray] = []
    for start in range(0, len(de), batch_size):
        batch = torch.from_numpy(de[start : start + batch_size]).to(device)
        z = model.encode(batch)
        logits.append(model.emotion_head(z).cpu().numpy())
        features.append(z.cpu().numpy())
    return np.concatenate(logits), np.concatenate(features)


def validation_score(
    model: SeedECSI, group: dict[str, np.ndarray], device: torch.device, batch_size: int
) -> tuple[float, float]:
    logits, _ = collect_outputs(model, group["de"], device, batch_size)
    trials = aggregate_trials(group, logits)
    metrics = classification_metrics(trials["y"], trials["values"])
    window_loss = classification_metrics(group["y"], logits)["log_loss"]
    return metrics["balanced_accuracy"], window_loss


def train_model(
    model: SeedECSI,
    variant: str,
    adv_lambda: float,
    guard_mode: str,
    train: dict[str, np.ndarray],
    val: dict[str, np.ndarray],
    local_by_stimulus: np.ndarray,
    global_by_stimulus: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[SeedECSI, dict]:
    set_seed(seed)
    local = local_by_stimulus[train["stimulus"]]
    global_id = global_by_stimulus[train["stimulus"]]
    if np.any(local < 0) or np.any(global_id < 0):
        raise AssertionError("training set contains held-out stimulus")
    generator = torch.Generator().manual_seed(seed)
    mixstyle_rng = np.random.default_rng(seed + 32452843)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train["de"]),
            torch.from_numpy(train["y"]),
            torch.from_numpy(local),
            torch.from_numpy(global_id),
            torch.from_numpy(train["stimulus"].astype(np.int64)),
            torch.from_numpy(train["subject"].astype(np.int64)),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    best_state = copy.deepcopy(model.state_dict())
    best_score, best_loss, best_epoch, stale = -1.0, float("inf"), 0, 0
    effective_batches = min(len(loader), args.max_train_batches) if args.max_train_batches else len(loader)
    use_conditional_adv = variant in {
        "ecsi_adv", "ecsi_warm", "ecsi_guard", "ecsi_warm_guard", "ecsi_hrex_adv",
        "ecsi_dvms_adv_v1", "ecsi_dvsc_adv_v1"
    }
    use_warmup = variant in {"ecsi_warm", "ecsi_warm_guard"}
    warmup_epochs = math.ceil(args.epochs * args.warmup_fraction) if use_warmup else 0
    if use_warmup and warmup_epochs >= args.epochs:
        raise ValueError("warm variants require at least one post-warmup epoch")
    total_steps = max(1, (args.epochs - warmup_epochs) * effective_batches)
    history: list[dict] = []
    total_guard_batches = 0
    total_guard_conflicts = 0
    total_mixstyle_batches = 0
    total_mixed_video_groups = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = np.zeros(7, dtype=float)
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
        for batch_index, (de, y, local_id, global_stimulus, stimulus, subject) in enumerate(loader, start=1):
            de, y = de.to(device), y.to(device)
            local_id, global_stimulus = local_id.to(device), global_stimulus.to(device)
            stimulus, subject = stimulus.to(device), subject.to(device)
            optimizer.zero_grad(set_to_none=True)
            features = model.encode(de)
            style_consistency_loss = features.sum() * 0.0
            if variant == "ecsi_dvms_adv_v1":
                features, mixstyle_audit = emotion_conditioned_dual_video_mixstyle(
                    features,
                    y,
                    stimulus,
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
                    stimulus,
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
            strength = adv_lambda * (2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0)
            guard_active = adversary_active and guard_mode == "project-conflict"
            if guard_active:
                conditional_loss = model.conditional_identity_loss(features, y, local_id)
            elif adversary_active:
                conditional_loss = model.conditional_loss(features, y, local_id, strength)
            else:
                conditional_loss = features.sum() * 0.0
            global_loss = (
                model.global_loss(features, global_stimulus, strength)
                if variant == "ecsi_global_adv"
                else features.sum() * 0.0
            )
            if variant == "ecsi_hrex_adv":
                subject_rex_loss, stimulus_rex_loss = hierarchical_rex_penalty(
                    per_sample_emotion_loss, y, stimulus, subject
                )
            else:
                subject_rex_loss = features.sum() * 0.0
                stimulus_rex_loss = features.sum() * 0.0
            loss = (
                emotion_loss + conditional_loss + global_loss
                + args.rex_weight * (subject_rex_loss + stimulus_rex_loss)
                + args.style_consistency_weight * style_consistency_loss
            )
            if guard_active:
                diagnostics = guarded_adversarial_backward(
                    emotion_loss,
                    conditional_loss,
                    encoder_parameters=model.encoder_parameters(),
                    emotion_head_parameters=model.emotion_head.parameters(),
                    nuisance_head_parameters=model.stimulus_heads.parameters(),
                    adversarial_strength=strength,
                )
                epoch_guard_batches += 1
                epoch_guard_conflicts += int(diagnostics.conflict)
                epoch_guard_dot_before += diagnostics.dot_before
                epoch_guard_dot_after += diagnostics.dot_after
            else:
                loss.backward()
            optimizer.step()
            totals += [
                float(loss.item()),
                float(emotion_loss.item()),
                float(conditional_loss.item()),
                float(global_loss.item()),
                float(subject_rex_loss.item()),
                float(stimulus_rex_loss.item()),
                float(style_consistency_loss.item()),
            ]
            batches += 1
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break
        val_score, val_loss = validation_score(model, val, device, args.eval_batch_size)
        row = {
            "epoch": epoch,
            "train_total_loss": totals[0] / batches,
            "train_emotion_loss": totals[1] / batches,
            "train_conditional_stimulus_loss": totals[2] / batches,
            "train_global_stimulus_loss": totals[3] / batches,
            "train_subject_rex_loss": totals[4] / batches,
            "train_within_emotion_stimulus_rex_loss": totals[5] / batches,
            "train_style_consistency_loss": totals[6] / batches,
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
            "val_trial_balanced_accuracy": val_score,
            "val_window_log_loss": val_loss,
        }
        numeric_values = (
            value for key, value in row.items() if key != "training_stage"
        )
        if not all(np.isfinite(float(value)) for value in numeric_values):
            raise FloatingPointError(f"non-finite training value: {row}")
        history.append(row)
        total_guard_batches += epoch_guard_batches
        total_guard_conflicts += epoch_guard_conflicts
        total_mixstyle_batches += epoch_mixstyle_batches
        total_mixed_video_groups += epoch_mixed_video_groups
        checkpoint_eligible = not use_warmup or epoch > warmup_epochs
        if checkpoint_eligible and (
            val_score > best_score + 1e-8
            or (abs(val_score - best_score) <= 1e-8 and val_loss < best_loss)
        ):
            best_score, best_loss, best_epoch = val_score, val_loss, epoch
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
        "best_val_window_log_loss": best_loss,
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
        "total_trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "inference_params": model.inference_parameter_count(),
        "history": history,
    }


def train_same_basin_soup(
    train: dict[str, np.ndarray],
    val: dict[str, np.ndarray],
    local_by_stimulus: np.ndarray,
    global_by_stimulus: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
    adv_lambda: float,
) -> tuple[SeedECSI, dict]:
    """Train both frozen components from one basin and average them 1:1."""

    set_seed(seed)
    dvms = SeedECSI(args.dropout, args.feature_dim).to(device)
    initial_state = copy.deepcopy(dvms.state_dict())
    set_seed(seed)
    dvsc = SeedECSI(args.dropout, args.feature_dim).to(device)
    initialization_identical = state_dicts_identical(initial_state, dvsc.state_dict())
    if not initialization_identical:
        raise AssertionError("same-basin component initializations are not identical")

    dvms, dvms_training = train_model(
        dvms, "ecsi_dvms_adv_v1", adv_lambda, "none", train, val,
        local_by_stimulus, global_by_stimulus, args, device, seed,
    )
    dvsc, dvsc_training = train_model(
        dvsc, "ecsi_dvsc_adv_v1", adv_lambda, "none", train, val,
        local_by_stimulus, global_by_stimulus, args, device, seed,
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
    model: SeedECSI,
    teacher: SeedECSI,
    train: dict[str, np.ndarray],
    val: dict[str, np.ndarray],
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[SeedECSI, dict]:
    set_seed(seed)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    generator = torch.Generator().manual_seed(seed + 104729)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train["de"]),
            torch.from_numpy(train["y"]),
            torch.from_numpy(train["stimulus"].astype(np.int64)),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.chsic_learning_rate, weight_decay=args.weight_decay
    )
    best_score, best_loss = validation_score(model, val, device, args.eval_batch_size)
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
            "val_window_log_loss": best_loss,
        }
    ]
    finetune_epochs = 1 if args.smoke else args.chsic_finetune_epochs
    patience = 1 if args.smoke else args.chsic_finetune_patience
    for epoch in range(1, finetune_epochs + 1):
        model.train()
        totals = np.zeros(4, dtype=float)
        batches = 0
        for batch_index, (de, y, stimulus) in enumerate(loader, start=1):
            de, y, stimulus = de.to(device), y.to(device), stimulus.to(device)
            optimizer.zero_grad(set_to_none=True)
            features = model.encode(de)
            student_logits = model.emotion_head(features)
            with torch.no_grad():
                teacher_logits = teacher(de)
            emotion_loss = F.cross_entropy(student_logits, y)
            temperature = args.kd_temperature
            kd_loss = F.kl_div(
                F.log_softmax(student_logits / temperature, dim=1),
                F.softmax(teacher_logits / temperature, dim=1),
                reduction="batchmean",
            ) * (temperature * temperature)
            hsic_loss = conditional_hsic_loss(features, y, stimulus)
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
        score, val_loss = validation_score(model, val, device, args.eval_batch_size)
        history.append(
            {
                "epoch": epoch,
                "training_stage": "conditional_hsic_kd_finetune",
                "train_total_loss": totals[0] / batches,
                "train_emotion_loss": totals[1] / batches,
                "train_kd_loss": totals[2] / batches,
                "train_conditional_hsic_loss": totals[3] / batches,
                "val_trial_balanced_accuracy": score,
                "val_window_log_loss": val_loss,
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
        "best_val_window_log_loss": best_loss,
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


def conditional_probe(
    train_group: dict[str, np.ndarray],
    train_features: np.ndarray,
    evaluation_group: dict[str, np.ndarray],
    evaluation_features: np.ndarray,
    labels_by_stimulus: np.ndarray,
) -> tuple[float, list[dict]]:
    train_trials = aggregate_trials(train_group, train_features)
    evaluation_trials = aggregate_trials(evaluation_group, evaluation_features)
    center = train_trials["values"].mean(axis=0)
    scale = train_trials["values"].std(axis=0)
    scale[scale < 1e-8] = 1.0
    train_z = (train_trials["values"] - center) / scale
    evaluation_z = (evaluation_trials["values"] - center) / scale
    rows: list[dict] = []
    for class_id in range(N_CLASSES):
        candidates = sorted(
            np.unique(
                train_trials["stimulus"][labels_by_stimulus[train_trials["stimulus"]] == class_id]
            ).tolist()
        )
        if len(candidates) != TRAIN_STIMULI_PER_CLASS:
            raise AssertionError("probe expected four candidates per class")
        centroids = np.stack(
            [train_z[train_trials["stimulus"] == stimulus].mean(axis=0) for stimulus in candidates]
        )
        mask = evaluation_trials["y"] == class_id
        distance = ((evaluation_z[mask, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        prediction = np.asarray(candidates)[np.argmin(distance, axis=1)]
        subjects = evaluation_trials["subject"][mask]
        truth = evaluation_trials["stimulus"][mask]
        for subject in sorted(np.unique(subjects).tolist()):
            subject_mask = subjects == subject
            rows.append(
                {
                    "subject": int(subject),
                    "emotion": class_id,
                    "accuracy": float(np.mean(prediction[subject_mask] == truth[subject_mask])),
                    "chance": 0.25,
                    "trials": int(subject_mask.sum()),
                }
            )
    per_subject = {
        subject: float(np.mean([row["accuracy"] for row in rows if row["subject"] == subject]))
        for subject in sorted({int(row["subject"]) for row in rows})
    }
    return float(np.mean(list(per_subject.values()))), rows


def run_dir(bundle: Path, subject_fold: int, stimulus_fold: int, seed: int, model: str) -> Path:
    return bundle / "runs" / (
        f"subjectfold{subject_fold}_stimulusfold{stimulus_fold}_seed{seed}_{model}"
    )


def consolidate(
    bundle: Path,
    subject_folds: list[int],
    stimulus_folds: list[int],
    seeds: list[int],
    models: list[str],
) -> tuple[list[dict], list[dict]]:
    metrics: list[dict] = []
    subject_metrics: list[dict] = []
    for subject_fold in subject_folds:
        for stimulus_fold in stimulus_folds:
            for seed in seeds:
                for model in models:
                    directory = run_dir(bundle, subject_fold, stimulus_fold, seed, model)
                    run_manifest = directory / "run_manifest.json"
                    subject_path = directory / "subject_metrics.csv"
                    if not run_manifest.is_file() or not subject_path.is_file():
                        raise FileNotFoundError(f"incomplete run: {directory}")
                    payload = json.loads(run_manifest.read_text(encoding="utf-8"))
                    if payload.get("status") != "PASS":
                        raise ValueError(f"non-PASS run: {directory}")
                    metrics.append(payload["metrics"])
                    subject_metrics.extend(read_csv(subject_path))
    expected = len(subject_folds) * len(stimulus_folds) * len(seeds) * len(models)
    if len(metrics) != expected:
        raise AssertionError(f"expected {expected} runs, found {len(metrics)}")
    write_csv_atomic(bundle / "metrics.csv", metrics)
    write_csv_atomic(bundle / "subject_metrics.csv", subject_metrics)
    return metrics, subject_metrics


def main() -> None:
    args = parse_args()
    model_tokens = (
        [args.model_variant]
        if args.model_variant
        else [item.strip() for item in args.models.split(",") if item.strip()]
    )
    model_specs = [resolve_model_spec(token, args.adv_lambda) for token in model_tokens]
    models = [str(spec["model_id"]) for spec in model_specs]
    if not model_specs or len(set(models)) != len(models):
        raise ValueError("model specifications must be nonempty and have unique model IDs")
    for spec in model_specs:
        spec["gradient_guard"] = resolved_guard_mode(
            str(spec["variant"]), args.gradient_guard
        )
    subject_folds = int_list(args.subject_folds)
    stimulus_folds = int_list(args.stimulus_folds)
    seeds = int_list(args.seeds)
    if not set(subject_folds) <= set(range(1, 6)):
        raise ValueError("subject folds must be 1..5")
    if not set(stimulus_folds) <= set(range(1, 6)):
        raise ValueError("stimulus folds must be 1..5")
    if args.smoke:
        subject_folds = subject_folds[:1]
        stimulus_folds = stimulus_folds[:1]
        seeds = seeds[:1]
        has_warm_variant = any(
            spec["variant"] in {"ecsi_warm", "ecsi_warm_guard"} for spec in model_specs
        )
        args.epochs, args.patience, args.max_train_batches = (
            (2, 2, 2) if has_warm_variant else (1, 1, 2)
        )
    if (
        args.epochs < 1
        or args.patience < 1
        or args.adv_lambda < 0
        or not 0.0 <= args.warmup_fraction < 1.0
    ):
        raise ValueError("invalid training hyperparameters")
    if "ecsi_dv_soup_v1" in models and not (
        abs(args.adv_lambda - 1.0) <= 1e-12
        and abs(args.mixstyle_probability - 0.5) <= 1e-12
        and abs(args.mixstyle_alpha - 0.1) <= 1e-12
        and abs(args.style_consistency_weight - 1.0) <= 1e-12
        and abs(args.style_consistency_temperature - 2.0) <= 1e-12
    ):
        raise ValueError("ecsi_dv_soup_v1 requires the frozen 20260825 hyperparameters")

    bundle = args.bundle_dir.resolve()
    bundle.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if args.consolidate_only:
        metrics, _ = consolidate(bundle, subject_folds, stimulus_folds, seeds, models)
        manifest_path = bundle / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
        payload.update(
            {
                "status": "PASS",
                "completed_runs": len(metrics),
                "expected_runs": len(metrics),
                "metrics_sha256": sha256(bundle / "metrics.csv"),
                "subject_metrics_sha256": sha256(bundle / "subject_metrics.csv"),
            }
        )
        write_json_atomic(manifest_path, payload)
        print(json.dumps({"status": "PASS", "completed_runs": len(metrics)}), flush=True)
        return

    de_sidecar_dir = args.de_sidecar_dir.resolve() if args.de_sidecar_dir else None
    corpus = load_corpus(args.data_dir.resolve(), de_sidecar_dir)
    subject_splits = make_subject_splits(args.split_seed)
    stimulus_splits = make_stimulus_splits(corpus.labels_by_stimulus, args.stimulus_seed)
    device = resolve_device(args.device)
    split_payload = {
        "subject_seed": args.split_seed,
        "stimulus_seed": args.stimulus_seed,
        "subject_splits": subject_splits,
        "stimulus_splits": stimulus_splits,
        "labels_by_stimulus_zero_based": corpus.labels_by_stimulus.tolist(),
    }
    split_path = bundle / "splits.json"
    write_json_atomic(split_path, split_payload)
    source_inventory = [
        {"path": source, "bytes": Path(source).stat().st_size, "sha256": sha256(Path(source))}
        for source in corpus.source_files
    ]
    input_descriptor = {
        "source_inventory_sha256": object_sha256(source_inventory),
        "split_sha256": sha256(split_path),
        "input_shape": "62_channels_x_5_bands",
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
        "de_sidecar_dir": str(de_sidecar_dir) if de_sidecar_dir else None,
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
        "status": "RUNNING",
        "dataset": "SEED",
        "protocol": "joint_unseen_subject_unseen_stimulus",
        "input": "SEED differential entropy features, 62 channels x 5 bands",
        "source_files": corpus.source_files,
        "de_sidecar_dir": str(de_sidecar_dir) if de_sidecar_dir else None,
        "models": models,
        "model_specs": model_specs,
        "subject_folds": subject_folds,
        "stimulus_folds": stimulus_folds,
        "seeds": seeds,
        "smoke": bool(args.smoke),
        "device": str(device),
        "independent_unit": "subject",
        "prediction_unit": "subject_x_session_x_stimulus_trial",
        "nested_repeats": ["session", "stimulus_fold", "training_seed"],
        "normalization_fit": "training_subjects_x_training_stimuli_only",
        "checkpoint_selection": "validation_subjects_x_training_stimuli_only",
        "test_subject_and_test_stimulus_used_for_fit": False,
        "identity_probe_fit": "training_subjects_x_training_stimuli_only",
        "identity_probe_evaluation": "test_subjects_x_training_stimuli_only",
        "identity_probe_chance": 0.25,
        "global_loss_scale": GLOBAL_LOSS_SCALE,
        "split_sha256": sha256(split_path),
        "source_inventory_sha256": input_descriptor["source_inventory_sha256"],
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
        "warmup_fraction": args.warmup_fraction,
        "resume": bool(args.resume),
        "export_representations": bool(args.export_representations),
    }
    write_json_atomic(bundle / "manifest.json", manifest)
    metric_rows: list[dict] = []
    subject_rows: list[dict] = []

    for subject_fold in subject_folds:
        subject_split = subject_splits[subject_fold]
        for stimulus_fold in stimulus_folds:
            stimulus_split = stimulus_splits[stimulus_fold]
            train = corpus.subset(subject_split["train"], stimulus_split["train"])
            val = corpus.subset(subject_split["val"], stimulus_split["train"])
            test = corpus.subset(subject_split["test"], stimulus_split["test"])
            probe_test = corpus.subset(subject_split["test"], stimulus_split["train"])
            mean, scale = fit_scaler(train["de"])
            for group in (train, val, test, probe_test):
                normalize(group["de"], mean, scale, args.clip_z)
            local_by_stimulus, global_by_stimulus = training_stimulus_ids(
                corpus.labels_by_stimulus, stimulus_split["train"]
            )
            for seed in seeds:
                for spec in model_specs:
                    model_id = str(spec["model_id"])
                    variant = str(spec["variant"])
                    variant_lambda = float(spec["adv_lambda"])
                    guard_mode = str(spec["gradient_guard"])
                    directory = run_dir(bundle, subject_fold, stimulus_fold, seed, model_id)
                    completed_manifest = directory / "run_manifest.json"
                    completed_subjects = directory / "subject_metrics.csv"
                    if completed_manifest.is_file() and completed_subjects.is_file():
                        completed = json.loads(completed_manifest.read_text(encoding="utf-8"))
                        if completed.get("status") == "PASS" and args.resume:
                            metric_rows.append(completed["metrics"])
                            subject_rows.extend(read_csv(completed_subjects))
                            print(
                                f"[skip] sf={subject_fold} vf={stimulus_fold} seed={seed} model={model_id}",
                                flush=True,
                            )
                            continue
                        if completed.get("status") == "PASS" and not args.resume:
                            raise FileExistsError(
                                f"completed run exists and --no-resume forbids overwrite: {directory}"
                            )
                    run_started = time.time()
                    if variant == "ecsi_dv_soup_v1":
                        model, training = train_same_basin_soup(
                            train, val, local_by_stimulus, global_by_stimulus,
                            args, device, seed, variant_lambda,
                        )
                    else:
                        set_seed(seed)
                        model = SeedECSI(args.dropout, args.feature_dim).to(device)
                        model, training = train_model(
                            model,
                            "ecsi_ce" if variant == "ecsi_chsic_kd" else variant,
                            variant_lambda,
                            guard_mode,
                            train,
                            val,
                            local_by_stimulus,
                            global_by_stimulus,
                            args,
                            device,
                            seed,
                        )
                    if variant == "ecsi_chsic_kd":
                        teacher_training = training
                        teacher = copy.deepcopy(model).to(device)
                        model, training = fine_tune_chsic_kd(
                            model, teacher, train, val, args, device, seed
                        )
                        training["teacher_best_epoch"] = teacher_training["best_epoch"]
                        training["teacher_best_val_trial_balanced_accuracy"] = teacher_training[
                            "best_val_trial_balanced_accuracy"
                        ]
                        del teacher
                    train_logits, train_features = collect_outputs(
                        model, train["de"], device, args.eval_batch_size
                    )
                    probe_test_logits, probe_test_features = collect_outputs(
                        model, probe_test["de"], device, args.eval_batch_size
                    )
                    test_logits, test_features = collect_outputs(
                        model, test["de"], device, args.eval_batch_size
                    )
                    test_trials = aggregate_trials(test, test_logits)
                    metrics = classification_metrics(test_trials["y"], test_trials["values"])
                    probe_accuracy, probe_rows = conditional_probe(
                        train,
                        train_features,
                        probe_test,
                        probe_test_features,
                        corpus.labels_by_stimulus,
                    )
                    history = training.pop("history")
                    row = {
                        "dataset": "SEED",
                        "subject_fold": subject_fold,
                        "stimulus_fold": stimulus_fold,
                        "seed": seed,
                        "model": model_id,
                        "model_variant": variant,
                        "adv_lambda": variant_lambda,
                        "training_stage": "post_hoc_development_or_extension",
                        "configuration_sha256": manifest["configuration_sha256"],
                        "input_descriptor_sha256": manifest["input_descriptor_sha256"],
                        **metrics,
                        **training,
                        "conditional_stimulus_probe_accuracy": probe_accuracy,
                        "conditional_stimulus_probe_chance": 0.25,
                        "test_subjects": len(subject_split["test"]),
                        "test_stimuli": len(stimulus_split["test"]),
                        "test_trials": len(test_trials["y"]),
                        "duration_seconds": time.time() - run_started,
                    }
                    if not all(
                        np.isfinite(float(row[key]))
                        for key in (
                            "accuracy",
                            "balanced_accuracy",
                            "log_loss",
                            "conditional_stimulus_probe_accuracy",
                            "duration_seconds",
                        )
                    ):
                        raise FloatingPointError(f"non-finite run metrics: {row}")
                    probe_by_subject = {
                        subject: float(
                            np.mean(
                                [
                                    item["accuracy"]
                                    for item in probe_rows
                                    if int(item["subject"]) == subject
                                ]
                            )
                        )
                        for subject in subject_split["test"]
                    }
                    per_subject: list[dict] = []
                    for subject in subject_split["test"]:
                        mask = test_trials["subject"] == subject
                        subject_metric = classification_metrics(
                            test_trials["y"][mask], test_trials["values"][mask]
                        )
                        subject_row = {
                            "dataset": "SEED",
                            "subject_fold": subject_fold,
                            "stimulus_fold": stimulus_fold,
                            "seed": seed,
                            "model": model_id,
                            "model_variant": variant,
                            "adv_lambda": variant_lambda,
                            "subject": subject,
                            **subject_metric,
                            "conditional_stimulus_probe_accuracy": probe_by_subject[subject],
                            "conditional_stimulus_probe_chance": 0.25,
                            "test_trials": int(mask.sum()),
                        }
                        per_subject.append(subject_row)
                        subject_rows.append(subject_row)
                    directory.mkdir(parents=True, exist_ok=True)
                    history_path = directory / "training_history.csv"
                    subject_path = directory / "subject_metrics.csv"
                    probe_path = directory / "conditional_stimulus_probe.csv"
                    write_csv_atomic(history_path, history)
                    write_csv_atomic(subject_path, per_subject)
                    write_csv_atomic(
                        probe_path,
                        [
                            {
                                "subject_fold": subject_fold,
                                "stimulus_fold": stimulus_fold,
                                "seed": seed,
                                "model": model_id,
                                **item,
                            }
                            for item in probe_rows
                        ],
                    )
                    representation_hashes: dict[str, str] = {}
                    if args.export_representations:
                        checkpoint_path = directory / "best_model_state.pt"
                        checkpoint_temporary = checkpoint_path.with_suffix(".pt.partial")
                        torch.save(model.state_dict(), checkpoint_temporary)
                        checkpoint_temporary.replace(checkpoint_path)

                        train_trials = aggregate_trials(train, train_features)
                        probe_trials = aggregate_trials(probe_test, probe_test_features)
                        test_feature_trials = aggregate_trials(test, test_features)
                        representation_path = directory / "trial_representations.npz"
                        temporary_representation = representation_path.with_suffix(".npz.partial")
                        with temporary_representation.open("wb") as handle:
                            np.savez_compressed(
                                handle,
                                train_trial_features=train_trials["values"],
                                train_y=train_trials["y"],
                                train_subject=train_trials["subject"],
                                train_session=train_trials["session"],
                                train_stimulus=train_trials["stimulus"],
                                probe_trial_features=probe_trials["values"],
                                probe_y=probe_trials["y"],
                                probe_subject=probe_trials["subject"],
                                probe_session=probe_trials["session"],
                                probe_stimulus=probe_trials["stimulus"],
                                test_trial_features=test_feature_trials["values"],
                                test_y=test_feature_trials["y"],
                                test_subject=test_feature_trials["subject"],
                                test_session=test_feature_trials["session"],
                                test_stimulus=test_feature_trials["stimulus"],
                                base_test_trial_logits=test_trials["values"],
                            )
                        temporary_representation.replace(representation_path)
                        representation_hashes = {
                            "checkpoint_sha256": sha256(checkpoint_path),
                            "trial_representations_sha256": sha256(representation_path),
                        }
                    run_manifest = {
                        "status": "PASS",
                        "subject_fold": subject_fold,
                        "stimulus_fold": stimulus_fold,
                        "seed": seed,
                        "model": model_id,
                        "model_variant": variant,
                        "adv_lambda": variant_lambda,
                        "gradient_guard": guard_mode,
                        "warmup_fraction": args.warmup_fraction,
                        "train_subjects": subject_split["train"],
                        "val_subjects": subject_split["val"],
                        "test_subjects": subject_split["test"],
                        "train_stimuli_zero_based": stimulus_split["train"],
                        "test_stimuli_zero_based": stimulus_split["test"],
                        "subject_disjoint": True,
                        "stimulus_disjoint": True,
                        "normalization_fit": "training_subjects_x_training_stimuli_only",
                        "test_subject_and_test_stimulus_used_for_fit": False,
                        "metrics": row,
                        "training_history_sha256": sha256(history_path),
                        "subject_metrics_sha256": sha256(subject_path),
                        "conditional_stimulus_probe_sha256": sha256(probe_path),
                        "representations_exported": bool(args.export_representations),
                        **representation_hashes,
                    }
                    write_json_atomic(completed_manifest, run_manifest)
                    metric_rows.append(row)
                    print(
                        f"[done] sf={subject_fold} vf={stimulus_fold} seed={seed} "
                        f"model={model_id} bacc={metrics['balanced_accuracy']:.4f} "
                        f"probe={probe_accuracy:.4f} sec={row['duration_seconds']:.1f}",
                        flush=True,
                    )
                    del train_logits, probe_test_logits, test_logits
                    del train_features, probe_test_features, test_features
                    del model
                    if device.type == "mps":
                        torch.mps.empty_cache()
                    elif device.type == "cuda":
                        torch.cuda.empty_cache()
            del train, val, test, probe_test

    expected = len(subject_folds) * len(stimulus_folds) * len(seeds) * len(models)
    if len(metric_rows) != expected:
        raise AssertionError(f"expected {expected} completed runs, got {len(metric_rows)}")
    write_csv_atomic(bundle / "metrics.csv", metric_rows)
    write_csv_atomic(bundle / "subject_metrics.csv", subject_rows)
    manifest.update(
        {
            "status": "PASS",
            "expected_runs": expected,
            "completed_runs": len(metric_rows),
            "elapsed_seconds": time.time() - started,
            "metrics_sha256": sha256(bundle / "metrics.csv"),
            "subject_metrics_sha256": sha256(bundle / "subject_metrics.csv"),
        }
    )
    write_json_atomic(bundle / "manifest.json", manifest)
    print(
        json.dumps({"status": "PASS", "completed_runs": len(metric_rows), "bundle": str(bundle)}),
        flush=True,
    )


if __name__ == "__main__":
    main()
