"""Pure-numpy helpers for the trial-safe SEED cross-session protocol.

This module intentionally avoids torch/scipy dependencies so split logic,
RG-TPC aggregation, and metric computation can be unit-tested in any
environment with numpy.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np


LABEL_MAP = {-1: 0, 0: 1, 1: 2}
CLASSES = [0, 1, 2]
METHODS = (
    "mean_logit",
    "majority_vote",
    "entropy_weighted",
    "prototype_only",
    "rg_tpc",
)


def parse_id_list(value: str, upper: int | None = None) -> list[int]:
    if value.strip().lower() == "all":
        if upper is None:
            raise ValueError("'all' requires an explicit upper bound")
        return list(range(1, upper + 1))
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError(f"Empty integer list: {value!r}")
    if upper is not None:
        bad = [item for item in values if item < 1 or item > upper]
        if bad:
            raise ValueError(f"IDs out of range 1..{upper}: {bad}")
    return values


def parse_float_list(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError(f"Empty float list: {value!r}")
    return values


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def normalized_entropy(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    n_classes = probs.shape[-1]
    entropy = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=-1)
    return entropy / math.log(max(2, n_classes))


def class_prototypes(probs: np.ndarray, labels: np.ndarray, n_classes: int) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    global_mean = probs.mean(axis=0)
    prototypes = []
    for class_id in range(n_classes):
        mask = labels == class_id
        if np.any(mask):
            proto = probs[mask].mean(axis=0)
        else:
            proto = global_mean.copy()
        proto = proto / max(float(proto.sum()), 1e-12)
        prototypes.append(proto)
    return np.stack(prototypes, axis=0)


def prototype_scores(prob: np.ndarray, prototypes: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    distances = np.linalg.norm(prototypes - prob[None, :], axis=1)
    scale = float(np.median(distances))
    if scale < 1e-8:
        scale = float(np.mean(distances) + 1e-8)
    scores = -distances / scale
    order = np.argsort(distances)
    closest = float(distances[order[0]])
    second = float(distances[order[1]]) if len(order) > 1 else closest
    margin = max(0.0, min(1.0, (second - closest) / (second + 1e-8)))
    return scores, closest, second, margin


def confidence_norm(prob: np.ndarray) -> float:
    n_classes = len(prob)
    conf = float(np.max(prob))
    return max(0.0, min(1.0, (conf - 1.0 / n_classes) / (1.0 - 1.0 / n_classes)))


def reliability_features(window_logits: np.ndarray, prototypes: np.ndarray) -> dict[str, float]:
    mean_logits = window_logits.mean(axis=0)
    mean_prob = softmax(mean_logits)
    window_pred = np.argmax(window_logits, axis=1)
    counts = np.bincount(window_pred, minlength=len(mean_logits))
    agreement = float(counts.max() / max(1, len(window_pred)))
    entropy_rel = float(1.0 - normalized_entropy(mean_prob))
    _proto_logits, closest, second, proto_margin = prototype_scores(mean_prob, prototypes)
    reliability = float(np.mean([confidence_norm(mean_prob), agreement, entropy_rel, proto_margin]))
    return {
        "confidence": float(np.max(mean_prob)),
        "agreement": agreement,
        "entropy": float(normalized_entropy(mean_prob)),
        "entropy_reliability": entropy_rel,
        "prototype_distance": closest,
        "prototype_second_distance": second,
        "prototype_margin": proto_margin,
        "reliability": reliability,
        "prototype_pred": int(np.argmax(prototype_scores(mean_prob, prototypes)[0])),
    }


def rgtpc_logits(
    window_logits: np.ndarray,
    prototypes: np.ndarray,
    alpha: float,
    tau: float,
) -> tuple[np.ndarray, dict[str, float]]:
    mean_logits = window_logits.mean(axis=0)
    mean_prob = softmax(mean_logits)
    proto_logits, closest, second, proto_margin = prototype_scores(mean_prob, prototypes)
    features = reliability_features(window_logits, prototypes)
    gate = max(0.0, min(1.0, (tau - features["reliability"]) / max(tau, 1e-8)))
    refined = mean_logits + alpha * gate * proto_logits
    features.update(
        {
            "gate": float(gate),
            "selected_alpha": float(alpha),
            "selected_tau": float(tau),
            "prototype_distance": closest,
            "prototype_second_distance": second,
            "prototype_margin": proto_margin,
        }
    )
    return refined, features


def entropy_weighted_logits(window_logits: np.ndarray) -> np.ndarray:
    probs = softmax(window_logits)
    weights = 1.0 - normalized_entropy(probs)
    weights = np.asarray(weights, dtype=np.float64) + 1e-6
    weights = weights / weights.sum()
    return np.sum(window_logits * weights[:, None], axis=0)


def majority_vote_logits(window_logits: np.ndarray) -> np.ndarray:
    mean_logits = window_logits.mean(axis=0)
    preds = np.argmax(window_logits, axis=1)
    counts = np.bincount(preds, minlength=window_logits.shape[1]).astype(np.float64)
    winners = np.flatnonzero(counts == counts.max())
    mean_pred = int(np.argmax(mean_logits))
    if mean_pred in winners:
        counts[mean_pred] += 1e-3
    return counts


def tune_rgtpc(
    val_logits: np.ndarray,
    val_labels: np.ndarray,
    prototypes: np.ndarray,
    alpha_grid: list[float],
    tau_grid: list[float],
) -> tuple[float, float, float]:
    best_acc = -1.0
    best_alpha = 0.0
    best_tau = tau_grid[0]
    for tau in tau_grid:
        for alpha in alpha_grid:
            preds = []
            for logits in val_logits:
                refined, _features = rgtpc_logits(logits[None, :], prototypes, alpha, tau)
                preds.append(int(np.argmax(refined)))
            acc = float(np.mean(np.asarray(preds, dtype=np.int64) == val_labels))
            if acc > best_acc + 1e-12 or (
                abs(acc - best_acc) <= 1e-12 and alpha < best_alpha
            ):
                best_acc = acc
                best_alpha = alpha
                best_tau = tau
    return best_alpha, best_tau, best_acc


def trial_prediction_rows(
    trial_logits_list: list[np.ndarray],
    trial_ids: list[int],
    trial_labels: np.ndarray,
    prototypes: np.ndarray,
    alpha: float,
    tau: float,
) -> list[dict[str, object]]:
    rows = []
    for trial_index, (window_logits, label) in enumerate(
        zip(trial_logits_list, trial_labels), start=1
    ):
        mean_logits = window_logits.mean(axis=0)
        mean_prob = softmax(mean_logits)
        proto_logits, _closest, _second, _margin = prototype_scores(mean_prob, prototypes)
        method_logits = {
            "mean_logit": mean_logits,
            "majority_vote": majority_vote_logits(window_logits),
            "entropy_weighted": entropy_weighted_logits(window_logits),
            "prototype_only": proto_logits,
        }
        refined_logits, features = rgtpc_logits(window_logits, prototypes, alpha, tau)
        method_logits["rg_tpc"] = refined_logits
        trial_id = trial_ids[trial_index - 1] if trial_ids else int(trial_index)
        for method, logits in method_logits.items():
            pred = int(np.argmax(logits))
            rows.append(
                {
                    "trial_id": int(trial_id),
                    "y_true": int(label),
                    "aggregation_method": method,
                    "y_pred": pred,
                    "score_0": float(logits[0]),
                    "score_1": float(logits[1]),
                    "score_2": float(logits[2]),
                    "confidence": float(features["confidence"]),
                    "agreement": float(features["agreement"]),
                    "entropy": float(features["entropy"]),
                    "prototype_distance": float(features["prototype_distance"]),
                    "prototype_margin": float(features["prototype_margin"]),
                    "reliability": float(features["reliability"]),
                    "gate": float(features["gate"]),
                    "selected_alpha": float(alpha),
                    "selected_tau": float(tau),
                }
            )
    return rows


def trial_roles(
    subject: int,
    session: int,
    fold: int,
    trials_by_class: dict[int, list[int]],
    split_seed: int = 20260804,
) -> dict[int, str]:
    rng = np.random.default_rng(split_seed + subject * 1000 + session * 100)
    roles: dict[int, str] = {}
    for class_id, trials in sorted(trials_by_class.items()):
        trials = sorted(trials)
        if len(trials) < 3:
            raise ValueError(f"Class {class_id} needs at least three trials, got {len(trials)}")
        order = rng.permutation(trials).tolist()
        for index, trial_id in enumerate(order):
            if index == fold % len(order):
                roles[trial_id] = "support"
            elif index == (fold + 1) % len(order):
                roles[trial_id] = "validation"
            else:
                roles[trial_id] = "test"
    return roles


def evenly_sample_indices(n: int, count: int) -> np.ndarray:
    if n < count:
        raise ValueError(f"Need at least {count} windows, only {n} available")
    indices = np.linspace(0, n - 1, num=count).round().astype(np.int64)
    return np.unique(indices)


@dataclass
class SplitData:
    x: np.ndarray
    y: np.ndarray
    trial_id: np.ndarray
    window_id: np.ndarray
    de: np.ndarray | None = None


def window_split(
    x: np.ndarray,
    y: np.ndarray,
    trial_id: np.ndarray,
    window_id: np.ndarray,
    roles: dict[int, str],
    shots_per_class: int,
    val_shots_per_class: int,
    classes: list[int] | None = None,
    de: np.ndarray | None = None,
) -> dict[str, SplitData]:
    classes = classes or CLASSES
    trial_ids = np.asarray(trial_id, dtype=np.int64)
    trial_labels = {
        int(value): int(y[np.flatnonzero(trial_ids == value)][0])
        for value in np.unique(trial_ids)
    }
    output: dict[str, SplitData] = {}
    for split_name in ("support", "validation", "test"):
        selected_trials = sorted(
            trial_id for trial_id, role in roles.items() if role == split_name
        )
        mask = np.isin(trial_ids, selected_trials)
        split_x = x[mask]
        split_y = y[mask]
        split_trial = trial_ids[mask]
        split_window = window_id[mask]
        if split_name in ("support", "validation"):
            budget = shots_per_class if split_name == "support" else val_shots_per_class
            chosen_x: list[np.ndarray] = []
            chosen_y: list[np.ndarray] = []
            chosen_trial: list[np.ndarray] = []
            chosen_window: list[np.ndarray] = []
            for class_id in classes:
                class_trials = [
                    trial_id
                    for trial_id in selected_trials
                    if trial_labels.get(trial_id) == class_id
                ]
                if len(class_trials) != 1:
                    raise ValueError(
                        f"{split_name} class {class_id} must have exactly one trial, "
                        f"got {len(class_trials)}"
                    )
                trial = class_trials[0]
                trial_mask = split_trial == trial
                indices = evenly_sample_indices(int(np.count_nonzero(trial_mask)), budget)
                chosen_x.append(split_x[trial_mask][indices])
                chosen_y.append(split_y[trial_mask][indices])
                chosen_trial.append(split_trial[trial_mask][indices])
                chosen_window.append(split_window[trial_mask][indices])
            chosen_de = None
            if de is not None:
                chosen_de_list = []
                for class_id in classes:
                    trial = [t for t in selected_trials if trial_labels.get(t) == class_id][0]
                    indices = evenly_sample_indices(int(np.count_nonzero(split_trial == trial)), budget)
                    chosen_de_list.append(de[mask][split_trial == trial][indices])
                chosen_de = np.concatenate(chosen_de_list, axis=0)
            output[split_name] = SplitData(
                x=np.concatenate(chosen_x, axis=0),
                y=np.concatenate(chosen_y, axis=0),
                trial_id=np.concatenate(chosen_trial, axis=0),
                window_id=np.concatenate(chosen_window, axis=0),
                de=chosen_de,
            )
        else:
            split_de = de[mask] if de is not None else None
            output[split_name] = SplitData(
                x=split_x, y=split_y, trial_id=split_trial, window_id=split_window, de=split_de
            )
    return output


def split_hash(role_assignments: dict[int, str]) -> str:
    payload = "\n".join(
        f"{trial_id},{role}"
        for trial_id, role in sorted(role_assignments.items())
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cohen_kappa(y_true: np.ndarray, y_pred: np.ndarray, classes: list[int]) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if len(y_true) == 0:
        return 0.0
    observed = float(np.mean(y_true == y_pred))
    expected = 0.0
    for class_id in classes:
        expected += float(np.mean(y_true == class_id)) * float(np.mean(y_pred == class_id))
    if abs(1.0 - expected) < 1e-12:
        return 0.0
    return (observed - expected) / (1.0 - expected)


def metrics_for_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list[int] | None = None,
) -> dict[str, float]:
    classes = classes or CLASSES
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    accuracy = float(np.mean(y_true == y_pred)) if len(y_true) else 0.0
    recalls: list[float] = []
    f1s: list[float] = []
    for class_id in classes:
        true_mask = y_true == class_id
        pred_mask = y_pred == class_id
        true_positive = float(np.sum(true_mask & pred_mask))
        recall = true_positive / max(1, float(np.sum(true_mask)))
        precision = true_positive / max(1, float(np.sum(pred_mask)))
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        recalls.append(recall)
        f1s.append(f1)
    return {
        "accuracy": accuracy,
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "kappa": cohen_kappa(y_true, y_pred, classes),
    }
