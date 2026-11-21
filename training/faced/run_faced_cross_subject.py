#!/usr/bin/env python3
"""Frozen subject-independent FACED validation for EEG emotion models."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from run_seed_cross_session import (
    build_model,
    collect_logits,
    resolve_device,
    set_seed,
    train_model,
)


N_SUBJECTS = 123
N_CHANNELS = 32
N_SAMPLES = 250
N_CLASSES = 9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--mapping-csv", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--models", default="de_conformer,multiscale_tcn,eegnet,tsception")
    parser.add_argument("--folds", default="1,2,3,4,5")
    parser.add_argument("--seeds", default="20260804,20260805,20260806")
    parser.add_argument("--split-seed", type=int, default=20260811)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--seconds-per-trial", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-train-batches", type=int, default=100)
    parser.add_argument("--clip-robust-z", type=float, default=20.0)
    parser.add_argument("--transformer-depth", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=10)
    parser.add_argument("--temporal-depth", type=int, default=3)
    parser.add_argument("--temporal-downsample", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu", "mps"], default="auto")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--consolidate-only", action="store_true")
    return parser.parse_args()


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def load_mapping(path: Path) -> np.ndarray:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if [int(row["video_index"]) for row in rows] != list(range(1, 29)):
        raise ValueError("Mapping must contain video_index 1..28 in order")
    labels = np.asarray([int(row["emotion_id"]) for row in rows], dtype=np.int64)
    if sorted(set(labels.tolist())) != list(range(N_CLASSES)):
        raise ValueError("Mapping must contain all nine emotion IDs 0..8")
    return labels


def outer_folds(split_seed: int) -> list[np.ndarray]:
    permutation = np.random.default_rng(split_seed).permutation(N_SUBJECTS)
    return [np.sort(part.astype(np.int64)) for part in np.array_split(permutation, 5)]


def split_subjects(fold_index: int, split_seed: int, val_fraction: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    folds = outer_folds(split_seed)
    test_subjects = folds[fold_index]
    remaining = np.setdiff1d(np.arange(N_SUBJECTS, dtype=np.int64), test_subjects)
    shuffled = np.random.default_rng(split_seed + 1000 + fold_index).permutation(remaining)
    n_val = max(1, int(round(len(remaining) * val_fraction)))
    val_subjects = np.sort(shuffled[:n_val])
    train_subjects = np.sort(shuffled[n_val:])
    if set(train_subjects) & set(val_subjects) or set(train_subjects) & set(test_subjects) or set(val_subjects) & set(test_subjects):
        raise AssertionError("Subject split overlap detected")
    return train_subjects, val_subjects, test_subjects


def load_subject(
    data_root: Path,
    subject_id: int,
    labels_by_trial: np.ndarray,
    seconds_per_trial: int,
) -> dict[str, np.ndarray]:
    processed_path = data_root / "Processed_data" / f"sub{subject_id:03d}.pkl"
    de_path = data_root / "EEG_Features" / "DE" / f"sub{subject_id:03d}.pkl.pkl"
    with processed_path.open("rb") as handle:
        processed = np.asarray(pickle.load(handle), dtype=np.float32)
    with de_path.open("rb") as handle:
        de = np.asarray(pickle.load(handle), dtype=np.float32)
    if processed.shape != (28, 32, 7500):
        raise ValueError(f"Unexpected processed shape for subject {subject_id}: {processed.shape}")
    if de.shape != (28, 32, 30, 5):
        raise ValueError(f"Unexpected DE shape for subject {subject_id}: {de.shape}")
    seconds = min(seconds_per_trial, 30)
    x = processed[:, :N_CHANNELS, : seconds * N_SAMPLES]
    x = x.reshape(28, N_CHANNELS, seconds, N_SAMPLES).transpose(0, 2, 1, 3)
    x = np.ascontiguousarray(x.reshape(28 * seconds, N_CHANNELS, N_SAMPLES))
    de_windows = de[:, :N_CHANNELS, :seconds, :].transpose(0, 2, 1, 3)
    de_windows = np.ascontiguousarray(de_windows.reshape(28 * seconds, N_CHANNELS, 5))
    return {
        "x": x,
        "de": de_windows,
        "y": np.repeat(labels_by_trial, seconds),
        "subject": np.full(28 * seconds, subject_id, dtype=np.int16),
        "trial": np.repeat(np.arange(28, dtype=np.int16), seconds),
        "second": np.tile(np.arange(seconds, dtype=np.int8), 28),
    }


def load_subject_group(
    data_root: Path,
    subject_ids: np.ndarray,
    labels_by_trial: np.ndarray,
    seconds_per_trial: int,
) -> dict[str, np.ndarray]:
    groups = [load_subject(data_root, int(subject), labels_by_trial, seconds_per_trial) for subject in subject_ids]
    return {key: np.concatenate([group[key] for group in groups], axis=0) for key in groups[0]}


def fit_streaming_channel_scaler(x: np.ndarray, chunk_size: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    total = np.zeros(N_CHANNELS, dtype=np.float64)
    total_sq = np.zeros(N_CHANNELS, dtype=np.float64)
    count = 0
    for start in range(0, len(x), chunk_size):
        chunk = x[start : start + chunk_size].astype(np.float64, copy=False)
        total += chunk.sum(axis=(0, 2))
        total_sq += np.square(chunk).sum(axis=(0, 2))
        count += chunk.shape[0] * chunk.shape[2]
    mean = total / count
    variance = np.maximum(total_sq / count - np.square(mean), 1e-12)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def normalize_in_place(x: np.ndarray, mean: np.ndarray, scale: np.ndarray, clip: float) -> np.ndarray:
    x -= mean[None, :, None]
    x /= scale[None, :, None]
    np.clip(x, -clip, clip, out=x)
    return x[:, None, :, :]


def fit_de_scaler(de: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = de.astype(np.float64, copy=False).mean(axis=0)
    scale = de.astype(np.float64, copy=False).std(axis=0)
    return mean.astype(np.float32), np.maximum(scale, 1e-6).astype(np.float32)


def normalize_de_in_place(
    de: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    clip: float,
) -> np.ndarray:
    de -= mean[None, :, :]
    de /= scale[None, :, :]
    np.clip(de, -clip, clip, out=de)
    return de


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities


def metric_values(y_true: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    predictions = logits.argmax(axis=1)
    recalls = []
    f1_values = []
    for class_id in range(N_CLASSES):
        true_positive = int(((y_true == class_id) & (predictions == class_id)).sum())
        false_positive = int(((y_true != class_id) & (predictions == class_id)).sum())
        false_negative = int(((y_true == class_id) & (predictions != class_id)).sum())
        recalls.append(true_positive / max(1, true_positive + false_negative))
        precision = true_positive / max(1, true_positive + false_positive)
        recall = recalls[-1]
        f1_values.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    probabilities = softmax(logits)
    log_loss = -np.log(np.clip(probabilities[np.arange(len(y_true)), y_true], 1e-12, 1.0)).mean()
    observed = float((predictions == y_true).mean())
    true_counts = np.bincount(y_true, minlength=N_CLASSES).astype(np.float64)
    predicted_counts = np.bincount(predictions, minlength=N_CLASSES).astype(np.float64)
    expected = float(np.dot(true_counts, predicted_counts) / max(1, len(y_true) ** 2))
    kappa = 0.0 if expected >= 1.0 else (observed - expected) / (1.0 - expected)
    return {
        "accuracy": observed,
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1_values)),
        "kappa": float(kappa),
        "log_loss": float(log_loss),
    }


def aggregate_trial_logits(test: dict[str, np.ndarray], logits: np.ndarray) -> dict[str, np.ndarray]:
    keys = np.column_stack([test["subject"], test["trial"]])
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
    trial_logits = np.empty((len(unique_keys), logits.shape[1]), dtype=np.float32)
    trial_y = np.empty(len(unique_keys), dtype=np.int64)
    for index in range(len(unique_keys)):
        mask = inverse == index
        labels = np.unique(test["y"][mask])
        if len(labels) != 1:
            raise AssertionError(f"Trial {tuple(unique_keys[index])} contains multiple labels")
        trial_y[index] = int(labels[0])
        trial_logits[index] = logits[mask].mean(axis=0)
    return {
        "y": trial_y,
        "logits": trial_logits,
        "subject": unique_keys[:, 0].astype(np.int16),
        "trial": unique_keys[:, 1].astype(np.int16),
    }


def model_args(base: argparse.Namespace, model_name: str) -> argparse.Namespace:
    args = copy.copy(base)
    args.sampling_rate = 250
    tcn_models = {
        "multiscale_tcn",
        "ablation_coarse_resolution",
        "ablation_single_scale",
        "ablation_no_de",
        "ablation_mean_pool",
    }
    args.temporal_backbone = "multiscale_tcn" if model_name in tcn_models else "transformer"
    args.temporal_kernels = "7,7,7" if model_name == "ablation_single_scale" else "3,7,15"
    args.token_pooling = "mean" if model_name == "ablation_mean_pool" else "mean_max"
    if model_name == "ablation_coarse_resolution":
        args.temporal_downsample = 5
    args.de_features = model_name in ({"de_conformer", "conformer"} | tcn_models) and model_name != "ablation_no_de"
    args.dann = False
    args.dann_lambda = 0.0
    args.contrastive = False
    args.contrastive_alpha = 0.0
    args.contrastive_temperature = 0.07
    args.calibration_learning_rate = 0.0
    args.calibration_augment = False
    args.augment_amplitude = 0.0
    args.augment_noise_std = 0.0
    args.augment_channel_dropout = 0.0
    args.mamba_d_state = 16
    args.mamba_d_conv = 4
    args.mamba_expand = 2
    args.adapter_depth = 2
    args.adapter_init_scale = 0.05
    return args


def write_csv_atomic(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows supplied for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def run_dir(bundle_dir: Path, fold: int, seed: int, model: str) -> Path:
    return bundle_dir / "runs" / f"fold{fold}_seed{seed}_{model}"


def run_complete(bundle_dir: Path, fold: int, seed: int, model: str) -> bool:
    directory = run_dir(bundle_dir, fold, seed, model)
    manifest_path = directory / "run_manifest.json"
    prediction_path = directory / "predictions.npz"
    subject_path = directory / "subject_metrics.csv"
    if not manifest_path.is_file() or not prediction_path.is_file() or not subject_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        manifest.get("status") == "PASS"
        and manifest.get("prediction_sha256") == sha256(prediction_path)
        and manifest.get("subject_metrics_sha256") == sha256(subject_path)
    )


def consolidate_results(
    bundle_dir: Path,
    folds: list[int],
    seeds: list[int],
    models: list[str],
) -> tuple[Path | None, Path | None]:
    metric_rows: list[dict[str, object]] = []
    subject_rows: list[dict[str, object]] = []
    for fold in folds:
        for seed in seeds:
            for model in models:
                if not run_complete(bundle_dir, fold, seed, model):
                    continue
                directory = run_dir(bundle_dir, fold, seed, model)
                manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
                metric_rows.append(manifest["metrics"])
                subject_rows.extend(csv.DictReader((directory / "subject_metrics.csv").open(encoding="utf-8")))
    metrics_path = bundle_dir / "metrics.csv"
    subject_metrics_path = bundle_dir / "subject_metrics.csv"
    if metric_rows:
        write_csv_atomic(metrics_path, metric_rows)
    if subject_rows:
        write_csv_atomic(subject_metrics_path, subject_rows)
    return (metrics_path if metric_rows else None, subject_metrics_path if subject_rows else None)


def save_run_predictions(
    path: Path,
    test: dict[str, np.ndarray],
    logits: np.ndarray,
    trials: dict[str, np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            y=test["y"],
            logits=logits.astype(np.float32),
            subject=test["subject"],
            trial=test["trial"],
            second=test["second"],
            trial_y=trials["y"],
            trial_logits=trials["logits"],
            trial_subject=trials["subject"],
            trial_index=trials["trial"],
        )
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if not 0 < args.validation_fraction < 0.5:
        raise ValueError("validation_fraction must be in (0, 0.5)")
    if not 1 <= args.seconds_per_trial <= 30:
        raise ValueError("seconds_per_trial must be in 1..30")
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    allowed = {
        "conformer",
        "de_conformer",
        "multiscale_tcn",
        "eegnet",
        "tsception",
        "ablation_coarse_resolution",
        "ablation_single_scale",
        "ablation_no_de",
        "ablation_mean_pool",
    }
    if not models or not set(models) <= allowed:
        raise ValueError(f"models must be a subset of {sorted(allowed)}")
    folds = parse_int_list(args.folds)
    seeds = parse_int_list(args.seeds)
    if not folds or not set(folds) <= set(range(1, 6)):
        raise ValueError("folds must be a subset of 1..5")
    if not seeds:
        raise ValueError("At least one seed is required")
    if args.smoke and args.consolidate_only:
        raise ValueError("--smoke and --consolidate-only are mutually exclusive")
    if args.smoke:
        folds = folds[:1]
        seeds = seeds[:1]
        args.epochs = 1
        args.patience = 1
        args.max_train_batches = 2
        args.seconds_per_trial = min(args.seconds_per_trial, 2)

    data_root = args.data_root.resolve()
    mapping_csv = args.mapping_csv.resolve()
    bundle_dir = args.bundle_dir.resolve()
    bundle_dir.mkdir(parents=True, exist_ok=True)
    labels_by_trial = load_mapping(mapping_csv)
    device = resolve_device(args.device)
    started = time.time()

    manifest = {
        "status": "RUNNING",
        "data_root": str(data_root),
        "mapping_csv": str(mapping_csv),
        "mapping_sha256": sha256(mapping_csv),
        "models": models,
        "folds": folds,
        "seeds": seeds,
        "split_seed": args.split_seed,
        "validation_fraction": args.validation_fraction,
        "seconds_per_trial": args.seconds_per_trial,
        "independent_unit": "subject",
        "primary_prediction_unit": "video_trial_mean_logit",
        "secondary_prediction_unit": "one_second_window",
        "sampling_rate_hz": 250,
        "channels": N_CHANNELS,
        "classes": N_CLASSES,
        "device": str(device),
        "smoke": bool(args.smoke),
        "consolidate_only": bool(args.consolidate_only),
    }
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    if args.consolidate_only:
        expected_runs = len(folds) * len(seeds) * len(models)
        completed_in_scope = sum(
            run_complete(bundle_dir, fold, seed, model)
            for fold in folds for seed in seeds for model in models
        )
        metrics_path, subject_metrics_path = consolidate_results(bundle_dir, folds, seeds, models)
        manifest.update({
            "status": "PASS" if completed_in_scope == expected_runs else "INCOMPLETE",
            "expected_runs": expected_runs,
            "completed_runs": completed_in_scope,
            "elapsed_seconds": time.time() - started,
            "metrics_sha256": sha256(metrics_path) if metrics_path else None,
            "subject_metrics_sha256": sha256(subject_metrics_path) if subject_metrics_path else None,
        })
        temporary = bundle_dir / "manifest.json.partial"
        temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        temporary.replace(bundle_dir / "manifest.json")
        print(json.dumps({
            "status": manifest["status"],
            "completed_runs": completed_in_scope,
            "expected_runs": expected_runs,
        }), flush=True)
        if manifest["status"] != "PASS":
            raise SystemExit(1)
        return

    for fold in folds:
        train_subjects, val_subjects, test_subjects = split_subjects(
            fold - 1, args.split_seed, args.validation_fraction
        )
        train = load_subject_group(data_root, train_subjects, labels_by_trial, args.seconds_per_trial)
        val = load_subject_group(data_root, val_subjects, labels_by_trial, args.seconds_per_trial)
        test = load_subject_group(data_root, test_subjects, labels_by_trial, args.seconds_per_trial)
        mean, scale = fit_streaming_channel_scaler(train["x"])
        train_x = normalize_in_place(train["x"], mean, scale, args.clip_robust_z)
        val_x = normalize_in_place(val["x"], mean, scale, args.clip_robust_z)
        test_x = normalize_in_place(test["x"], mean, scale, args.clip_robust_z)
        train_de = train["de"].astype(np.float32, copy=False)
        val_de = val["de"].astype(np.float32, copy=False)
        test_de = test["de"].astype(np.float32, copy=False)
        de_mean, de_scale = fit_de_scaler(train_de)
        train_de = normalize_de_in_place(train_de, de_mean, de_scale, args.clip_robust_z)
        val_de = normalize_de_in_place(val_de, de_mean, de_scale, args.clip_robust_z)
        test_de = normalize_de_in_place(test_de, de_mean, de_scale, args.clip_robust_z)

        for seed in seeds:
            for model_name in models:
                key = (fold, seed, model_name)
                if run_complete(bundle_dir, *key):
                    print(f"[skip] fold={fold} seed={seed} model={model_name}", flush=True)
                    continue
                current_args = model_args(args, model_name)
                build_name = (
                    "conformer"
                    if model_name in {
                        "conformer",
                        "de_conformer",
                        "multiscale_tcn",
                        "ablation_coarse_resolution",
                        "ablation_single_scale",
                        "ablation_no_de",
                        "ablation_mean_pool",
                    }
                    else model_name
                )
                set_seed(seed)
                model = build_model(current_args, build_name, N_CHANNELS, N_SAMPLES, N_CLASSES).to(device)
                uses_de = current_args.de_features
                run_start = time.time()
                model, training_info = train_model(
                    model,
                    train_x,
                    train["y"],
                    val_x,
                    val["y"],
                    train_de if uses_de else None,
                    val_de if uses_de else None,
                    None,
                    None,
                    current_args,
                    device,
                    seed,
                )
                logits = collect_logits(
                    model,
                    test_x,
                    device,
                    current_args.eval_batch_size,
                    test_de if uses_de else None,
                )
                trials = aggregate_trial_logits(test, logits)
                metrics = metric_values(trials["y"], trials["logits"])
                window_metrics = metric_values(test["y"], logits)
                current_run_dir = run_dir(bundle_dir, fold, seed, model_name)
                current_run_dir.mkdir(parents=True, exist_ok=True)
                prediction_path = current_run_dir / "predictions.npz"
                save_run_predictions(prediction_path, test, logits, trials)
                metric_row: dict[str, object] = {
                    "fold": fold,
                    "seed": seed,
                    "model": model_name,
                    **metrics,
                    **{f"window_{key}": value for key, value in window_metrics.items()},
                    "best_epoch": training_info["best_epoch"],
                    "best_val_accuracy": training_info["best_val_acc"],
                    "trainable_params": training_info["trainable_params"],
                    "train_subjects": len(train_subjects),
                    "val_subjects": len(val_subjects),
                    "test_subjects": len(test_subjects),
                    "train_windows": len(train["y"]),
                    "val_windows": len(val["y"]),
                    "test_windows": len(test["y"]),
                    "test_trials": len(trials["y"]),
                    "seconds_per_trial": args.seconds_per_trial,
                    "duration_seconds": time.time() - run_start,
                    "prediction_file": str(prediction_path),
                    "prediction_sha256": sha256(prediction_path),
                }
                subject_rows: list[dict[str, object]] = []
                for subject_id in test_subjects:
                    trial_mask = trials["subject"] == subject_id
                    window_mask = test["subject"] == subject_id
                    subject_trial_metrics = metric_values(trials["y"][trial_mask], trials["logits"][trial_mask])
                    subject_window_metrics = metric_values(test["y"][window_mask], logits[window_mask])
                    subject_rows.append({
                        "fold": fold,
                        "seed": seed,
                        "model": model_name,
                        "subject": int(subject_id),
                        **subject_trial_metrics,
                        **{f"window_{key}": value for key, value in subject_window_metrics.items()},
                        "trials": int(trial_mask.sum()),
                        "windows": int(window_mask.sum()),
                    })
                subject_path = current_run_dir / "subject_metrics.csv"
                write_csv_atomic(subject_path, subject_rows)
                run_manifest = {
                    "status": "PASS",
                    "fold": fold,
                    "seed": seed,
                    "model": model_name,
                    "train_subject_ids": train_subjects.tolist(),
                    "val_subject_ids": val_subjects.tolist(),
                    "test_subject_ids": test_subjects.tolist(),
                    "normalization": {
                        "raw_eeg": "train_subject_channel_mean_std",
                        "de": "train_subject_channel_band_mean_std",
                        "clip_z": float(args.clip_robust_z),
                        "test_data_used_for_fit": False,
                    },
                    "prediction_unit": "video_trial_mean_logit",
                    "metrics": metric_row,
                    "prediction_sha256": sha256(prediction_path),
                    "subject_metrics_sha256": sha256(subject_path),
                }
                run_manifest_path = current_run_dir / "run_manifest.json"
                run_manifest_tmp = run_manifest_path.with_suffix(run_manifest_path.suffix + ".partial")
                run_manifest_tmp.write_text(json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8")
                run_manifest_tmp.replace(run_manifest_path)
                print(
                    f"[done] fold={fold} seed={seed} model={model_name} "
                    f"trial_accuracy={metrics['accuracy']:.6f} trial_macro_f1={metrics['macro_f1']:.6f}",
                    flush=True,
                )
                del model, logits
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        del train, val, test, train_x, val_x, test_x, train_de, val_de, test_de

    expected_runs = len(folds) * len(seeds) * len(models)
    completed_in_scope = sum(
        run_complete(bundle_dir, fold, seed, model)
        for fold in folds for seed in seeds for model in models
    )
    metrics_path, subject_metrics_path = consolidate_results(bundle_dir, folds, seeds, models)
    manifest.update({
        "status": "PASS" if completed_in_scope == expected_runs else "INCOMPLETE",
        "expected_runs": expected_runs,
        "completed_runs": completed_in_scope,
        "elapsed_seconds": time.time() - started,
        "metrics_sha256": sha256(metrics_path) if metrics_path else None,
        "subject_metrics_sha256": sha256(subject_metrics_path) if subject_metrics_path else None,
    })
    temporary = bundle_dir / "manifest.json.partial"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(bundle_dir / "manifest.json")
    print(json.dumps({"status": manifest["status"], "completed_runs": completed_in_scope, "expected_runs": expected_runs}), flush=True)


if __name__ == "__main__":
    main()
