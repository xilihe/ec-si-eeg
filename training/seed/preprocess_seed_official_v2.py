#!/usr/bin/env python3
"""Preprocess SEED official MAT files (named {subject}_{YYYYMMDD}.mat).

Handles the actual SEED file naming convention where sessions are identified
by date rather than sequential session numbers. Groups files by subject,
sorts by date, and assigns session numbers 1-3.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.io import loadmat

# Import from the existing seed_pipeline module
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from seed_pipeline import LABEL_MAP, parse_id_list


TRIAL_PATTERN = re.compile(r"^(?:(?:subject(?P<subject>[0-9]+)_)|(?:[a-zA-Z]+_))?eeg(?P<trial>[0-9]+)$")

# Regex for official SEED file naming: {subject}_{YYYYMMDD}.mat
FILE_PATTERN = re.compile(r"^(\d+)_(\d{8})\.mat$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label-path", type=Path, default=None)
    parser.add_argument("--subjects", default="all")
    parser.add_argument("--window-size", type=int, default=200)
    parser.add_argument("--stride", type=int, default=200)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--report-json", type=Path, default=None)
    return parser.parse_args()


def discover_sessions(raw_dir: Path) -> dict[int, dict[int, Path]]:
    """Scan raw_dir for {subject}_{YYYYMMDD}.mat files and map by date order.

    Returns: {subject: {session: path}} where session 1=earliest date.
    """
    discovered: dict[int, dict[str, Path]] = defaultdict(dict)
    for path in sorted(raw_dir.iterdir()):
        if not path.is_file() or path.suffix != ".mat":
            continue
        match = FILE_PATTERN.match(path.name)
        if match is None:
            continue
        subject = int(match.group(1))
        date_str = match.group(2)
        discovered[subject][date_str] = path

    # Sort dates within each subject and assign session numbers
    result: dict[int, dict[int, Path]] = {}
    for subject, date_map in sorted(discovered.items()):
        dates = sorted(date_map.keys())
        if len(dates) != 3:
            print(f"[WARN] Subject {subject}: expected 3 sessions, found {len(dates)}: {dates}")
        result[subject] = {}
        for idx, date_str in enumerate(dates, start=1):
            result[subject][idx] = date_map[date_str]
            print(f"[discover] subject={subject} session={idx} <- {date_str} ({date_map[date_str].name})")
    return result


def load_labels(label_path: Path) -> np.ndarray:
    data = loadmat(str(label_path), squeeze_me=True, struct_as_record=False)
    candidates = []
    for name, value in data.items():
        if name.startswith("__") or not hasattr(value, "size"):
            continue
        try:
            values = [int(item) for item in np.asarray(value).reshape(-1).tolist()]
        except (TypeError, ValueError):
            continue
        if len(values) == 15 and set(values) == {-1, 0, 1}:
            candidates.append(values)
    if not candidates:
        raise ValueError(f"No SEED label vector found in {label_path}")
    return np.asarray(candidates[0], dtype=np.int64)


def preprocess_session(
    mat_path: Path,
    labels: np.ndarray,
    subject: int,
    session: int,
    window_size: int,
    stride: int,
) -> dict[str, np.ndarray]:
    data = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    windows: list[np.ndarray] = []
    window_labels: list[int] = []
    window_trials: list[int] = []
    found = set()
    for name, value in data.items():
        match = TRIAL_PATTERN.match(name)
        if match is None:
            continue
        if match.group("subject") is not None and int(match.group("subject")) != subject:
            continue
        trial = int(match.group("trial"))
        if not 1 <= trial <= 15:
            continue
        found.add(trial)
        array = np.asarray(value)
        if array.ndim != 2:
            raise ValueError(f"{mat_path.name}:{name} is not 2D")
        if array.shape[0] == 62:
            array = array
        elif array.shape[1] == 62:
            array = array.T
        else:
            raise ValueError(f"{mat_path.name}:{name} has no 62-channel axis")
        if array.shape[0] != 62 or array.shape[1] < window_size:
            raise ValueError(
                f"{mat_path.name}:{name} has shape {array.shape}, needs 62 x >= {window_size}"
            )
        label = int(labels[trial - 1])
        mapped = LABEL_MAP[label]
        for start in range(0, array.shape[1] - window_size + 1, stride):
            windows.append(array[:, start : start + window_size].astype(np.float32, copy=False))
            window_labels.append(mapped)
            window_trials.append(trial)
    missing = [trial for trial in range(1, 16) if trial not in found]
    if missing:
        raise ValueError(f"{mat_path.name} missing trials: {missing}")
    x = np.stack(windows, axis=0)
    window_id = np.arange(1, len(x) + 1, dtype=np.int64)
    return {
        "x": x,
        "y": np.asarray(window_labels, dtype=np.int64),
        "trial_id": np.asarray(window_trials, dtype=np.int64),
        "window_id": window_id,
        "subject": np.full(len(x), subject, dtype=np.int64),
        "session": np.full(len(x), session, dtype=np.int64),
    }


def main() -> None:
    args = parse_args()
    raw_dir = args.raw_dir.resolve()
    label_path = (args.label_path or raw_dir / "label.mat").resolve()

    # Discover files by date order
    session_map = discover_sessions(raw_dir)
    subjects = parse_id_list(args.subjects, 15)
    if set(subjects) - set(session_map.keys()):
        missing = set(subjects) - set(session_map.keys())
        raise ValueError(f"Subjects not found in raw dir: {missing}")

    labels = load_labels(label_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "raw_dir": str(raw_dir),
        "output_dir": str(args.output_dir.resolve()),
        "subjects": subjects,
        "window_size": args.window_size,
        "stride": args.stride,
        "files": [],
    }

    for subject in subjects:
        for session in sorted(session_map[subject].keys()):
            mat_path = session_map[subject][session]
            arrays = preprocess_session(
                mat_path, labels, subject, session, args.window_size, args.stride
            )
            output = args.output_dir / f"subject{subject:02d}_session{session}.npz"
            if output.exists() and not args.force:
                raise FileExistsError(f"{output} exists; use --force to overwrite")
            np.savez_compressed(output, **arrays)
            per_trial = {
                int(trial): int(np.count_nonzero(arrays["trial_id"] == trial))
                for trial in np.unique(arrays["trial_id"])
            }
            total_windows = len(arrays["y"])
            print(
                f"[preprocess] subject={subject} session={session} "
                f"windows={total_windows} trials={len(per_trial)} "
                f"file={mat_path.name} -> {output.name}",
                flush=True,
            )
            summary["files"].append(
                {
                    "path": str(output),
                    "subject": subject,
                    "session": session,
                    "source_file": mat_path.name,
                    "windows": total_windows,
                    "trials": int(len(np.unique(arrays["trial_id"]))),
                    "windows_per_trial": per_trial,
                }
            )

    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(
            json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
        )
        print(f"[report] {args.report_json}")

    total_w = sum(f["windows"] for f in summary["files"])
    total_subj = len({f["subject"] for f in summary["files"]})
    print(f"\nDone: {len(summary['files'])} sessions, {total_subj} subjects, {total_w} total windows")


if __name__ == "__main__":
    main()
