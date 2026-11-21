#!/usr/bin/env python3
"""Export compact, row-bound SEED DE sidecars without copying raw EEG windows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from run_seed_ecsi_joint_holdout import metadata_sha256, sha256


def array_sha256(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    paths = sorted(args.source_dir.glob("subject??_session?.npz"))
    if len(paths) != 45:
        raise ValueError(f"expected 45 SEED files, found {len(paths)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for source in paths:
        with np.load(source, allow_pickle=False) as data:
            required = {
                "de_features",
                "y",
                "trial_id",
                "subject",
                "session",
                "window_id",
            }
            if not required <= set(data.files):
                raise ValueError(f"{source} lacks {sorted(required - set(data.files))}")
            de = data["de_features"].astype(np.float32)
            metadata = {key: np.asarray(data[key]) for key in required - {"de_features"}}
        if de.ndim != 3 or de.shape[1:] != (62, 5):
            raise ValueError(f"unexpected DE shape in {source}: {de.shape}")
        row_hash = metadata_sha256(metadata)
        destination = args.output_dir / source.name
        temporary = destination.with_suffix(destination.suffix + ".partial")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                de_features=de,
                metadata_sha256=np.asarray(row_hash),
            )
        temporary.replace(destination)
        rows.append(
            {
                "source_name": source.name,
                "source_sha256": sha256(source),
                "rows": int(len(de)),
                "metadata_sha256": row_hash,
                "de_array_sha256": array_sha256(de),
                "sidecar_bytes": destination.stat().st_size,
                "sidecar_sha256": sha256(destination),
            }
        )
        print(f"[exported] {source.name} rows={len(de)}", flush=True)

    manifest = {
        "status": "PASS",
        "format": "SEED_DE_SIDECAR_V1",
        "files": rows,
        "file_count": len(rows),
        "total_bytes": sum(int(row["sidecar_bytes"]) for row in rows),
    }
    manifest_path = args.output_dir / "manifest.json"
    temporary = manifest_path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    print(json.dumps({"status": "PASS", "manifest": str(manifest_path)}))


if __name__ == "__main__":
    main()
