#!/usr/bin/env python3
"""Add DE (Differential Entropy) features to existing preprocessed SEED .npz files.

Computes log band-power from FFT as a fast approximation of differential entropy
in five standard EEG frequency bands: delta (1-4), theta (4-8), alpha (8-14),
beta (14-31), gamma (31-50 Hz).

Preserves all existing arrays; adds `de_features` of shape (n_windows, 62, 5).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

# Standard EEG frequency bands (Hz)
BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 14),
    "beta": (14, 31),
    "gamma": (31, 50),
}
N_BANDS = len(BANDS)
FS = 200  # SEED sampling rate after downsampling


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--preprocessed-dir", type=Path, required=True)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def compute_de_features(x: np.ndarray, fs: int = FS) -> np.ndarray:
    """Compute DE features for windows of shape (n_windows, 62, n_samples).

    Returns array of shape (n_windows, 62, 5).
    Uses FFT-based log band-power as a Gaussian DE approximation.
    """
    n_windows, n_channels, n_samples = x.shape
    de = np.zeros((n_windows, n_channels, N_BANDS), dtype=np.float32)

    # FFT along time axis
    X = np.fft.rfft(x, axis=2)  # (n_windows, 62, n_freq)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)
    power = np.abs(X) ** 2  # (n_windows, 62, n_freq)

    for band_idx, (name, (f_low, f_high)) in enumerate(BANDS.items()):
        mask = (freqs >= f_low) & (freqs < f_high)
        band_power = power[:, :, mask].mean(axis=2)  # avg power in band
        # DE for Gaussian: 0.5 * log(2*pi*e * variance)
        # Use log band-power approximation
        de[:, :, band_idx] = np.log(np.maximum(band_power, 1e-12))

    return de


def main() -> None:
    args = parse_args()
    preprocessed_dir = args.preprocessed_dir.resolve()

    files = sorted(preprocessed_dir.glob("subject*.npz"))
    if not files:
        raise FileNotFoundError(f"No subject*.npz found in {preprocessed_dir}")

    print(f"Found {len(files)} files. Computing DE features...")
    for path in files:
        data = dict(np.load(path))
        if "de_features" in data and not args.force:
            print(f"  SKIP {path.name} (already has de_features)")
            continue

        de = compute_de_features(data["x"])
        data["de_features"] = de

        np.savez_compressed(path, **data)
        print(f"  DONE {path.name}: de_features shape={de.shape}")

    print(f"\nAll {len(files)} files processed.")


if __name__ == "__main__":
    main()
