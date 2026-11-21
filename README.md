# EC-SI EEG: complete controls and supplementary analyses

Training and analysis sources for *Emotion-Conditional Stimulus-Identity Suppression for EEG Emotion Recognition*. The release separates the complete shared-runner controls from historical exploratory results.

The complete matrix contains 1,188 FACED fits (98 evaluation participants) and 528 binary SEED fits (12 evaluation participants). The post-hoc projection control reuses 108 FACED and 48 SEED CE representations. Seen/held-out stimulus analysis uses all 1,716 fits. Participant-level data, EEG, checkpoints and trial representations are not included.

## Environment

Reference training: Python 3.11.10, PyTorch 2.5.1+cu124, NumPy 2.1.2; lab image `lab-pytorch:cuda12.4-py3.11-torch2.5.1`, originally on an RTX 2080 Ti. The lab image is an execution reference, not a publicly downloadable dependency. Install PyTorch 2.5.1 from the official PyTorch distribution for your platform and use `environment/requirements-reference.txt` for the recorded core versions. CUDA builds require the corresponding official PyTorch package index. Dependencies are not installed by the scripts. CPU execution is supported, with different runtime; bitwise equality across devices or library versions is not guaranteed.

`SOURCE_LINEAGE.json` records byte-identical retained sources. The portable launcher, preprocessing helpers and new analysis are additional entry points, identified separately. No fresh full retraining is claimed by packaging this code.

## Data preparation

Obtain FACED from [Synapse syn50614194](https://www.synapse.org/Synapse:syn50614194), subject to its access terms. The data root must contain `Processed_data/sub000.pkl` through `sub122.pkl` and `EEG_Features/DE/sub000.pkl.pkl` through `sub122.pkl.pkl`. The retained loader checks `(28,32,7500)` raw arrays and `(28,32,30,5)` DE arrays, uses the 30 one-second 250-Hz windows and aligns them with the supplied stimulus mapping. Use provider files from a trusted source, since Python pickle loading executes deserialization code.

Obtain the original SEED preprocessed MAT recordings and label file through the [BCMI SEED access page](https://bcmi.sjtu.edu.cn/home/seed/). No SEED files are redistributed. Create working copies in a new destination, then run:

```bash
python training/seed/preprocess_seed_official_v2.py --raw-dir /path/to/SEED/Preprocessed_EEG --label-path /path/to/label.mat --output-dir /path/to/seed_npz
python training/seed/add_de_features.py --preprocessed-dir /path/to/seed_npz
```

The first script sorts each participant's three date-named sessions and creates non-overlapping 200-sample windows. The second adds the study's FFT log-band-power approximation in 1–4, 4–8, 8–14, 14–31 and 31–50 Hz bands, with `log(max(power,1e-12))`. It is not the provider's LDS-smoothed DE feature. The binary runner excludes neutral before fitting, maps negative to 0 and positive to 1, and validates eight training and two test clips. An optional retained sidecar exporter preserves metadata alignment when storage of raw-window copies is undesirable. The loader accepts embedded DE arrays, so sidecars are unnecessary for these commands.

Both training runners fit normalization on training participants and training clips only. Validation participants share the training clip set; test participants and test clips are excluded from that fit. Exact seeds, fold lists and evaluation counts are in `configs/matrix.json`; FACED stimulus partitions and mapping are supplied in `configs/`.

## Train the complete matrix

First inspect a command with `--dry-run`. Omitting that flag starts real computation. Run each condition once for each task, with a new output directory. Each FACED condition runs the applicable CE/EC-SI/Global models serially. CE is fitted only for the standard policy. The SEED standard and 0.3 bundles retain the original source-compatible directory names.

```bash
python run_training.py --task faced --data-root /path/to/FACED --output /path/to/experiments --condition standard --device cuda --dry-run
python run_training.py --task seed_binary --data-root /path/to/seed_npz --output /path/to/experiments --condition standard --device cuda --dry-run
```

For the complete matrix, remove `--dry-run` and repeat with conditions `standard`, `0.1`, `0.3`, `1`, `3` for each task. Default subject folds are 2–5, FACED stimulus folds 4–12, SEED stimulus folds 2–5. Initializations are 20260812/13/14 for FACED and 20260817/18/19 for SEED. Split seeds and all training defaults are in the retained source. The scripts preserve per-fit roles, source hashes, checkpoints, predictions, representations, stopping records and gradient diagnostics. Do not merge outputs created by different source versions. Existing completed runs are detected by the training runners.

## Analyze frozen representations

```bash
python analysis/analyze_projection_behavior.py --experiments /path/to/experiments --output /path/to/new_analysis --plan docs/projection_behavior_plan.md
```

This command reads the complete matrix in the launcher's directory structure, verifies manifests, hashes, roles and saved heads, and writes private participant-level outputs plus group summaries. Use `--smoke` with a separate output directory for an engineering check, which is explicitly excluded from reported results. It requires PyTorch only to read saved checkpoints. The fixed projection fits a global stimulus ridge decoder on standardized training CE features without emotion labels, projects into its weight nullspace once, and refits identical ridge emotion readers before/after projection. Identity centroids are also refitted on training data. No test selection or strength search is used. Full definitions and limitations are in `docs/projection_behavior_plan.md` and the retained `docs/analysis_scope_clarification.md`. SEED seen scores cover ten clips and held-out scores eight across folds 2–5. Its new projection Probe uses equal clips within category, whereas the original curve Probe weights runs equally; their absolute summaries differ. The paired before/after projection comparison uses one fixed aggregation rule.

`analysis/complete_curves/` retains the original validation, aggregation and statistical source. Some original integrity gates require archival block certificates and source freezes beyond a newly generated training matrix; their absence should not be bypassed or labeled PASS. The new analysis command above is portable and verifies directly from the generated per-fit bundles. `results/complete_curves/` contains group-level summaries and all original 20-per-task and joint-40 contrasts for comparison. The fixed inference uses whole participants, not seeds or folds as independent samples.

## Historical boundary

The historical Global-1 versus Global-48 exploration is a separate strategy-level sensitivity. The original Global-1 training runner was not retained; a single exact replay does not establish complete source equivalence. This release provides the executed shared-runner gradient controls and does not claim end-to-end reproduction of that historical exploration. Earlier nonlinear-reader, self-report and restricted-logit analyses also remain historical/post-hoc and are not silently promoted into new confirmatory results.

## Interpretation

FACED's lower-strength identity-readout differences and stronger Global suppression at high strength should be interpreted alongside the category recalls. SEED's unestablished identity advantage is not evidence of equivalence. The seen-minus-unseen gap is descriptive and cannot by itself identify a causal shortcut mechanism. See the manuscript for the development reuse and fixed-stimulus inference boundaries.

## Redistribution

This repository distributes author-provided code and aggregate summaries. It grants no rights to third-party EEG, ratings or derived participant-level records. Obtain source data separately under provider conditions. No new license grant is implied for third-party code; provenance is retained in source headers. Check `SHA256SUMS` for release integrity.
