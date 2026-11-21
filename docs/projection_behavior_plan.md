# EC-SI supplementary projection and behavior analysis, version 1

Status: AUTHORIZED; fixed before execution of the new analyses. These are post-hoc descriptive analyses, separate from the frozen 20/40-comparison families. No new confirmatory p values or equivalence claims.

## Question and scope

Compare a simple, emotion-class-agnostic projection against the same unprojected frozen CE representation, using matched readouts. Analyze seen versus held-out stimulus classification on the same unseen participants with the original saved emotion head. Reuse all 108 FACED CE and 48 SEED CE fits for projection; all 1,188 FACED and 528 SEED fits for behavior. The sampling units remain the 98 FACED and 12 SEED evaluation participants; seeds, folds, clips and sessions are repeated measurements.

## Fixed projection

For each CE fit, standardize its 64-dimensional trial representations using training participants and training-visible stimuli only. Fit global stimulus one-hot ridge regression with an unpenalized intercept and objective mean squared error plus alpha=1 times squared coefficient norm. Its input arguments contain no emotion labels. Compute the right nullspace of the transposed coefficient matrix using float64 SVD and relative singular threshold 1e-10. Project into that nullspace, without selecting rank or strength from validation or test outcomes. This is a one-pass global linear projection motivated by nullspace removal, not iterative INLP and not a guarantee of all identity information removal.

Fit identical alpha=1 emotion ridge readouts on original-standardized and projected training representations, including training-only label-mean intercepts. Report both held-out classification and conditional nearest-centroid identity Probe. The Probe refits training centroids independently in each representation space; no additional coordinate rescaling after projection. The conditional Probe uses the true category to restrict candidate clips, as in the main protocol. Emotion labels enter the downstream readers, never projection fitting. Native neural-head CE accuracy is a contextual reference; changing to a ridge reader prevents attributing native-versus-projected differences to projection alone.

## Behavior

Use the saved neural emotion head without fitting anything. Seen clips: probe-role representations of evaluation participants on training-visible clips. Unseen clips: test-role representations of the same people on held-out clips, using saved test logits. Verify the checkpoint head reproduces those logits within existing numerical tolerances. Retain ambiguous arithmetic-boundary decisions in the audit; do not exclude outcomes. Average repeated predictions within participant, clip and session (SEED), then sessions within clip, clips within assigned category, and categories within participant. Report mean seen minus unseen balanced accuracy and class recalls, with paired whole-participant bootstrap intervals. Both statuses contain the complete fixed clip set across the retained partitions. Differences are descriptive: changing stimulus roles changes the training fit, and content/difficulty and development reuse preclude a causal shortcut interpretation.

## Uncertainty, integrity and resources

Use 100,000 participant bootstrap resamples with RNG 20261120, independently reset for each reported estimand. Paired contrasts resample participant differences. These intervals are nonsimultaneous descriptive intervals. Retain all complete planned fits, including null or unfavorable outcomes. Check participant/stimulus roles, manifest hashes, finite values, orthogonality/idempotence and removed decoder directions. A smoke run is engineering evidence only, then the complete fixed matrix. No tuning after the smoke.

Use the previously authorized lab812 environment and existing lab PyTorch image, CPU-only (GPU=0), one serial process, two numerical threads, at most 3,600 seconds for this analysis job, no package installation, no GPU retraining, no automatic retry. Preserve job UID, terminal state, stdout, source/plan hashes and per-fit output records. Stop on role/hash or numerical failure and retain evidence. Restricted representations, participant-level outputs and checkpoints remain private; public code and group summaries contain no raw EEG or participant identifiers.

## Evidence boundaries

Inputs are the immutable completed-curve run bundles and reused binary SEED bundles. New code and outputs live under ecsi_projection_behavior_20261120_v1. Existing audited releases remain unchanged. Historical Global-1 has no retained training runner and is not advertised as end-to-end reproducible. Literature: Ravfogel et al., ACL 2020, DOI 10.18653/v1/2020.acl-main.647. Statistical interpretation follows the ASA 2016 statement; non-significance is not equivalence, and a large adjusted p value is not the probability of a null hypothesis.
