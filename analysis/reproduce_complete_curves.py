#!/usr/bin/env python3
"""Portable per-bundle audits and original whole-participant curve statistics.

The original laboratory block/resource certificates remain separate historical
evidence. This entry point audits newly generated bundles directly and never
claims to recreate those laboratory certificates or an independent cohort.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'complete_curves'))
from ecsi_curve_statistics_20261116 import complete_comparisons, mean_interval, holm


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def write(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def component_specs(experiments):
    curve = experiments / 'ecsi_final_gap_closure_20261116_v2'
    for condition, prefix in [('standard', 'standard'), ('rho0.1', 'norm01'), ('rho0.3', 'norm03'), ('rho1', 'norm10'), ('rho3', 'norm30')]:
        for model in (('ce', 'ecsi', 'global') if condition == 'standard' else ('ecsi', 'global')):
            name = f'faced_{prefix}_{model}'
            yield 'faced', condition, name, curve / name
    binary = experiments / 'ecsi_binary_completion_20261116_v1'
    for condition, bundle in [('standard', binary / 'formal_original'), ('rho0.1', curve / 'seed_norm01'),
                              ('rho0.3', binary / 'formal_norm'), ('rho1', curve / 'seed_norm10'), ('rho3', curve / 'seed_norm30')]:
        yield 'seed_binary', condition, 'seed_' + condition, bundle


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--experiments', type=Path, help='Audit every generated bundle, then aggregate')
    group.add_argument('--components', type=Path, help='Aggregate existing per-bundle audited components')
    parser.add_argument('--output', required=True, type=Path, help='New directory; existing output is preserved')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    components = args.components.resolve() if args.components else args.output / 'components'
    if args.experiments:
        for task, condition, name, bundle in component_specs(args.experiments.resolve()):
            script = 'audit_faced_curve_bundle_20261116.py' if task == 'faced' else 'analyze_seed_curve_component_20261116.py'
            subprocess.run([sys.executable, str(HERE / 'complete_curves' / script), '--bundle', str(bundle),
                '--condition', condition, '--output', str(components / name)], check=True)
    all_comparisons, input_hashes = [], {}
    for task in ('faced', 'seed_binary'):
        rows = []
        partition_rows, video_rows = [], []
        for _, condition, name, _ in [x for x in component_specs(Path('.')) if x[0] == task]:
            directory = components / name
            proof = directory / ('component_validation.json' if task == 'faced' else 'independent_artifact_validation.json')
            cert = json.loads(proof.read_text())
            assert cert['status'] == 'PASS', proof
            for filename, expected in cert['output_hashes'].items():
                assert sha(directory / filename) == expected, (directory, filename)
            expected_fits = 108 if task == 'faced' else (144 if condition == 'standard' else 96)
            assert cert['fits' if task == 'faced' else 'fit_grid_verified'] == expected_fits
            component_rows = read(directory / 'subject_scores.csv')
            assert all(r['condition'] == condition for r in component_rows)
            rows.extend(component_rows)
            if task == 'faced':
                partition_rows.extend(read(directory / 'partition_subject_scores.csv'))
                video_rows.extend(read(directory / 'video_scores.csv'))
            input_hashes[str(proof)] = sha(proof)
            input_hashes[str(directory / 'subject_scores.csv')] = sha(directory / 'subject_scores.csv')
        n = 98 if task == 'faced' else 12
        assert len(rows) == 11 * n
        comparisons = complete_comparisons(rows, task, n)
        metrics = ('balanced_accuracy', 'probe', 'macro_ovr_auc', *[f'recall_{i}' for i in range(9)]) if task == 'faced' else (
            'balanced_accuracy', 'probe', 'nonneutral_auc', 'negative_recall', 'positive_recall')
        groups = defaultdict(list)
        for row in rows:
            groups[(row['condition'], row['model'])].append(row)
        summaries = []
        for (condition, model), values in groups.items():
            summary = dict(task=task, condition=condition, model=model, participants=n)
            for metric in metrics:
                mean, low, high = mean_interval([float(r[metric]) for r in values])
                summary.update({metric: mean, metric + '_ci_low': low, metric + '_ci_high': high})
            summaries.append(summary)
        directory = args.output / ('analysis_' + task)
        directory.mkdir()
        write(directory / 'subject_scores.csv', rows)
        write(directory / 'condition_summary.csv', summaries)
        write(directory / 'comparisons_twenty.csv', comparisons)
        if task == 'faced':
            from ecsi_curve_stability_20261117 import describe
            for filename, values in describe(partition_rows, video_rows, rows).items():
                write(directory / (filename if filename.endswith('.csv') else filename + '.csv'), values)
            write(directory / 'stimulus_partition_subject_scores.csv', partition_rows)
        all_comparisons.extend(comparisons)
    for row, adjusted in zip(all_comparisons, holm([r['raw_p'] for r in all_comparisons])):
        row['holm_forty_p'] = adjusted
    write(args.output / 'comparisons_forty.csv', all_comparisons)
    result = dict(status='PASS', scope='Direct per-bundle numerical audits and original participant-level summary statistics',
        mode='audit_generated_bundles_and_aggregate' if args.experiments else 'aggregate_existing_audited_components',
        archival_laboratory_block_certificates_recreated=False, independently_retrained_models_claimed=False,
        independent_new_cohort_claimed=False, fits={'faced': 1188, 'seed_binary': 528},
        participants={'faced': 98, 'seed_binary': 12}, bootstrap_draws=100000, bootstrap_seed=20261116,
        faced_sign_flips=1000000, seed_sign_flips=4096, comparisons_per_task=20, joint_family=40,
        input_hashes=input_hashes, source_sha256=sha(__file__),
        statistics_sha256=sha(HERE / 'complete_curves/ecsi_curve_statistics_20261116.py'),
        output_hashes={str(p.relative_to(args.output)): sha(p) for p in args.output.rglob('*.csv') if components not in p.parents})
    (args.output / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('input_hashes', 'output_hashes')}), flush=True)


if __name__ == '__main__':
    main()
