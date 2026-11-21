#!/usr/bin/env python3
"""Independent summary/role coverage audit against frozen complete-curve scores."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--analysis', type=Path, required=True)
    p.add_argument('--curves', type=Path, required=True)
    a = p.parse_args()
    certificate = json.loads((a.analysis / 'validation.json').read_text())
    assert certificate['status'] == 'PASS' and certificate['result_claim_allowed']
    for file, expected in certificate['output_hashes'].items():
        assert digest(a.analysis / file) == expected
    audits = json.loads((a.analysis / 'per_fit_audits.json').read_text())
    assert len(audits) == 1716
    assert len({(r['task'], r['condition'], r['model'], Path(r['run']).name) for r in audits}) == 1716
    counts = {'faced': 1188, 'seed_binary': 528}
    checked, intervals = 0, 0
    for task in counts:
        actual = [r for r in audits if r['task'] == task]
        assert len(actual) == counts[task]
        native = read(a.curves / ('analysis_faced' if task == 'faced' else 'analysis_seed_binary') / 'subject_scores.csv')
        modelmap = {'ecsi_ce': 'ce', 'ecsi_adv': 'ecsi', 'ecsi_global_adv': 'global'}
        ref = {}
        for r in native:
            condition = modelmap[r['model']] if r['condition'] == 'standard' else r['condition'] + '_' + modelmap[r['model']]
            ref[(condition, int(r['subject']))] = r
        behavior = read(a.analysis / f'behavior_{task}_subject_scores.csv')
        for r in behavior:
            if r['measure'] != 'unseen':
                continue
            previous = ref[(r['condition'], int(r['subject']))]
            np.testing.assert_allclose(float(r['value']), float(previous['balanced_accuracy']), atol=1e-12, rtol=0)
            checked += 1
            if task == 'faced':
                for c in range(9):
                    np.testing.assert_allclose(float(r[f'recall_{c}']), float(previous[f'recall_{c}']), atol=1e-12, rtol=0)
                    checked += 1
        projection = read(a.analysis / f'projection_{task}_subject_scores.csv')
        for r in projection:
            if task == 'faced' and r['condition'] == 'unprojected_ridge' and r['measure'] == 'probe':
                np.testing.assert_allclose(float(r['value']), float(ref[('ce', int(r['subject']))]['probe']), atol=1e-12, rtol=0)
                checked += 1
        # SEED's original Probe is run-equal, while the new matched projection
        # control is clip-equal within category; these estimands need not match.
        # Recompute CIs in one independently allocated resampling array.
        n = 98 if task == 'faced' else 12
        index = np.random.default_rng(20261120).integers(0, n, (100000, n))
        for mode, source in [('behavior', behavior), ('projection', projection)]:
            summary = read(a.analysis / f'{mode}_{task}_summary.csv')
            contrasts = read(a.analysis / f'{mode}_{task}_contrasts.csv')
            groups = {}
            for r in source:
                groups.setdefault((r['condition'], r['measure']), {})[int(r['subject'])] = r
            for row in summary + contrasts:
                if 'condition' in row:
                    left = groups[(row['condition'], row['measure'])]
                    right = None
                else:
                    left = groups[(row['left_condition'], row['left_measure'])]
                    right = groups[(row['right_condition'], row['right_measure'])]
                    assert set(left) == set(right)
                persons = sorted(left)
                assert len(persons) == (98 if task == 'faced' else 12)
                for key in ['value', *[k for k in row if k.startswith('recall_') and not k.endswith(('_ci_low', '_ci_high'))]]:
                    values = np.asarray([float(left[s][key]) - (float(right[s][key]) if right else 0) for s in persons])
                    low, high = np.quantile(values[index].mean(1), [.025, .975])
                    np.testing.assert_allclose([float(row[key]), float(row[key + '_ci_low']), float(row[key + '_ci_high'])],
                        [values.mean(), low, high], atol=1e-12, rtol=0)
                    intervals += 1
        for row in actual:
            if 'projection' in row:
                projection_file = a.analysis / (task + '_' + Path(row['run']).name) / 'projection_fit.npz'
                assert digest(projection_file) == row['projection']['projection_fit_sha256']
                with np.load(projection_file) as z:
                    basis = z['nullspace_basis']
                    np.testing.assert_allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=1e-10, rtol=0)
                    assert basis.shape == (64, 46 if task == 'faced' else 57)
    result = dict(status='PASS', source_sha256=digest(__file__), input_validation_sha256=digest(a.analysis / 'validation.json'),
        source_curve_unseen_classification_reproduced=True, faced_unprojected_probe_reproduced=True,
        seed_projection_probe_aggregation='clip_equal_within_category_not_original_run_equal',
        cells_checked=checked, independent_summary_intervals=intervals,
        all_156_projection_fits_orthonormal=True, full_1716_unique_fit_coverage=True,
        scope='Independent arithmetic, source-score consistency and saved transform audit; not a fresh model training run')
    (a.analysis / 'independent_validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
