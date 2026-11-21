#!/usr/bin/env python3
"""Post-hoc frozen-representation controls; no model training or test fitting.

Requires numpy and torch (checkpoint loading only). See the fixed versioned plan.
No raw EEG or participant-level output should be redistributed by this script.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import itertools
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def writecsv(path, rows):
    assert rows
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def ridge(x, y):
    """Mean-loss ridge, fixed alpha=1; intercept unpenalized."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xm, ym = x.mean(0), y.mean(0)
    xc, yc = x - xm, y - ym
    w = np.linalg.solve(xc.T @ xc / len(x) + np.eye(x.shape[1]), xc.T @ yc / len(x))
    return w, ym - xm @ w


def fit_projection(train, stimulus):
    """Class-agnostic: this function never receives emotion labels."""
    train = np.asarray(train, dtype=np.float64)
    mu, scale = train.mean(0), train.std(0)
    scale[scale < 1e-8] = 1
    x = (train - mu) / scale
    ids, index = np.unique(stimulus, return_inverse=True)
    w, intercept = ridge(x, np.eye(len(ids))[index])
    _, singular, vt = np.linalg.svd(w.T, full_matrices=True)
    rank = int(np.sum(singular > singular[0] * 1e-10)) if singular[0] else 0
    basis = vt[rank:].T
    p = basis @ basis.T
    assert basis.shape[0] == 64 and basis.shape[1] > 0
    np.testing.assert_allclose(p @ p, p, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(w.T @ basis, 0, atol=1e-9, rtol=0)
    return mu, scale, basis, dict(removed_rank=rank, retained_dimensions=int(basis.shape[1]),
        projector_idempotence_error=float(np.max(np.abs(p @ p - p))),
        removed_decoder_residual=float(np.max(np.abs(w.T @ basis))))


def readnpz(path, keys=None):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in (z.files if keys is None else keys)}


def verify_cartesian(subject, stimulus, persons, clips, session=None):
    if session is None:
        actual = list(zip(subject.tolist(), stimulus.tolist()))
        expected = set(itertools.product(persons, clips))
    else:
        actual = list(zip(subject.tolist(), stimulus.tolist(), session.tolist()))
        expected = set(itertools.product(persons, clips, sorted(set(session.tolist()))))
    assert len(actual) == len(set(actual)) and set(actual) == expected


def load_run(task, path, labels):
    m = json.loads(path.read_text())
    assert m['status'] == 'PASS' and m['subject_disjoint'] and m['stimulus_disjoint']
    d = path.parent
    if task == 'faced':
        names = [('train_probe_trial_representations.npz', 'train_probe_trial_representations_sha256'),
                 ('predictions.npz', 'prediction_sha256'), ('best_model_state.pt', 'checkpoint_sha256')]
        trainpeople, valpeople, testpeople = [m[k + '_subject_ids'] for k in ('train', 'val', 'test')]
        seen, unseen = m['train_val_video_ids'], m['test_video_ids']
    else:
        names = [('trial_representations.npz', 'trial_representations_sha256'), ('best_model_state.pt', 'checkpoint_sha256')]
        trainpeople, valpeople, testpeople = [m[k + '_subjects'] for k in ('train', 'val', 'test')]
        seen, unseen = m['train_stimuli_zero_based'], m['test_stimuli_zero_based']
    assert not set(trainpeople) & set(valpeople)
    assert not (set(trainpeople) | set(valpeople)) & set(testpeople)
    assert not set(seen) & set(unseen)
    hashes = {name: sha(d / name) for name, _ in names}
    for name, key in names:
        assert hashes[name] == m[key], (d, name)
    if task == 'faced':
        rep = readnpz(d / names[0][0])
        pred = readnpz(d / 'predictions.npz', ['trial_features', 'trial_y', 'trial_logits', 'trial_subject', 'trial_index'])
        arrays = {}
        for prefix in ('train', 'probe'):
            arrays[prefix] = dict(x=rep[prefix + '_trial_features'], subject=rep[prefix + '_subject'],
                stimulus=rep[prefix + '_video'], session=np.zeros(len(rep[prefix + '_subject']), dtype=int))
            arrays[prefix]['y'] = labels[arrays[prefix]['stimulus']]
        arrays['test'] = dict(x=pred['trial_features'], subject=pred['trial_subject'], stimulus=pred['trial_index'],
            session=np.zeros(len(pred['trial_y']), dtype=int), y=pred['trial_y'])
        logits = pred['trial_logits']
    else:
        rep = readnpz(d / names[0][0])
        arrays = {prefix: dict(x=rep[prefix + '_trial_features'], y=rep[prefix + '_y'],
            subject=rep[prefix + '_subject'], stimulus=rep[prefix + '_stimulus'], session=rep[prefix + '_session'])
            for prefix in ('train', 'probe', 'test')}
        logits = rep['base_test_trial_logits']
    for prefix, persons, clips in [('train', trainpeople, seen), ('probe', testpeople, seen), ('test', testpeople, unseen)]:
        a = arrays[prefix]
        assert a['x'].shape == (len(a['y']), 64) and np.isfinite(a['x']).all()
        verify_cartesian(a['subject'], a['stimulus'], persons, clips, a['session'] if task != 'faced' else None)
        np.testing.assert_array_equal(a['y'], labels[a['stimulus']])
    weights = torch.load(d / 'best_model_state.pt', map_location='cpu', weights_only=True)
    w, b = weights['emotion_head.weight'].numpy(), weights['emotion_head.bias'].numpy()
    assert w.shape == (9 if task == 'faced' else 2, 64)
    reconstructed = arrays['test']['x'] @ w.T + b
    np.testing.assert_allclose(reconstructed, logits, rtol=1e-5, atol=3e-6)
    margin = np.diff(np.sort(logits, axis=1)[:, -2:], axis=1)[:, 0]
    err = np.max(np.abs(reconstructed - logits), axis=1)
    changed = reconstructed.argmax(1) != logits.argmax(1)
    assert np.all(margin[changed] <= 2 * err[changed] + 1e-7)
    assert abs(float(np.mean(logits.argmax(1) == arrays['test']['y'])) - m['metrics']['accuracy']) < 1e-7
    audit = dict(manifest_sha256=sha(path), input_hashes=hashes,
        checkpoint_head_max_error=float(err.max()), arithmetic_boundary_decisions=int(changed.sum()),
        training_rows=len(arrays['train']['y']), probe_rows=len(arrays['probe']['y']), test_rows=len(arrays['test']['y']))
    return m, arrays, arrays['probe']['x'] @ w.T + b, logits, audit


def add_outcomes(store, key, a, decision):
    for s, clip, session, category, value in zip(a['subject'], a['stimulus'], a['session'], a['y'], decision == a['y']):
        store[(*key, int(s), int(clip), int(session), int(category))].append(float(value))


def probe_decision(train, train_ids, train_y, evaluation, eval_y, classes):
    decision = np.empty(len(evaluation), dtype=int)
    for category in range(classes):
        ids = np.unique(train_ids[train_y == category])
        assert len(ids) >= 2
        centroids = np.stack([train[train_ids == clip].mean(0) for clip in ids])
        selected = eval_y == category
        distance = ((evaluation[selected, None] - centroids[None]) ** 2).sum(2)
        decision[selected] = ids[distance.argmin(1)]
    return decision


def projected_outcomes(arrays, store, task, fitdir):
    train = arrays['train']
    mu, scale, basis, audit = fit_projection(train['x'], train['stimulus'])
    classes = 9 if task == 'faced' else 2
    standardized = {role: (a['x'].astype(np.float64) - mu) / scale for role, a in arrays.items()}
    for name, values in [('unprojected_ridge', standardized),
                         ('projected_ridge', {r: x @ basis for r, x in standardized.items()})]:
        w, intercept = ridge(values['train'], np.eye(classes)[train['y']])
        pred = (values['test'] @ w + intercept).argmax(1)
        add_outcomes(store, (task, name, 'accuracy'), arrays['test'], pred)
        ids = probe_decision(values['train'], train['stimulus'], train['y'], values['probe'], arrays['probe']['y'], classes)
        a = arrays['probe']
        for s, clip, session, category, value in zip(a['subject'], a['stimulus'], a['session'], a['y'], ids == a['stimulus']):
            store[(task, name, 'probe', int(s), int(clip), int(session), int(category))].append(float(value))
    fitdir.mkdir()
    np.savez_compressed(fitdir / 'projection_fit.npz', center=mu, scale=scale, nullspace_basis=basis)
    audit['projection_fit_sha256'] = sha(fitdir / 'projection_fit.npz')
    audit['fit_uses_emotion_labels'] = False
    audit['fit_subject_role'] = 'train'
    audit['fit_stimulus_role'] = 'train_visible'
    return audit


def aggregate(store):
    # Every key contains prefix=(task, model/condition, status/metric), then
    # subject, clip, session and category. Equal repeats, sessions, clips, classes.
    clips = defaultdict(list)
    for key, values in store.items():
        prefix, s, clip, session, category = key[:-4], *key[-4:]
        clips[(*prefix, s, clip, category)].append(float(np.mean(values)))
    classes = defaultdict(list)
    for key, sessions in clips.items():
        prefix, s, clip, category = key[:-3], *key[-3:]
        classes[(*prefix, s, category)].append(float(np.mean(sessions)))
    people = defaultdict(dict)
    for key, values in classes.items():
        prefix, s, category = key[:-2], *key[-2:]
        people[(*prefix, s)][category] = float(np.mean(values))
    rows = []
    for key, values in sorted(people.items()):
        task, condition, measure, subject = key
        assert len(values) == (9 if task == 'faced' else 2)
        rows.append(dict(task=task, condition=condition, measure=measure, subject=subject,
            value=float(np.mean(list(values.values()))), **{f'recall_{c}': v for c, v in sorted(values.items())}))
    return rows


def interval(values):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(20261120)
    # Chunked to bound memory; preserves the fixed generator sequence.
    samples = np.concatenate([values[rng.integers(0, len(values), (10000, len(values)))].mean(1) for _ in range(10)])
    lo, hi = np.quantile(samples, [.025, .975])
    return float(values.mean()), float(lo), float(hi)


def summaries(rows, mode):
    by = defaultdict(list)
    for row in rows:
        by[(row['task'], row['condition'], row['measure'])].append(row)
    result, contrasts = [], []
    for (task, condition, measure), values in sorted(by.items()):
        people = {r['subject']: r for r in values}
        assert len(people) == (98 if task == 'faced' else 12)
        row = dict(task=task, condition=condition, measure=measure, participants=len(people))
        for k in [k for k in values[0] if k == 'value' or k.startswith('recall_')]:
            mean, lo, hi = interval([people[s][k] for s in sorted(people)])
            row.update({k: mean, k + '_ci_low': lo, k + '_ci_high': hi})
        result.append(row)
    tasks = sorted({r['task'] for r in rows})
    for task in tasks:
        conditions = sorted({r['condition'] for r in rows if r['task'] == task})
        pairs = [(c, 'seen', c, 'unseen') for c in conditions] if mode == 'behavior' else [
            ('projected_ridge', metric, 'unprojected_ridge', metric) for metric in ('accuracy', 'probe')]
        for c1, m1, c0, m0 in pairs:
            left = {r['subject']: r for r in by[(task, c1, m1)]}
            right = {r['subject']: r for r in by[(task, c0, m0)]}
            assert set(left) == set(right)
            row = dict(task=task, left_condition=c1, left_measure=m1, right_condition=c0,
                right_measure=m0, participants=len(left))
            for k in [k for k in next(iter(left.values())) if k == 'value' or k.startswith('recall_')]:
                mean, lo, hi = interval([left[s][k] - right[s][k] for s in sorted(left)])
                row.update({k: mean, k + '_ci_low': lo, k + '_ci_high': hi})
            contrasts.append(row)
    return result, contrasts


def bundles(root):
    exp = root / 'ecsi_final_gap_closure_20261116_v2'
    for condition, prefix in [('standard', 'standard'), ('rho0.1', 'norm01'), ('rho0.3', 'norm03'), ('rho1', 'norm10'), ('rho3', 'norm30')]:
        for model in (('ce', 'ecsi', 'global') if condition == 'standard' else ('ecsi', 'global')):
            yield 'faced', condition, model, exp / f'faced_{prefix}_{model}'
    old = root / 'ecsi_binary_completion_20261116_v1'
    for condition, path in [('standard', old / 'formal_original'), ('rho0.3', old / 'formal_norm'),
                            ('rho0.1', exp / 'seed_norm01'), ('rho1', exp / 'seed_norm10'), ('rho3', exp / 'seed_norm30')]:
        yield 'seed_binary', condition, None, path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--experiments', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--plan', required=True, type=Path)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    started = time.time()
    behavior, projection = defaultdict(list), defaultdict(list)
    audits, counts, pc = [], defaultdict(int), defaultdict(int)
    for task, condition, model, bundle in bundles(args.experiments):
        manifest = json.loads((bundle / 'manifest.json').read_text())
        assert manifest['status'] == 'PASS'
        if task == 'faced':
            mapping = Path(manifest['mapping_csv'])
            assert sha(mapping) == manifest['mapping_sha256']
            with mapping.open() as f:
                labels = np.array([int(r['emotion_id']) for r in csv.DictReader(f)])
        else:
            policy = json.loads((bundle / 'binary_policy.json').read_text())
            assert policy['classes'] == 2 and policy['neutral_windows_excluded_before_fitting']
            labels = np.array(json.loads((bundle / 'splits.json').read_text())['labels_by_stimulus_zero_based'])
        paths = sorted((bundle / 'runs').glob('*/run_manifest.json'))
        assert len(paths) == manifest['completed_runs']
        if args.smoke:
            if condition != 'standard' or (task == 'faced' and model != 'ce'):
                continue
            paths = [f for f in paths if json.loads(f.read_text())['model'] == 'ecsi_ce'][:1]
        for path in paths:
            m, arrays, seen, unseen, audit = load_run(task, path, labels)
            actual_model = {'ecsi_ce': 'ce', 'ecsi_adv': 'ecsi', 'ecsi_global_adv': 'global'}[m['model']]
            if model is not None:
                assert actual_model == model, m['model']
            key = actual_model if condition == 'standard' else condition + '_' + actual_model
            add_outcomes(behavior, (task, key, 'seen'), arrays['probe'], seen.argmax(1))
            add_outcomes(behavior, (task, key, 'unseen'), arrays['test'], unseen.argmax(1))
            counts[task] += 1
            if actual_model == 'ce':
                fitdir = args.output / (task + '_' + path.parent.name)
                audit['projection'] = projected_outcomes(arrays, projection, task, fitdir)
                pc[task] += 1
            audits.append(dict(task=task, condition=condition, model=actual_model, run=str(path.parent), **audit))
            if len(audits) % 25 == 0 or args.smoke:
                print(json.dumps(dict(processed=len(audits), counts=dict(counts), projection=dict(pc), elapsed=time.time() - started)), flush=True)
    outputs = []
    if not args.smoke:
        assert dict(counts) == {'faced': 1188, 'seed_binary': 528}, dict(counts)
        assert dict(pc) == {'faced': 108, 'seed_binary': 48}, dict(pc)
        for name, data in [('behavior', behavior), ('projection', projection)]:
            rows = aggregate(data)
            # Separate task files because FACED and SEED have different class counts.
            for task in ('faced', 'seed_binary'):
                selected = [r for r in rows if r['task'] == task]
                summary, contrasts = summaries(selected, name)
                for suffix, values in [('subject_scores', selected), ('summary', summary), ('contrasts', contrasts)]:
                    filename = name + '_' + task + '_' + suffix + '.csv'
                    writecsv(args.output / filename, values)
                    outputs.append(filename)
    auditfile = args.output / 'per_fit_audits.json'
    auditfile.write_text(json.dumps(audits, indent=2) + '\n')
    outputs.append(auditfile.name)
    certificate = dict(status='ENGINEERING_SMOKE_PASS' if args.smoke else 'PASS', result_claim_allowed=not args.smoke,
        counts=dict(counts), projection_counts=dict(pc), plan_sha256=sha(args.plan), source_sha256=sha(__file__),
        source=str(Path(__file__).resolve()), python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
        elapsed_seconds=time.time() - started, started_at=started, completed_at=time.time(),
        independent_units={'faced': 98, 'seed_binary': 12}, bootstrap_replicates=100000, bootstrap_seed=20261120,
        analysis_type='post_hoc_descriptive', new_confirmatory_p_values=False,
        output_hashes={name: sha(args.output / name) for name in outputs})
    (args.output / 'validation.json').write_text(json.dumps(certificate, indent=2) + '\n')
    print(json.dumps(certificate), flush=True)


if __name__ == '__main__':
    main()
