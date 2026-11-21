#!/usr/bin/env python3
"""Portable entry point for the retained complete-curve training sources."""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', choices=['faced', 'seed_binary'], required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True, help='Experiments parent directory')
    p.add_argument('--condition', choices=['standard', '0.1', '0.3', '1', '3'], required=True)
    p.add_argument('--device', choices=['cpu', 'cuda', 'mps', 'auto'], default='auto')
    p.add_argument('--subject-folds', default='2,3,4,5')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    root = Path(__file__).resolve().parent
    data, output = a.data_root.resolve(), a.output.resolve()
    ratio = [] if a.condition == 'standard' else ['--encoder-norm-ratio', a.condition]
    exp = output / 'ecsi_final_gap_closure_20261116_v2'
    if a.task == 'faced':
        code = root / 'training/faced'
        prefix = {'standard': 'standard', '0.1': 'norm01', '0.3': 'norm03', '1': 'norm10', '3': 'norm30'}[a.condition]
        commands = []
        for model, suffix in [('ecsi_ce', 'ce'), ('ecsi_adv', 'ecsi'), ('ecsi_global_adv', 'global')]:
            if suffix == 'ce' and ratio:
                continue
            commands.append([sys.executable, str(code / 'run_faced_memory_lifecycle_v1_20261116.py'),
                '--curve-protocol', str(code / '20261116_ecsi_final_gap_closure_v2.md'), *ratio,
                '--data-root', str(data), '--mapping-csv', str(root / 'configs/20260811_faced_stimulus_mapping_v1.csv'),
                '--stimulus-splits-json', str(root / 'configs/20260812_faced_joint_subject_stimulus_splits_v1.json'),
                '--bundle-dir', str(exp / f'faced_{prefix}_{suffix}'), '--models', model,
                '--subject-folds', a.subject_folds, '--stimulus-folds', '4,5,6,7,8,9,10,11,12',
                '--seeds', '20260812,20260813,20260814', '--adv-lambda', '1', '--adv-schedule', 'dann', '--device', a.device])
    else:
        code = root / 'training/seed'
        if a.condition in ('standard', '0.3'):
            bundle = output / 'ecsi_binary_completion_20261116_v1' / ('formal_original' if a.condition == 'standard' else 'formal_norm')
        else:
            bundle = exp / ('seed_norm' + {'0.1': '01', '1': '10', '3': '30'}[a.condition])
        commands = [[sys.executable, str(code / 'run_seed_ecsi_binary_completion_20261116.py'),
            '--completion-protocol', str(code / '20261116_ecsi_final_gap_closure_v2.md'), *ratio,
            '--data-dir', str(data), '--bundle-dir', str(bundle),
            '--models', 'ecsi_ce,ecsi_adv,ecsi_global_adv' if not ratio else 'ecsi_adv,ecsi_global_adv',
            '--subject-folds', a.subject_folds, '--stimulus-folds', '2,3,4,5',
            '--seeds', '20260817,20260818,20260819', '--adv-lambda', '1', '--device', a.device, '--export-representations']]
    for command in commands:
        print(shlex.join(command), flush=True)
        if not a.dry_run:
            subprocess.run(command, cwd=code, check=True)


if __name__ == '__main__':
    main()
