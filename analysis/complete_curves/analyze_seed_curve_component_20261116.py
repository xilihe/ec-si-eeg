#!/usr/bin/env python3
"""Summarize and independently validate one frozen SEED binary curve point."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import analyze_seed_ecsi_binary_completion_20261116 as summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--condition',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    manifest=json.loads((a.bundle/'manifest.json').read_text());assert manifest['status']=='PASS'
    policy=json.loads((a.bundle/'completion_policy.json').read_text())
    binary=json.loads((a.bundle/'binary_policy.json').read_text())
    ratio=policy['encoder_norm_ratio'];expected=None if a.condition=='standard' else float(a.condition.removeprefix('rho'))
    assert ratio==expected
    models=set(manifest['models'])
    assert models==({'ecsi_ce','ecsi_adv','ecsi_global_adv'} if ratio is None else {'ecsi_adv','ecsi_global_adv'})
    assert manifest['completed_runs']==48*len(models)
    a.output.mkdir(parents=True,exist_ok=False)
    subjects,summaries,runs,confusion,checks,hashes=summary.summarize(a.bundle,a.condition)
    for name,rows in [('subject_scores.csv',subjects),('condition_summary.csv',summaries),('run_scores.csv',runs),('confusion.csv',confusion)]:
        summary.writecsv(a.output/name,rows)
    initials={}
    for f in sorted((a.bundle/'runs').glob('*/run_manifest.json')):
        m=json.loads(f.read_text());key=f"{m['subject_fold']}:{m['stimulus_fold']}:{m['seed']}"
        fingerprint=m['metrics']['initial_state_sha256']
        if key in initials:assert initials[key]==fingerprint
        initials[key]=fingerprint
    assert len(initials)==48
    result=dict(status='PASS',condition=a.condition,task='seed_binary',checks=checks,
        completion_policy=policy,binary_policy=binary,input_descriptor_sha256=manifest['input_descriptor_sha256'],
        source_inventory_sha256=manifest['source_inventory_sha256'],split_sha256=manifest['split_sha256'],
        configuration_sha256=manifest['configuration_sha256'],
        run_manifest_hashes=hashes,initial_states=initials,
        primary_aggregation='participant, session, clip and seed rules retained from the frozen binary task',
        script_sha256=summary.sha(Path(__file__)),summarizer_sha256=summary.sha(Path(summary.__file__)))
    (a.output/'analysis_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    subprocess.run([sys.executable,str(Path(__file__).with_name('validate_seed_curve_component_20261116.py')),
                    '--bundle',str(a.bundle),'--analysis',str(a.output),'--condition',a.condition],check=True)
    print(json.dumps(dict(status='PASS',condition=a.condition,fits=checks['runs'],predictions=checks['predictions'])))


if __name__=='__main__':main()
