#!/usr/bin/env python3
"""Train a real two-output non-neutral SEED classifier; never mask test outputs."""
import json
import math
from pathlib import Path

import numpy as np
import run_seed_ecsi_completion_20261116 as completion

core=completion.core
ORIGINAL_LOAD=core.load_corpus


def binary_stimulus_splits(labels, seed):
    assert len(labels)==15 and sorted(np.bincount(np.where(labels<0,1,np.where(labels==1,2,0))).tolist())==[5,5,5]
    original=np.where(labels<0,1,np.where(labels==1,2,0))
    rng=np.random.default_rng(seed)
    byclass={c:rng.permutation(np.flatnonzero(original==c)).tolist() for c in range(3)}
    included=set(np.flatnonzero(labels>=0).tolist())
    result={}
    for fold in range(1,6):
        test=sorted([int(byclass[0][fold-1]),int(byclass[2][fold-1])])
        train=sorted(included-set(test))
        assert len(train)==8 and len(test)==2 and sorted(labels[test].tolist())==[0,1]
        result[fold]=dict(train=train,test=test)
    assert sorted(v for r in result.values() for v in r['test'])==sorted(included)
    return result


def binary_training_ids(labels, train):
    if len(train)!=8 or any(labels[v]<0 for v in train):raise ValueError('binary training must contain only eight non-neutral clips')
    local=np.full(15,-1,dtype=np.int64);global_id=np.full(15,-1,dtype=np.int64)
    for c in range(2):
        videos=sorted(v for v in train if labels[v]==c)
        assert len(videos)==4
        for i,v in enumerate(videos):local[v]=i;global_id[v]=4*c+i
    assert sorted(global_id[global_id>=0].tolist())==list(range(8))
    return local,global_id


def load_binary(data_dir,de_sidecar_dir=None):
    saved=core.N_CLASSES
    core.N_CLASSES=3
    try: original=ORIGINAL_LOAD(data_dir,de_sidecar_dir)
    finally: core.N_CLASSES=saved
    keep=original.y!=1
    mapping=np.where(original.labels_by_stimulus==1,-1,np.where(original.labels_by_stimulus==2,1,0))
    return core.Corpus(de=original.de[keep],y=np.where(original.y[keep]==2,1,0).astype(np.int64),
        stimulus=original.stimulus[keep],subject=original.subject[keep],session=original.session[keep],
        window=original.window[keep],labels_by_stimulus=mapping,source_files=original.source_files)


def install():
    core.N_CLASSES=2
    core.GLOBAL_LOSS_SCALE=float(math.log(4)/math.log(8))
    core.load_corpus=load_binary
    core.make_stimulus_splits=binary_stimulus_splits
    core.training_stimulus_ids=binary_training_ids


def main():
    install()
    parse=core.parse_args
    def binary_parse():
        args=parse()
        args.bundle_dir.mkdir(parents=True,exist_ok=True)
        path=args.bundle_dir/'binary_policy.json'
        policy=dict(task='non_neutral_binary_from_training',classes=2,neutral_windows_excluded_before_fitting=True,
            adapter_sha256=core.sha256(Path(__file__)),global_loss_scale=core.GLOBAL_LOSS_SCALE,
            training_stimuli=8,test_stimuli=2,outputs_are_not_masked=True)
        if path.exists() and json.loads(path.read_text())!=policy:raise ValueError('binary policy differs from existing bundle')
        core.write_json_atomic(path,policy)
        return args
    core.parse_args=binary_parse
    completion.main()


if __name__=='__main__':main()
