#!/usr/bin/env python3
"""Frozen SEED supplement: original training plus measured gradient controls."""
import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import functional as F

import run_seed_ecsi_joint_holdout as core
from ecsi_gradient_norm_control_20261116 import measured_backward

RATIO = None
ACTIVE_ARGS = None


def state_hash(model):
    return hashlib.sha256(b"".join(t.detach().cpu().numpy().tobytes() for t in model.state_dict().values())).hexdigest()


def train_model(model, variant, adv_lambda, guard_mode, train, val, local_by_stimulus,
                global_by_stimulus, args, device, seed):
    if variant not in ("ecsi_ce", "ecsi_adv", "ecsi_global_adv") or guard_mode != "none":
        raise ValueError("completion runner supports only original CE/EC-SI/Global")
    core.set_seed(seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(train["de"]), torch.from_numpy(train["y"]),
        torch.from_numpy(local_by_stimulus[train["stimulus"]]),
        torch.from_numpy(global_by_stimulus[train["stimulus"]])), batch_size=args.batch_size,
        shuffle=True, generator=torch.Generator().manual_seed(seed))
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    initial_hash = state_hash(model)
    steps = min(len(loader), args.max_train_batches) if args.max_train_batches else len(loader)
    total_steps = args.epochs*steps
    best_score, best_loss, best_epoch, stale = -1., float("inf"), 0, 0
    best = copy.deepcopy(model.state_dict())
    history, diagnostics = [], []
    for epoch in range(1, args.epochs+1):
        model.train(); values=[]
        for index, (de,y,local,global_id) in enumerate(loader, 1):
            de,y,local,global_id=[x.to(device) for x in (de,y,local,global_id)]
            opt.zero_grad(set_to_none=True)
            z=model.encode(de)
            task=F.cross_entropy(model.emotion_head(z),y,reduction="none").mean()
            ramp=2/(1+math.exp(-10*((epoch-1)*steps+index)/total_steps))-1
            strength=adv_lambda*ramp
            aux=z.sum()*0
            if variant=="ecsi_adv": aux=model.conditional_loss(z,y,local,strength)
            if variant=="ecsi_global_adv": aux=model.global_loss(z,global_id,strength)
            if not bool(torch.isfinite(task+aux)):
                raise FloatingPointError("nonfinite objective")
            row=measured_backward(task,aux,model.encoder.named_parameters(),
                target_ratio=RATIO*ramp if RATIO is not None and variant!="ecsi_ce" else None)
            opt.step()
            row.update(epoch=epoch,batch=index,schedule=ramp,grl_strength=strength)
            diagnostics.append(row)
            values.append((float(task),float(aux),row["raw_ratio"],row["effective_ratio"]))
            if index>=steps: break
        score,loss=core.validation_score(model,val,device,args.eval_batch_size)
        means=np.mean(values,axis=0)
        history.append(dict(epoch=epoch,train_emotion_loss=float(means[0]),train_identity_loss=float(means[1]),
            raw_adversarial_task_ratio=float(means[2]),effective_adversarial_task_ratio=float(means[3]),
            val_trial_balanced_accuracy=score,val_window_log_loss=loss))
        if score>best_score+1e-8 or (abs(score-best_score)<=1e-8 and loss<best_loss):
            best_score,best_loss,best_epoch,stale=score,loss,epoch,0
            best=copy.deepcopy(model.state_dict())
        else: stale+=1
        if stale>=args.patience: break
    model.load_state_dict(best)
    # The core writes per-run manifests; this separate immutable record identifies
    # the wrapper/policy and retains all update-level gradient measurements.
    record_dir=Path(args.bundle_dir)/"gradient_records"
    record_dir.mkdir(exist_ok=True)
    train_ids=sorted(set(train["subject"].tolist()))
    stimuli=sorted(set(train["stimulus"].tolist()))
    key=hashlib.sha256(json.dumps([train_ids,stimuli,seed,variant],sort_keys=True).encode()).hexdigest()[:20]
    path=record_dir/f"{key}.json"
    if path.exists(): raise FileExistsError(path)
    core.write_json_atomic(path,dict(variant=variant,seed=seed,train_subjects=train_ids,train_stimuli=stimuli,
        target_ratio=RATIO,initial_state_sha256=initial_hash,best_epoch=best_epoch,
        wrapper_sha256=core.sha256(Path(__file__)),
        gradient_module_sha256=core.sha256(Path(__file__).with_name("ecsi_gradient_norm_control_20261116.py")),
        steps=diagnostics))
    return model,dict(best_epoch=best_epoch,best_val_trial_balanced_accuracy=best_score,
        best_val_window_log_loss=best_loss,total_trainable_params=sum(p.numel() for p in model.parameters()),
        inference_params=model.inference_parameter_count(),initial_state_sha256=initial_hash,
        gradient_policy="original" if RATIO is None else "encoder_norm_control",
        target_ratio=-1.0 if RATIO is None else RATIO,gradient_record_sha256=core.sha256(path),
        gradient_record=str(path),history=history)


def main():
    global RATIO, ACTIVE_ARGS
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument("--encoder-norm-ratio",type=float)
    p.add_argument("--completion-protocol",type=Path,required=True)
    extra, remaining=p.parse_known_args()
    RATIO=extra.encoder_norm_ratio
    if RATIO is not None and (not math.isfinite(RATIO) or RATIO<=0):
        raise ValueError("positive finite encoder-norm-ratio required")
    sys.argv=[sys.argv[0],*remaining]
    base_parse=core.parse_args
    def parse():
        args=base_parse()
        if not args.export_representations: raise ValueError("trial export is mandatory")
        args.bundle_dir.mkdir(parents=True,exist_ok=True)
        receipt=args.bundle_dir/"completion_policy.json"
        policy=dict(wrapper_sha256=core.sha256(Path(__file__)),protocol_sha256=core.sha256(extra.completion_protocol),
                    encoder_norm_ratio=RATIO,core_sha256=core.sha256(Path(core.__file__)),
                    module_sha256=core.sha256(Path(__file__).with_name("ecsi_gradient_norm_control_20261116.py")))
        if receipt.exists() and json.loads(receipt.read_text())!=policy:
            raise ValueError("existing bundle has a different completion policy")
        core.write_json_atomic(receipt,policy)
        return args
    core.parse_args=parse
    core.train_model=train_model
    torch.set_num_threads(2)
    core.main()


if __name__=="__main__": main()
