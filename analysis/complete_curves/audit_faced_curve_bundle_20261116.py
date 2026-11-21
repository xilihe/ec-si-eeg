#!/usr/bin/env python3
"""Independent FACED role/head/centroid reconstruction for one curve condition."""
import argparse
from collections import defaultdict
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import torch

from ecsi_curve_statistics_20261116 import binary_auc,mean_interval


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda:stream.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def readcsv(path):
    with path.open() as f:return list(csv.DictReader(f))


def writecsv(path,rows):
    assert rows
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def subject_roles(sf):
    groups=np.array_split(np.random.default_rng(20260811).permutation(123),5)
    test=sorted(map(int,groups[sf-1]));remaining=sorted(set(range(123))-set(test))
    order=np.random.default_rng(20260811+1000+sf-1).permutation(remaining)
    nval=round(len(remaining)*.15)
    return sorted(map(int,order[nval:])),sorted(map(int,order[:nval])),test


def reconstruct_probe(rep,labels):
    train=rep['train_trial_features'].astype(np.float32);tv=rep['train_video'].astype(int)
    evaluation=rep['probe_trial_features'].astype(np.float32);ev=rep['probe_video'].astype(int)
    subjects=rep['probe_subject'].astype(int)
    center=train.mean(axis=0);scale=train.std(axis=0);scale[scale<1e-8]=1
    train=(train-center)/scale;evaluation=(evaluation-center)/scale
    result=[]
    for category in range(9):
        candidates=np.unique(tv[labels[tv]==category])
        centroids=np.stack([train[tv==video].mean(axis=0) for video in candidates])
        selected=labels[ev]==category
        distance=((evaluation[selected,None,:]-centroids[None,:,:])**2).sum(axis=2)
        predicted=candidates[distance.argmin(axis=1)]
        for person,video,decision in zip(subjects[selected],ev[selected],predicted):
            result.append(dict(subject=int(person),video=int(video),category=category,
                               correct=int(video==decision),prediction=int(decision)))
    return result


def validate_run(directory,labels,split):
    m=json.loads((directory/'run_manifest.json').read_text());assert m['status']=='PASS'
    sf,vf=m['subject_fold'],m['stimulus_fold'];tr,va,te=subject_roles(sf)
    assert sorted(m['train_subject_ids'])==tr and sorted(m['val_subject_ids'])==va and sorted(m['test_subject_ids'])==te
    visible=split[vf]['train_val_video_indices_zero_based'];held=split[vf]['test_video_indices_zero_based']
    assert sorted(m['train_val_video_ids'])==sorted(visible) and sorted(m['test_video_ids'])==sorted(held)
    assert not set(visible)&set(held) and len(visible)==19 and len(held)==9
    assert sorted(labels[held].tolist())==list(range(9))
    assert np.bincount(labels[visible],minlength=9).tolist()==[2,2,2,2,3,2,2,2,2]
    for filename,key in [('predictions.npz','prediction_sha256'),('best_model_state.pt','checkpoint_sha256'),
                         ('training_history.csv','training_history_sha256'),('subject_metrics.csv','subject_metrics_sha256'),
                         ('conditional_video_probe.csv','conditional_video_probe_sha256'),
                         ('train_probe_trial_representations.npz','train_probe_trial_representations_sha256')]:
        assert sha(directory/filename)==m[key],directory/filename
    with np.load(directory/'predictions.npz') as z:
        pred={k:z[k] for k in z.files}
    for name in ('logits','features','trial_logits','trial_features'):
        assert np.isfinite(pred[name]).all(),(directory,name)
    with np.load(directory/'train_probe_trial_representations.npz') as z:
        rep={k:z[k] for k in z.files}
    for prefix,persons,videos in [('train',tr,visible),('probe',te,visible)]:
        actual=list(zip(rep[prefix+'_subject'].tolist(),rep[prefix+'_video'].tolist()))
        assert len(actual)==len(set(actual)) and set(actual)==set(itertools.product(persons,videos))
        values=rep[prefix+'_trial_features'];assert values.shape==(len(actual),64) and np.isfinite(values).all()
    windows=list(zip(pred['subject'].tolist(),pred['trial'].tolist(),pred['second'].tolist()))
    assert len(windows)==len(set(windows)) and set(windows)==set(itertools.product(te,held,range(30)))
    np.testing.assert_array_equal(pred['y'],labels[pred['trial']])
    trial_keys=list(zip(pred['trial_subject'].tolist(),pred['trial_index'].tolist()))
    assert len(trial_keys)==len(set(trial_keys)) and set(trial_keys)==set(itertools.product(te,held))
    np.testing.assert_array_equal(pred['trial_y'],labels[pred['trial_index']])
    np.testing.assert_array_equal(pred['feature_subject'],pred['trial_subject'])
    np.testing.assert_array_equal(pred['feature_trial'],pred['trial_index'])
    for i,(person,video) in enumerate(trial_keys):
        mask=(pred['subject']==person)&(pred['trial']==video)
        np.testing.assert_array_equal(pred['logits'][mask].mean(axis=0),pred['trial_logits'][i])
        np.testing.assert_array_equal(pred['features'][mask].mean(axis=0),pred['trial_features'][i])
    weights=torch.load(directory/'best_model_state.pt',map_location='cpu',weights_only=True)
    assert all(bool(torch.isfinite(t).all()) for t in weights.values())
    w=weights['emotion_head.weight'].numpy();b=weights['emotion_head.bias'].numpy()
    assert w.shape==(9,64) and b.shape==(9,)
    reconstructed=pred['trial_features']@w.T+b
    np.testing.assert_allclose(reconstructed,pred['trial_logits'],rtol=1e-5,atol=3e-6)
    original=pred['trial_logits'].argmax(axis=1);head=reconstructed.argmax(axis=1)
    tolerance=np.max(np.abs(reconstructed-pred['trial_logits']),axis=1)
    top=np.sort(pred['trial_logits'],axis=1)[:,-2:];margin=top[:,1]-top[:,0]
    discrepant=original!=head
    assert np.all(margin[discrepant]<=2*tolerance[discrepant]+1e-7),'Head decisions differ beyond arithmetic tolerance'
    assert abs(np.mean(original==pred['trial_y'])-m['metrics']['accuracy'])<1e-12
    probes=reconstruct_probe(rep,labels);pc=defaultdict(list)
    for row in probes:pc[(row['subject'],row['category'])].append(row['correct'])
    stored=readcsv(directory/'conditional_video_probe.csv')
    assert len(stored)==9*len(te)
    for row in stored:
        assert abs(np.mean(pc[(int(row['subject']),int(row['emotion']))])-float(row['accuracy']))<1e-12
    assert abs(np.mean([np.mean(v) for v in pc.values()])-m['metrics']['conditional_video_probe_accuracy'])<1e-12
    gp=directory.parent.parent/'gradient_records'/Path(m['metrics']['gradient_record']).name
    assert sha(gp)==m['metrics']['gradient_record_sha256']
    gradient=json.loads(gp.read_text())
    policy=json.loads((directory.parent.parent/'curve_policy.json').read_text())
    assert gradient['target_ratio']==policy['encoder_norm_ratio']
    assert gradient['initial_state_sha256']==m['metrics']['initial_state_sha256']
    actual_state=hashlib.sha256(b''.join(t.cpu().numpy().tobytes() for t in weights.values())).hexdigest()
    assert gradient['restored_checkpoint_state_sha256']==actual_state
    ratios=[];cos=[];maxerror=0.
    for step in gradient['steps']:
        assert all(math.isfinite(step[k]) for k in ('task_norm','adversarial_norm','raw_ratio','effective_ratio','cosine'))
        assert abs(step['cosine'])<=1+1e-9
        for field,global_field in [('task_norm','task_norm'),('adversarial_norm','adversarial_norm')]:
            norm=math.sqrt(sum(layer[field]**2 for layer in step['layers']))
            assert math.isclose(norm,step[global_field],abs_tol=1e-12,rel_tol=1e-9)
        if gradient['target_ratio'] is not None:
            expected=gradient['target_ratio']*(2/(1+math.exp(-10*step['progress']))-1)
            error=abs(step['effective_ratio']-expected);maxerror=max(maxerror,error);assert error<=1e-5
        ratios.append(step['effective_ratio']);cos.append(step['cosine'])
    logits=pred['trial_logits'].astype(np.float64);logits-=logits.max(axis=1,keepdims=True)
    probabilities=np.exp(logits);probabilities/=probabilities.sum(axis=1,keepdims=True)
    trials=[];aucs=[]
    for i,(person,video) in enumerate(trial_keys):
        truth=int(pred['trial_y'][i]);trials.append(dict(subject=int(person),video=int(video),category=truth,
            correct=int(original[i]==truth),prediction=int(original[i]),log_loss=float(-np.log(max(probabilities[i,truth],1e-15)))))
    for person in te:
        mask=pred['trial_subject']==person
        aucs.append(dict(subject=person,auc=float(np.mean([binary_auc(pred['trial_y'][mask],probabilities[mask,c],c) for c in range(9)]))))
    result=dict(subject_fold=sf,stimulus_fold=vf,seed=m['seed'],model=m['model'],
        initial_state_sha256=gradient['initial_state_sha256'],gradient_updates=len(ratios),
        gradient_ratio_mean=float(np.mean(ratios)),gradient_cosine_mean=float(np.mean(cos)),
        max_ratio_error=maxerror,head_logit_max_abs_error=float(np.max(tolerance)),
        arithmetic_boundary_head_decisions=int(discrepant.sum()),test_trial_count=len(trials),identity_trial_count=len(probes))
    return trials,probes,aucs,result


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--condition',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--engineering-run',type=Path)
    a=p.parse_args();manifest=json.loads((a.bundle/'manifest.json').read_text())
    assert sha(Path(manifest['mapping_csv']))==manifest['mapping_sha256']
    assert sha(Path(manifest['stimulus_splits_json']))==manifest['stimulus_splits_sha256']
    policy=json.loads((a.bundle/'curve_policy.json').read_text())
    expected_ratio=None if a.condition=='standard' else float(a.condition.removeprefix('rho'))
    assert policy['encoder_norm_ratio']==expected_ratio,'Condition label does not match actual intervention'
    labels=np.array([int(r['emotion_id']) for r in readcsv(Path(manifest['mapping_csv']))])
    splits={row['stimulus_fold']:row for row in json.loads(Path(manifest['stimulus_splits_json']).read_text())['folds']}
    if a.engineering_run:
        _,_,_,audit=validate_run(a.engineering_run,labels,splits)
        a.output.mkdir(parents=True,exist_ok=True)
        (a.output/'engineering_run_audit.json').write_text(json.dumps(dict(status='PASS',scope='engineering audit only',audit=audit),indent=2)+'\n')
        print(json.dumps(dict(status='PASS',scope='engineering only',audit=audit)));return
    assert manifest['status']=='PASS' and manifest['completed_runs']==108
    assert len(manifest['models'])==1;model=manifest['models'][0]
    expected=set(itertools.product((2,3,4,5),range(4,13),(20260812,20260813,20260814)))
    observed=set();trial_groups=defaultdict(list);probe_groups=defaultdict(list);auc_groups=defaultdict(list);audits=[];hashes={}
    partition_tasks=defaultdict(list);partition_probes=defaultdict(list)
    for f in sorted((a.bundle/'runs').glob('*/run_manifest.json')):
        m=json.loads(f.read_text());key=(m['subject_fold'],m['stimulus_fold'],m['seed'])
        assert key in expected and key not in observed and m['model']==model;observed.add(key)
        trials,probes,aucs,audit=validate_run(f.parent,labels,splits);audits.append(audit);hashes[str(f)]=sha(f)
        byperson=defaultdict(list);byclass=defaultdict(list)
        for row in trials:byperson[row['subject']].append(row['correct'])
        for row in probes:byclass[(row['subject'],row['category'])].append(row['correct'])
        for person,values in byperson.items():
            assert len(values)==9
            partition_tasks[(person,key[1])].append(float(np.mean(values)))
            partition_probes[(person,key[1])].append(float(np.mean([np.mean(byclass[(person,c)]) for c in range(9)])))
        for row in trials:trial_groups[(row['subject'],row['video'])].append(row)
        for row in probes:probe_groups[(row['subject'],row['video'])].append(row)
        for row in aucs:auc_groups[row['subject']].append(row['auc'])
    assert observed==expected
    people=sorted({s for s,_ in trial_groups});assert len(people)==98
    partition_rows=[]
    assert set(partition_tasks)==set(partition_probes)==set(itertools.product(people,range(4,13)))
    for person,vf in sorted(partition_tasks):
        assert len(partition_tasks[(person,vf)])==len(partition_probes[(person,vf)])==3
        partition_rows.append(dict(condition=a.condition,model=model,subject=person,stimulus_fold=vf,
            balanced_accuracy=float(np.mean(partition_tasks[(person,vf)])),probe=float(np.mean(partition_probes[(person,vf)]))))
    target=set(itertools.product(people,range(28)))
    assert set(trial_groups)==set(probe_groups)==target
    videos=[];subjects=[];confusions=[]
    for person in people:
        classes=[];classprobes=[];losses=[]
        for category in range(9):
            category_videos=np.flatnonzero(labels==category);correct=[];identity=[];classloss=[];prediction_rates=np.zeros(9)
            for video in category_videos:
                tr=trial_groups[(person,int(video))];pr=probe_groups[(person,int(video))]
                expected_test=3*sum(int(video) in splits[v]['test_video_indices_zero_based'] for v in range(4,13))
                expected_probe=27-expected_test
                assert len(tr)==expected_test and len(pr)==expected_probe
                acc=float(np.mean([row['correct'] for row in tr]));probe=float(np.mean([row['correct'] for row in pr]))
                loss=float(np.mean([row['log_loss'] for row in tr]));correct.append(acc);identity.append(probe);classloss.append(loss)
                prediction_rates+=np.bincount([row['prediction'] for row in tr],minlength=9)/len(tr)/len(category_videos)
                videos.append(dict(condition=a.condition,model=model,subject=person,video=int(video),category=category,
                                   accuracy=acc,probe=probe,log_loss=loss,test_repeats=len(tr),probe_repeats=len(pr)))
            classes.append(float(np.mean(correct)));classprobes.append(float(np.mean(identity)));losses.append(float(np.mean(classloss)))
            for decision,rate in enumerate(prediction_rates):
                confusions.append(dict(condition=a.condition,model=model,subject=person,truth=category,prediction=decision,rate=float(rate)))
        assert len(auc_groups[person])==27
        subjects.append(dict(condition=a.condition,model=model,subject=person,balanced_accuracy=float(np.mean(classes)),
            probe=float(np.mean(classprobes)),log_loss=float(np.mean(losses)),macro_ovr_auc=float(np.mean(auc_groups[person])),
            **{f'recall_{i}':value for i,value in enumerate(classes)}))
    a.output.mkdir(parents=True,exist_ok=False)
    for name,rows in [('subject_scores.csv',subjects),('video_scores.csv',videos),('confusion.csv',confusions),('run_audits.csv',audits),('partition_subject_scores.csv',partition_rows)]:writecsv(a.output/name,rows)
    summary=dict(condition=a.condition,model=model,participants=98)
    for field in ('balanced_accuracy','probe','macro_ovr_auc',*[f'recall_{i}' for i in range(9)]):
        mean,lo,hi=mean_interval([r[field] for r in subjects]);summary.update({field:mean,field+'_ci_low':lo,field+'_ci_high':hi})
    writecsv(a.output/'condition_summary.csv',[summary])
    receipt=dict(status='PASS',condition=a.condition,model=model,fits=108,participants=98,
        source_policy_sha256=sha(a.bundle/'curve_policy.json'),source_policy=json.loads((a.bundle/'curve_policy.json').read_text()),
        input_descriptor_sha256=manifest['input_descriptor_sha256'],configuration_sha256=manifest['configuration_sha256'],run_manifest_hashes=hashes,
        test_trials=sum(r['test_trial_count'] for r in audits),identity_trials=sum(r['identity_trial_count'] for r in audits),
        gradient_updates=sum(r['gradient_updates'] for r in audits),max_ratio_error=max(r['max_ratio_error'] for r in audits),
        arithmetic_boundary_head_decisions=sum(r['arithmetic_boundary_head_decisions'] for r in audits),
        label_and_role_cartesian_products_verified=True,checkpoint_heads_reconstructed=True,independent_centroid_reconstructed=True,
        primary_video_equal_aggregation_verified=True,script_sha256=sha(Path(__file__)))
    receipt['output_hashes']={name:sha(a.output/name) for name in ('subject_scores.csv','condition_summary.csv','run_audits.csv','video_scores.csv','confusion.csv','partition_subject_scores.csv')}
    (a.output/'component_validation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('source_policy','run_manifest_hashes')}))


if __name__=='__main__':main()
