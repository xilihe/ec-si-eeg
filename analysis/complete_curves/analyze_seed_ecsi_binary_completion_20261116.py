#!/usr/bin/env python3
"""Audit and summarize new same-code SEED fits, with full class-level readouts."""
import argparse
from collections import defaultdict
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def writecsv(path, rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def binary_auc(labels, scores):
    positive=np.asarray(scores)[np.asarray(labels)==1]
    negative=np.asarray(scores)[np.asarray(labels)==0]
    assert len(positive)>0 and len(negative)>0
    differences=positive[:,None]-negative[None,:]
    return float(np.mean((differences>0)+.5*(differences==0)))


def mean_ci(values):
    values=np.asarray(values,dtype=float)
    rng=np.random.default_rng(20261116)
    means=values[rng.integers(0,len(values),(100000,len(values)))].mean(axis=1)
    return [float(x) for x in np.quantile(means,[.025,.975])]


def summarize(bundle, label):
    policy=json.loads((bundle/'completion_policy.json').read_text())
    binary=json.loads((bundle/'binary_policy.json').read_text())
    assert binary['classes']==2 and binary['neutral_windows_excluded_before_fitting']
    manifest=json.loads((bundle/'manifest.json').read_text())
    assert manifest['status']=='PASS'
    runs=sorted((bundle/'runs').glob('*/run_manifest.json'))
    assert len(runs)==manifest['expected_runs']==manifest['completed_runs']
    observations=defaultdict(list); probe=defaultdict(list); confusion=defaultdict(list); auc=defaultdict(list)
    initial=defaultdict(set); runrows=[]
    checks=dict(runs=0,predictions=0,gradient_updates=0,max_ratio_error=0.,verified_hashes=0)
    input_hashes={}
    for path in runs:
        m=json.loads(path.read_text()); assert m['status']=='PASS'
        d=path.parent; model=m['model']; sf=m['subject_fold']; vf=m['stimulus_fold']; seed=m['seed']
        assert not set(m['train_subjects'])&set(m['test_subjects'])
        assert not set(m['val_subjects'])&set(m['test_subjects'])
        assert not set(m['train_subjects'])&set(m['val_subjects'])
        assert not set(m['train_stimuli_zero_based'])&set(m['test_stimuli_zero_based'])
        for name,key in [('training_history.csv','training_history_sha256'),('subject_metrics.csv','subject_metrics_sha256'),
                         ('conditional_stimulus_probe.csv','conditional_stimulus_probe_sha256'),
                         ('trial_representations.npz','trial_representations_sha256'),('best_model_state.pt','checkpoint_sha256')]:
            assert sha(d/name)==m[key],(path,name)
            checks['verified_hashes']+=1
        initial[(sf,vf,seed)].add(m['metrics']['initial_state_sha256'])
        with np.load(d/'trial_representations.npz') as a:
            y=a['test_y']; pred=a['base_test_trial_logits'].argmax(1)
            subjects=a['test_subject']; sessions=a['test_session']; stimuli=a['test_stimulus']
            assert set(subjects)==set(m['test_subjects'])
            assert set(stimuli)==set(m['test_stimuli_zero_based'])
            assert set(a['train_subject'])==set(m['train_subjects'])
            assert set(a['probe_subject'])==set(m['test_subjects'])
            assert set(a['probe_stimulus'])==set(m['train_stimuli_zero_based'])
            assert np.isfinite(a['base_test_trial_logits']).all()
            assert abs(float(np.mean(pred==y))-m['metrics']['accuracy'])<1e-7
            assert a['base_test_trial_logits'].shape[1]==2 and set(y)=={0,1}
            score=a['base_test_trial_logits'][:,1]-a['base_test_trial_logits'][:,0]
            for s in np.unique(subjects):
                mask=subjects==s
                auc[(model,int(s))].append(binary_auc(y[mask],score[mask]))
            for s,se,st,c,p in zip(subjects,sessions,stimuli,y,pred):
                observations[(model,int(s),int(se),int(st),int(c))].append(float(p==c))
                confusion[(model,int(c),int(p))].append((int(s),int(se),int(st),seed))
            checks['predictions']+=len(y)
        # Equal conditional classes within subject/run, then repeats within subject.
        with (d/'conditional_stimulus_probe.csv').open() as f:
            pr=list(csv.DictReader(f))
        bysubject=defaultdict(list)
        for r in pr: bysubject[int(r['subject'])].append(float(r['accuracy']))
        for s,values in bysubject.items(): probe[(model,s)].append(float(np.mean(values)))
        gp=bundle/'gradient_records'/Path(m['metrics']['gradient_record']).name
        assert sha(gp)==m['metrics']['gradient_record_sha256']
        gradient=json.loads(gp.read_text())
        assert gradient['initial_state_sha256']==m['metrics']['initial_state_sha256']
        ratios=[]; cosines=[]
        for g in gradient['steps']:
            assert all(np.isfinite(g[k]) for k in ('task_norm','adversarial_norm','raw_ratio','effective_ratio','cosine'))
            if gradient['target_ratio'] is not None and model!='ecsi_ce':
                error=abs(g['effective_ratio']-gradient['target_ratio']*g['schedule'])
                checks['max_ratio_error']=max(checks['max_ratio_error'],error)
                assert error<=1e-5
            ratios.append(g['effective_ratio']); cosines.append(g['cosine'])
        checks['gradient_updates']+=len(ratios); checks['runs']+=1
        runrows.append(dict(condition=label,model=model,subject_fold=sf,stimulus_fold=vf,seed=seed,
            accuracy=m['metrics']['accuracy'],probe=m['metrics']['conditional_stimulus_probe_accuracy'],
            gradient_ratio_mean=float(np.mean(ratios)),gradient_cosine_mean=float(np.mean(cosines)),
            best_epoch=gradient['best_epoch']))
        input_hashes[str(path)]=sha(path)
    assert all(len(v)==1 for v in initial.values()),'unequal initial states'
    subjects=[]; summaries=[]; matrices=[]
    models=sorted({k[0] for k in observations})
    for model in models:
        people=sorted({k[1] for k in observations if k[0]==model})
        for s in people:
            recalls=[]
            for c in range(2):
                vals=[float(np.mean(v)) for (mo,su,_,_,cl),v in observations.items() if mo==model and su==s and cl==c]
                assert vals
                recalls.append(float(np.mean(vals)))
            subjects.append(dict(condition=label,model=model,subject=s,balanced_accuracy=float(np.mean(recalls)),
                negative_recall=recalls[0],positive_recall=recalls[1],
                nonneutral_macro_recall=(recalls[0]+recalls[1])/2,probe=float(np.mean(probe[(model,s)])),
                nonneutral_auc=float(np.mean(auc[(model,s)]))))
        ss=[r for r in subjects if r['model']==model]
        summaries.append(dict(condition=label,model=model,subjects=len(people),**{k:float(np.mean([r[k] for r in ss])) for k in
            ('balanced_accuracy','negative_recall','positive_recall','nonneutral_macro_recall','probe','nonneutral_auc')}))
        for key in ('balanced_accuracy','negative_recall','positive_recall','nonneutral_macro_recall','nonneutral_auc'):
            low,high=mean_ci([r[key] for r in ss])
            summaries[-1][key+'_ci_low']=low
            summaries[-1][key+'_ci_high']=high
        # All fits contain equal sessions/stimuli per class, so these rows weight
        # subjects equally. Repeated seed counts are fixed and validated above.
        for c in range(2):
            total=sum(len(confusion[(model,c,p)]) for p in range(2))
            for p in range(2): matrices.append(dict(condition=label,model=model,truth=c,prediction=p,rate=len(confusion[(model,c,p)])/total))
    return subjects,summaries,runrows,matrices,checks,input_hashes


def paired_stats(d):
    d=np.asarray(d,dtype=float); n=len(d)
    signs=np.array(list(itertools.product((-1.,1.),repeat=n)))
    raw=float(np.mean(np.abs(signs@d)>=abs(d.sum())-1e-12))
    rng=np.random.default_rng(20261116)
    means=np.mean(d[rng.integers(0,n,(100000,n))],axis=1)
    lo,hi=np.quantile(means,[.025,.975])
    return dict(mean_difference=float(d.mean()),ci_low=float(lo),ci_high=float(hi),exact_two_sided_p=raw)


def holm(rows,key,out):
    order=sorted(range(len(rows)),key=lambda i:rows[i][key]); previous=0.
    for rank,i in enumerate(order):
        previous=max(previous,min(1.,(len(rows)-rank)*rows[i][key])); rows[i][out]=previous


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',action='append',required=True,help='label=path')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--formal',action='store_true')
    p.add_argument('--analysis-addendum',type=Path);a=p.parse_args()
    if a.formal: assert a.analysis_addendum is not None and a.analysis_addendum.is_file()
    a.output.mkdir(parents=True,exist_ok=False)
    subjects=[];summaries=[];runrows=[];matrices=[];checks={};hashes={};common_code=set();common_input=set();initial=defaultdict(set)
    for item in a.bundle:
        label,path=item.split('=',1)
        bundle=Path(path)
        po=json.loads((bundle/'completion_policy.json').read_text())
        bp=json.loads((bundle/'binary_policy.json').read_text())
        ma=json.loads((bundle/'manifest.json').read_text())
        common_code.add(tuple(po[k] for k in ('wrapper_sha256','core_sha256','module_sha256'))+(bp['adapter_sha256'],))
        common_input.add(ma['input_descriptor_sha256'])
        for f in (bundle/'runs').glob('*/run_manifest.json'):
            m=json.loads(f.read_text());initial[(m['subject_fold'],m['stimulus_fold'],m['seed'])].add(m['metrics']['initial_state_sha256'])
        sr,su,rr,cm,ch,ha=summarize(Path(path),label)
        subjects+=sr;summaries+=su;runrows+=rr;matrices+=cm;checks[label]=ch;hashes.update(ha)
    assert len(common_code)==1 and len(common_input)==1 and all(len(v)==1 for v in initial.values())
    for name,rows in [('subject_scores.csv',subjects),('condition_summary.csv',summaries),('run_scores.csv',runrows),('confusion.csv',matrices)]:
        writecsv(a.output/name,rows)
    gate=None
    if not a.formal:
        required=[r for r in summaries if r['condition']=='original' and r['model'] in ('ecsi_ce','ecsi_adv')]
        assert len(required)==2
        gate=all(r['balanced_accuracy']>=.60 and min(r['negative_recall'],r['positive_recall'])>=.55 for r in required)
    comparisons=[]
    if a.formal:
        for condition in sorted({r['condition'] for r in summaries}):
            comparators=['ecsi_ce','ecsi_global_adv'] if condition=='original' else ['ecsi_global_adv']
            for comparator in comparators:
                for endpoint in ('balanced_accuracy','probe'):
                    left={r['subject']:r[endpoint] for r in subjects if r['condition']==condition and r['model']=='ecsi_adv'}
                    right={r['subject']:r[endpoint] for r in subjects if r['condition']==condition and r['model']==comparator}
                    assert set(left)==set(right) and len(left)==12
                    d=[(left[s]-right[s])*(1 if endpoint=='balanced_accuracy' else -1) for s in sorted(left)]
                    comparisons.append(dict(condition=condition,comparator=comparator,endpoint=endpoint,subjects=12,**paired_stats(d)))
        holm(comparisons,'exact_two_sided_p','holm_all_six_p')
        for condition in sorted({r['condition'] for r in comparisons}):
            family=[r for r in comparisons if r['condition']==condition]
            holm(family,'exact_two_sided_p','holm_within_policy_p')
        writecsv(a.output/'paired_comparisons.csv',comparisons)
    output=dict(status='PASS',task_gate=None if gate is None else ('PASS' if gate else 'FAIL'),checks=checks,
        summaries=summaries,input_manifest_hashes=hashes,script_sha256=sha(Path(__file__)),
        independently_sourced_stimuli=False,independent_cohort_confirmation=False,
        common_code_and_inputs_verified=True,cross_policy_initialization_verified=True,
        analysis_addendum_sha256=sha(a.analysis_addendum) if a.analysis_addendum else None,
        task='non_neutral_binary_from_training',number_of_actual_classifier_outputs=2)
    (a.output/'analysis_manifest.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(dict(task_gate=output['task_gate'],summaries=summaries,checks=checks,comparisons=comparisons),indent=2))


if __name__=='__main__': main()
