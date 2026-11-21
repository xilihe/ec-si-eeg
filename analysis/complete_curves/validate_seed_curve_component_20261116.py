#!/usr/bin/env python3
"""Independent scope and numerical reconstruction for one complete SEED curve bundle."""
import argparse
from collections import defaultdict
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch


def read(path):
    with path.open() as f:return list(csv.DictReader(f))


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--analysis',type=Path,required=True);p.add_argument('--condition',required=True);a=p.parse_args()
    exp=a.bundle.parent;analysis=a.analysis
    final=json.loads((analysis/'analysis_manifest.json').read_text());assert final['status']=='PASS'
    class_correct=defaultdict(list);probe_scores=defaultdict(list);auc_pairs=defaultdict(list)
    class_confusion=defaultdict(int);initial=defaultdict(set);trials=0;fits=0;probe_trials=0
    fingerprints={}
    models=tuple(json.loads((a.bundle/'manifest.json').read_text())['models'])
    expected_fits=48*len(models)
    for condition,folder,models in [(a.condition,a.bundle.name,models)]:
        bundle=exp/folder
        manifest=json.loads((bundle/'manifest.json').read_text());assert manifest['status']=='PASS'
        bp=json.loads((bundle/'binary_policy.json').read_text());assert bp['classes']==2 and bp['neutral_windows_excluded_before_fitting']
        splits=json.loads((bundle/'splits.json').read_text());assert sha(bundle/'splits.json')==manifest['split_sha256']
        mapping=np.asarray(splits['labels_by_stimulus_zero_based'])
        keys=set();people=set()
        for mp in sorted((bundle/'runs').glob('*/run_manifest.json')):
            m=json.loads(mp.read_text());assert m['status']=='PASS'
            sf,vf,seed,model=m['subject_fold'],m['stimulus_fold'],m['seed'],m['model']
            key=(sf,vf,seed,model);assert key not in keys;keys.add(key)
            subject=splits['subject_splits'][str(sf)];stimulus=splits['stimulus_splits'][str(vf)]
            assert set(m['train_subjects'])==set(subject['train'])
            assert set(m['val_subjects'])==set(subject['val'])
            assert set(m['test_subjects'])==set(subject['test'])
            assert set(m['train_stimuli_zero_based'])==set(stimulus['train'])
            assert set(m['test_stimuli_zero_based'])==set(stimulus['test'])
            assert not set(subject['train'])&set(subject['test'])
            assert not set(subject['val'])&set(subject['test'])
            assert not set(subject['train'])&set(subject['val'])
            assert not set(stimulus['train'])&set(stimulus['test'])
            assert len(stimulus['train'])==8 and len(stimulus['test'])==2
            assert all(mapping[v]>=0 for v in stimulus['train']+stimulus['test'])
            directory=mp.parent
            for name,keyhash in [('trial_representations.npz','trial_representations_sha256'),('best_model_state.pt','checkpoint_sha256')]:
                assert sha(directory/name)==m[keyhash]
            state=torch.load(directory/'best_model_state.pt',map_location='cpu',weights_only=True)
            assert all(bool(torch.isfinite(t).all()) for t in state.values())
            gp=bundle/'gradient_records'/Path(m['metrics']['gradient_record']).name
            assert sha(gp)==m['metrics']['gradient_record_sha256']
            gradient=json.loads(gp.read_text())
            policy=json.loads((bundle/'completion_policy.json').read_text())
            assert gradient['target_ratio']==policy['encoder_norm_ratio']
            assert gradient['initial_state_sha256']==m['metrics']['initial_state_sha256']
            with np.load(directory/'trial_representations.npz') as ar:
                arrays={k:ar[k] for k in ar.files}
            for prefix,persons,clips in [('train',subject['train'],stimulus['train']),
                                          ('probe',subject['test'],stimulus['train']),
                                          ('test',subject['test'],stimulus['test'])]:
                so,se,st=(arrays[prefix+'_'+k] for k in ('subject','session','stimulus'))
                expected=set(itertools.product(persons,(1,2,3),clips))
                observed=set(zip(so.tolist(),se.tolist(),st.tolist()))
                assert len(so)==len(observed)==len(expected) and observed==expected,(mp,prefix)
                np.testing.assert_array_equal(arrays[prefix+'_y'],mapping[st])
                features=arrays[prefix+'_trial_features']
                assert features.shape==(len(so),64) and np.isfinite(features).all()
            logits=arrays['base_test_trial_logits'];y=arrays['test_y'];sids=arrays['test_subject']
            head=arrays['test_trial_features']@state['emotion_head.weight'].numpy().T+state['emotion_head.bias'].numpy()
            np.testing.assert_allclose(logits,head,rtol=2e-5,atol=2e-5)
            pred=logits.argmax(1)
            assert len(y)==18 and np.bincount(y,minlength=2).tolist()==[9,9]
            for s,c,prediction in zip(sids,y,pred):
                class_correct[(condition,model,int(s),int(c))].append(float(c==prediction))
                class_confusion[(condition,model,int(c),int(prediction))]+=1
            assert logits.shape[1]==2
            score=logits[:,1]-logits[:,0]
            for s in subject['test']:
                pos=score[(sids==s)&(y==1)];neg=score[(sids==s)&(y==0)]
                auc_pairs[(condition,model,s)].extend(float(v>w)+.5*float(v==w) for v in pos for w in neg)
            # Independent reconstruction of the fresh conditional centroid reader.
            z=arrays['train_trial_features'];mu=z.mean(0);sd=z.std(0);sd[sd<1e-8]=1.
            train_z=(z-mu)/sd;eval_z=(arrays['probe_trial_features']-mu)/sd
            csv_probe={(int(r['subject']),int(r['emotion'])):r for r in read(directory/'conditional_stimulus_probe.csv')}
            assert set(csv_probe)==set(itertools.product(subject['test'],range(2)))
            for c in range(2):
                videos=sorted(v for v in stimulus['train'] if mapping[v]==c);assert len(videos)==4
                centroids=np.stack([train_z[arrays['train_stimulus']==v].mean(0) for v in videos])
                for s in subject['test']:
                    mask=(arrays['probe_subject']==s)&(arrays['probe_y']==c)
                    distances=((eval_z[mask,None,:]-centroids[None,:,:])**2).sum(2)
                    recovered=np.asarray(videos)[distances.argmin(1)]
                    observed=float(np.mean(recovered==arrays['probe_stimulus'][mask]))
                    row=csv_probe[(s,c)]
                    assert int(mask.sum())==int(row['trials'])==12 and float(row['chance'])==.25
                    assert abs(observed-float(row['accuracy']))<1e-12
                    probe_scores[(condition,model,s)].append(observed)
                    probe_trials+=int(mask.sum())
            initial[(sf,vf,seed)].add(m['metrics']['initial_state_sha256'])
            people.update(subject['test']);trials+=len(y);fits+=1;fingerprints[str(mp)]=sha(mp)
        expected=set(itertools.product(range(2,6),range(2,6),(20260817,20260818,20260819),models))
        assert keys==expected and len(people)==12
        assert not people&set(splits['subject_splits']['1']['test'])
    assert fits==expected_fits and trials==18*expected_fits and probe_trials==72*expected_fits
    assert all(len(values)==1 for values in initial.values())
    for r in read(analysis/'subject_scores.csv'):
        c,m,s=r['condition'],r['model'],int(r['subject'])
        rates=[]
        for emotion,name in enumerate(('negative_recall','positive_recall')):
            observations=class_correct[(c,m,s,emotion)];assert len(observations)==36
            estimate=sum(observations)/len(observations);rates.append(estimate)
            assert abs(estimate-float(r[name]))<1e-12
        assert abs(sum(rates)/2-float(r['balanced_accuracy']))<1e-12
        assert abs(sum(rates)/2-float(r['nonneutral_macro_recall']))<1e-12
        assert len(auc_pairs[(c,m,s)])==108
        assert abs(np.mean(auc_pairs[(c,m,s)])-float(r['nonneutral_auc']))<1e-12
        assert len(probe_scores[(c,m,s)])==24
        assert abs(np.mean(probe_scores[(c,m,s)])-float(r['probe']))<1e-12
    for r in read(analysis/'confusion.csv'):
        c,m,t,pred=r['condition'],r['model'],int(r['truth']),int(r['prediction'])
        denominator=sum(class_confusion[(c,m,t,p)] for p in range(2))
        assert abs(class_confusion[(c,m,t,pred)]/denominator-float(r['rate']))<1e-12
    result=dict(status='PASS',fit_grid_verified=fits,test_trial_predictions_verified=trials,
        independently_reconstructed_identity_trials=probe_trials,distinct_evaluation_subjects=12,
        complete_subject_session_stimulus_cartesian_products=True,labels_match_split_mapping=True,
        predictions_match_saved_linear_heads=True,probe_reconstructed_from_frozen_representations=True,
        primary_class_recall_and_auc_aggregation_verified=True,cross_policy_initial_states_match=True,
        validator_sha256=sha(Path(__file__)),analysis_sha256=sha(analysis/'analysis_manifest.json'),
        run_manifest_hashes=fingerprints,task='non_neutral_binary_from_training',actual_classifier_outputs=2,
        saved_checkpoint_file_hashes_verified=True,
        output_hashes={name:sha(analysis/name) for name in ('subject_scores.csv','condition_summary.csv','run_scores.csv','confusion.csv')})
    (analysis/'independent_artifact_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='run_manifest_hashes'},indent=2))


if __name__=='__main__':main()
