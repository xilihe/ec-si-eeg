#!/usr/bin/env python3
"""Close one complete fixed curve only after independent component audits.

No selection of strengths, participants, models or endpoints is supported.
The forty-comparison sensitivity is emitted only when both full tasks exist.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import itertools
import json
from pathlib import Path

from ecsi_curve_statistics_20261116 import CONDITIONS, complete_comparisons, holm, mean_interval


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def read(path):
    with path.open() as f:return list(csv.DictReader(f))


def write(path,rows):
    assert rows
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def task_components(exp,task):
    if task=='seed_binary':
        old=exp.parent/'ecsi_binary_completion_20261116_v1'
        return [(c,old/('formal_original' if c=='standard' else 'formal_norm') if c in ('standard','rho0.3')
                 else exp/('seed_norm'+{'rho0.1':'01','rho1':'10','rho3':'30'}[c]),
                 exp/'analysis_components'/('seed_'+c)) for c in CONDITIONS]
    result=[]
    for condition in CONDITIONS:
        prefix='standard' if condition=='standard' else 'norm'+{'rho0.1':'01','rho0.3':'03','rho1':'10','rho3':'30'}[condition]
        for model in (('ce','ecsi','global') if condition=='standard' else ('ecsi','global')):
            name='faced_'+prefix+'_'+model
            result.append((condition,exp/name,exp/'analysis_components'/name))
    return result


def validate_blocks(exp,task):
    expected=[b for b in json.loads((exp/'code/blocks.json').read_text())
              if b['task']==('seed' if task=='seed_binary' else 'faced')]
    assert len(expected)==(3 if task=='seed_binary' else 44)
    retained={}
    for block in expected:
        path=exp/'execution/block_certificates'/(block['block']+'.json')
        proof=json.loads(path.read_text())
        assert proof['status']=='PASS' and proof['completed_runs']==block['expected_runs']
        assert proof['block']==block['block'] and proof['task']==block['task']
        for source,expected_sha in proof['artifact_hashes'].items():
            assert sha(Path(source))==expected_sha,source
        retained[str(path)]=sha(path)
    return retained


def aggregate(exp,task,output):
    assert not output.exists(),'Retain prior analysis; select a new versioned output if a repair is needed'
    blocks=validate_blocks(exp,task)
    learning_dir=exp/('seed_code' if task=='seed_binary' else 'faced_code_v3')
    frozen=exp/'execution'/('frozen_seed_learning_sources.json' if task=='seed_binary' else 'frozen_faced_v3_learning_sources.json')
    expected_sources=json.loads(frozen.read_text())
    for name,digest in expected_sources.items():assert sha(learning_dir/name)==digest,name
    initials=defaultdict(set);inputs=set();sources=set();configs=defaultdict(set)
    inventory=set();splits=set();people=set();component_hashes={};rows=[];gradients=[];fit_count=0;input_files=[]
    memory_reference=None;partition_rows=[];video_rows=[]
    if task=='faced':
        replay=exp/'execution/memory_lifecycle_replay_validation.json'
        gate=exp/'execution/memory_lifecycle_resource_gate.json'
        resource=json.loads(gate.read_text());reference=json.loads(replay.read_text())
        assert resource['status']==reference['status']=='PASS'
        assert resource['parity_certificate_sha256']==sha(replay)
        assert resource['oom_events']==0 and resource['all_three_reference_runs_exact'] and resource['two_full_stimulus_folds_complete']
        memory_reference=reference['memory_policy']
        assert sha(exp/'faced_code_v3/run_faced_memory_lifecycle_v1_20261116.py')==memory_reference['launcher_sha256']
        for filename,digest in reference['artifact_hashes'].items():assert sha(Path(filename))==digest
        component_hashes[str(replay)]=sha(replay);component_hashes[str(gate)]=sha(gate)
        freeze=exp/'execution/faced_input_freeze.json'
        frozen_inputs=json.loads(freeze.read_text());assert frozen_inputs['status']=='PASS' and len(frozen_inputs['files'])==248
        for filename,expected in frozen_inputs['files'].items():
            path=Path(filename);assert path.stat().st_size==expected['size'] and sha(path)==expected['sha256'],filename
        input_files=frozen_inputs['files']
    for condition,bundle,directory in task_components(exp,task):
        manifest=json.loads((bundle/'manifest.json').read_text());assert manifest['status']=='PASS' and not manifest['smoke']
        expected_ratio=None if condition=='standard' else float(condition.removeprefix('rho'))
        if task=='seed_binary':
            result=json.loads((directory/'analysis_manifest.json').read_text())
            independent=json.loads((directory/'independent_artifact_validation.json').read_text())
            assert result['status']==independent['status']=='PASS'
            assert independent['analysis_sha256']==sha(directory/'analysis_manifest.json')
            assert independent['fit_grid_verified']==manifest['completed_runs']==(144 if condition=='standard' else 96)
            assert independent['test_trial_predictions_verified']==18*manifest['completed_runs']
            assert independent['independently_reconstructed_identity_trials']==72*manifest['completed_runs']
            assert result['condition']==condition and result['task']==task
            policy=json.loads((bundle/'completion_policy.json').read_text())
            binary=json.loads((bundle/'binary_policy.json').read_text())
            assert policy==result['completion_policy'] and binary==result['binary_policy']
            sources.add(tuple(policy[k] for k in ('wrapper_sha256','core_sha256','module_sha256'))+(binary['adapter_sha256'],))
            assert policy['wrapper_sha256']==expected_sources['run_seed_ecsi_completion_20261116.py']
            assert policy['core_sha256']==expected_sources['run_seed_ecsi_joint_holdout.py']
            assert policy['module_sha256']==expected_sources['ecsi_gradient_norm_control_20261116.py']
            assert binary['adapter_sha256']==expected_sources['run_seed_ecsi_binary_completion_20261116.py']
            inventory.add(manifest['source_inventory_sha256']);splits.add(manifest['split_sha256'])
            actual_inventory=[dict(path=p,bytes=Path(p).stat().st_size,sha256=sha(Path(p))) for p in manifest['source_files']]
            assert object_sha(actual_inventory)==manifest['source_inventory_sha256']
            assert sha(bundle/'splits.json')==manifest['split_sha256']
            input_files=actual_inventory
            for key,digest in result['initial_states'].items():initials[key].add(digest)
            configs['standard' if condition=='standard' else 'norm'].add(manifest['configuration_sha256'])
            assert independent['run_manifest_hashes']==result['run_manifest_hashes']
            for name,digest in independent['output_hashes'].items():assert sha(directory/name)==digest
            gradients+=read(directory/'run_scores.csv')
            component_files=('analysis_manifest.json','independent_artifact_validation.json','subject_scores.csv','condition_summary.csv','run_scores.csv','confusion.csv')
        else:
            result=json.loads((directory/'component_validation.json').read_text());assert result['status']=='PASS'
            memory=bundle/'memory_lifecycle_policy.json'
            assert json.loads(memory.read_text())==memory_reference
            generated=bundle/'memory_safe_main.py.txt'
            assert sha(generated)==memory_reference['memory_safe_main_sha256']
            component_hashes[str(memory)]=sha(memory);component_hashes[str(generated)]=sha(generated)
            assert result['fits']==manifest['completed_runs']==108 and result['condition']==condition
            policy=json.loads((bundle/'curve_policy.json').read_text());assert policy==result['source_policy']
            assert len(policy['source_hashes'])==11
            assert policy['source_hashes']=={name:expected_sources[name] for name in policy['source_hashes']}
            sources.add(object_sha({k:v for k,v in policy.items() if k!='encoder_norm_ratio'}))
            for flag in ('label_and_role_cartesian_products_verified','checkpoint_heads_reconstructed',
                         'independent_centroid_reconstructed','primary_video_equal_aggregation_verified'):assert result[flag]
            audits=read(directory/'run_audits.csv');assert len(audits)==108
            for row in audits:
                key=':'.join(row[k] for k in ('subject_fold','stimulus_fold','seed'))
                initials[key].add(row['initial_state_sha256'])
            gradients+=[dict(condition=condition,**row) for row in audits]
            configs[result['model']].add(manifest['configuration_sha256'])
            for name,digest in result['output_hashes'].items():assert sha(directory/name)==digest
            component_files=('component_validation.json','subject_scores.csv','condition_summary.csv','run_audits.csv','video_scores.csv','confusion.csv')
            component_files+=('partition_subject_scores.csv',)
            partition_rows+=read(directory/'partition_subject_scores.csv')
            video_rows+=read(directory/'video_scores.csv')
        assert policy['encoder_norm_ratio']==expected_ratio
        assert result['input_descriptor_sha256']==manifest['input_descriptor_sha256']
        if 'configuration_sha256' in result:assert result['configuration_sha256']==manifest['configuration_sha256']
        inputs.add(manifest['input_descriptor_sha256']);fit_count+=manifest['completed_runs']
        for source,digest in result['run_manifest_hashes'].items():
            path=Path(source);assert sha(path)==digest,source
            run=json.loads(path.read_text())
            assert run['metrics']['input_descriptor_sha256']==manifest['input_descriptor_sha256']
            assert run['metrics']['configuration_sha256']==manifest['configuration_sha256']
        score_rows=read(directory/'subject_scores.csv')
        assert all(r['condition']==condition for r in score_rows)
        if task=='faced':assert all(r['model']==result['model'] for r in score_rows)
        rows+=score_rows;people.update(int(r['subject']) for r in score_rows)
        for name in component_files:component_hashes[str(directory/name)]=sha(directory/name)
        component_hashes[str(bundle/'manifest.json')]=sha(bundle/'manifest.json')
    assert len(sources)==len(inputs)==1 and all(len(v)==1 for v in configs.values())
    assert fit_count==(528 if task=='seed_binary' else 1188)
    assert len(initials)==(48 if task=='seed_binary' else 108) and all(len(v)==1 for v in initials.values())
    if task=='seed_binary':assert len(inventory)==len(splits)==1
    participants=12 if task=='seed_binary' else 98
    assert len(people)==participants and len(rows)==11*participants
    comparisons=complete_comparisons(rows,task,participants)
    summaries=[]
    metrics=('balanced_accuracy','probe','nonneutral_auc','negative_recall','positive_recall') if task=='seed_binary' else (
        'balanced_accuracy','probe','macro_ovr_auc',*[f'recall_{i}' for i in range(9)])
    grouped=defaultdict(list)
    for row in rows:grouped[(row['condition'],row['model'])].append(row)
    for (condition,model),group in grouped.items():
        result=dict(task=task,condition=condition,model=model,participants=participants)
        for metric in metrics:
            value,lo,hi=mean_interval([float(r[metric]) for r in group])
            result.update({metric:value,metric+'_ci_low':lo,metric+'_ci_high':hi})
        summaries.append(result)
    output.mkdir(parents=True)
    output_names=['subject_scores.csv','condition_summary.csv','comparisons_twenty.csv','gradient_runs.csv']
    for name,data in [('subject_scores.csv',rows),('condition_summary.csv',summaries),('comparisons_twenty.csv',comparisons),('gradient_runs.csv',gradients)]:write(output/name,data)
    if task=='faced':
        from ecsi_curve_stability_20261117 import describe
        stability=describe(partition_rows,video_rows,rows)
        for name,data in stability.items():
            name=name if name.endswith('.csv') else name+'.csv'
            write(output/name,data);output_names.append(name)
        write(output/'stimulus_partition_subject_scores.csv',partition_rows);output_names.append('stimulus_partition_subject_scores.csv')
    receipt=dict(status='PASS',task=task,independent_participants=participants,fits=fit_count,
        curve_condition_models=11,comparisons=20,holm_family_size=20,bootstrap_draws=100000,
        sign_configurations_or_draws=4096 if task=='seed_binary' else 1000000,
        full_fixed_matrix=True,matched_learning_code=True,matched_inputs=True,matched_initial_states=True,
        same_model_configuration_across_strengths=True,block_certificate_hashes=blocks,
        resource_lifetime_repair_verified=(task=='faced'),
        input_files=input_files,component_hashes=component_hashes,script_sha256=sha(Path(__file__)),
        statistics_sha256=sha(Path(__file__).with_name('ecsi_curve_statistics_20261116.py')),
        output_hashes={str(output/name):sha(output/name) for name in output_names},
        interpretation='Positive difference means higher classification accuracy or lower identity Probe. No noninferiority margin was specified. Scheduled gradient norm is matched; cumulative exposure, direction and AdamW updates are not equated.')
    (output/'validation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('component_hashes','input_files','block_certificate_hashes','output_hashes')}))


def combine(exp,output):
    assert not output.exists()
    rows=[];inputs={}
    for task in ('faced','seed_binary'):
        directory=exp/('analysis_'+task)
        proof=json.loads((directory/'validation.json').read_text());assert proof['status']=='PASS'
        for name,digest in proof['output_hashes'].items():assert sha(Path(name))==digest
        rows+=read(directory/'comparisons_twenty.csv');inputs[str(directory/'validation.json')]=sha(directory/'validation.json')
    assert len(rows)==40
    for row,pvalue in zip(rows,holm([float(r['raw_p']) for r in rows])):row['holm_forty_p']=pvalue
    output.mkdir(parents=True);write(output/'comparisons_forty.csv',rows)
    (output/'validation.json').write_text(json.dumps(dict(status='PASS',sensitivity_family_size=40,
        input_hashes=inputs,output_sha256=sha(output/'comparisons_forty.csv'),script_sha256=sha(Path(__file__))),indent=2)+'\n')
    print(json.dumps(dict(status='PASS',comparisons=40,output=str(output))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--task',choices=('faced','seed_binary','combined'),required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    (combine(a.experiment,a.output) if a.task=='combined' else aggregate(a.experiment,a.task,a.output))
