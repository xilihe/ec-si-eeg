"""Complete fixed-partition and fixed-prediction influence descriptions; no tests."""
from collections import defaultdict
import itertools

import numpy as np

CONDITIONS=('standard','rho0.1','rho0.3','rho1','rho3')


def describe(partition_rows,video_rows,primary_rows,expected_participants=98):
    groups={('standard',m) for m in ('ecsi_ce','ecsi_adv','ecsi_global_adv')}
    groups|={(c,m) for c in CONDITIONS[1:] for m in ('ecsi_adv','ecsi_global_adv')}
    people=sorted({int(r['subject']) for r in primary_rows});assert len(people)==expected_participants
    primary={(r['condition'],r['model'],int(r['subject'])):r for r in primary_rows}
    assert len(primary)==len(primary_rows)==11*expected_participants
    partitions={(r['condition'],r['model'],int(r['subject']),int(r['stimulus_fold'])):r for r in partition_rows}
    assert len(partitions)==len(partition_rows)==11*expected_participants*9
    assert set(partitions)=={(*g,s,v) for g in groups for s in people for v in range(4,13)}
    videos={(r['condition'],r['model'],int(r['subject']),int(r['video'])):r for r in video_rows}
    assert len(videos)==len(video_rows)==11*expected_participants*28
    assert set(videos)=={(*g,s,v) for g in groups for s in people for v in range(28)}
    categories={}
    for r in video_rows:
        v=int(r['video']);c=int(r['category'])
        if v in categories:assert categories[v]==c
        categories[v]=c
    assert sorted(categories)==list(range(28))
    assert np.bincount(list(categories.values()),minlength=9).tolist()==[3,3,3,3,4,3,3,3,3]
    labels={c:[v for v in range(28) if categories[v]==c] for c in range(9)}
    partition_summary=[];influence=[]
    for condition,model in sorted(groups):
        for vf in range(4,13):
            chosen=[partitions[(condition,model,s,vf)] for s in people]
            partition_summary.append(dict(condition=condition,model=model,stimulus_fold=vf,participants=len(people),
                balanced_accuracy=float(np.mean([float(r['balanced_accuracy']) for r in chosen])),
                probe=float(np.mean([float(r['probe']) for r in chosen]))))
        for person in people:
            for endpoint,field in [('balanced_accuracy','accuracy'),('probe','probe')]:
                point=np.mean([np.mean([float(videos[(condition,model,person,v)][field]) for v in labels[c]]) for c in range(9)])
                assert abs(point-float(primary[(condition,model,person)][endpoint]))<1e-12
        for omitted in range(28):
            estimates=defaultdict(list)
            for person in people:
                for endpoint,field in [('balanced_accuracy','accuracy'),('probe','probe')]:
                    classes=[np.mean([float(videos[(condition,model,person,v)][field]) for v in labels[c] if v!=omitted]) for c in range(9)]
                    estimates[endpoint].append(float(np.mean(classes)))
            influence.append(dict(condition=condition,model=model,omitted_video=omitted,participants=len(people),
                                  **{endpoint:float(np.mean(values)) for endpoint,values in estimates.items()}))
    assert len(partition_summary)==99 and len(influence)==308
    outputs={};ranges=[]
    for scope,rows,column,units in [('stimulus_partition',partition_summary,'stimulus_fold',range(4,13)),
                                     ('single_video_influence',influence,'omitted_video',range(28))]:
        lookup={(r['condition'],r['model'],r[column]):r for r in rows};contrasts=[]
        for condition in CONDITIONS:
            for comparator in ('ecsi_ce','ecsi_global_adv'):
                rc='standard' if comparator=='ecsi_ce' else condition
                for endpoint in ('balanced_accuracy','probe'):
                    differences=[]
                    for unit in units:
                        left=lookup[(condition,'ecsi_adv',unit)][endpoint];right=lookup[(rc,comparator,unit)][endpoint]
                        effect=(left-right)*(1 if endpoint=='balanced_accuracy' else -1);differences.append(effect)
                        contrasts.append(dict(scope=scope,condition=condition,comparator=comparator,comparator_condition=rc,
                            endpoint=endpoint,unit=unit,mean_difference=effect,
                            positive_direction='EC-SI minus reference' if endpoint=='balanced_accuracy' else 'reference minus EC-SI'))
                    ranges.append(dict(scope=scope,condition=condition,comparator=comparator,comparator_condition=rc,
                        endpoint=endpoint,units=len(differences),minimum=min(differences),maximum=max(differences),
                        positive_count=sum(d>1e-12 for d in differences),negative_count=sum(d< -1e-12 for d in differences),
                        zero_count=sum(abs(d)<=1e-12 for d in differences),inference='descriptive only; no p value or independent-source claim'))
        assert len(contrasts)==20*len(units)
        outputs[scope+'_contrasts.csv']=contrasts
    outputs.update(stimulus_partition_summary=partition_summary,single_video_influence_summary=influence,
                   stability_ranges=ranges)
    return outputs
