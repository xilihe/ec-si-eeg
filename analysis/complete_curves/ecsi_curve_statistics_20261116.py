"""Fixed complete-curve comparisons; participants remain the paired units."""
import itertools

import numpy as np

CONDITIONS=('standard','rho0.1','rho0.3','rho1','rho3')
MODELS=('ecsi_ce','ecsi_adv','ecsi_global_adv')


def holm(pvalues):
    order=np.argsort(pvalues,kind='stable');out=np.empty(len(pvalues));last=0.
    for rank,index in enumerate(order):
        last=max(last,min(1.,float(pvalues[index])*(len(order)-rank)));out[index]=last
    return out.tolist()


def paired_matrix_statistics(differences,exact,bootstrap_runs=100000,permutations=1000000,seed=20261116):
    d=np.asarray(differences,dtype=np.float64)
    assert d.ndim==2 and len(d)>=2 and np.isfinite(d).all()
    n,k=d.shape;observed=np.abs(d.mean(axis=0));rng=np.random.default_rng(seed)
    boots=[]
    for start in range(0,bootstrap_runs,1000):
        indices=rng.integers(0,n,size=(min(1000,bootstrap_runs-start),n))
        boots.append(d[indices].mean(axis=1))
    low,high=np.quantile(np.concatenate(boots),[.025,.975],axis=0)
    if exact:
        assert n<=16
        signs=np.asarray(list(itertools.product((-1.,1.),repeat=n)))
        raw=(np.abs(signs@d/n)>=observed-1e-12).mean(axis=0)
        test_count=len(signs)
    else:
        rng=np.random.default_rng(seed);extreme=np.zeros(k,dtype=np.int64)
        for start in range(0,permutations,2000):
            signs=rng.integers(0,2,size=(min(2000,permutations-start),n),dtype=np.int8)*2-1
            extreme+=(np.abs(signs@d/n)>=observed-1e-12).sum(axis=0)
        raw=(extreme+1)/(permutations+1);test_count=permutations
    return [dict(mean_difference=float(d[:,i].mean()),ci_low=float(low[i]),ci_high=float(high[i]),
                 raw_p=float(raw[i]),paired_standard_deviation=float(d[:,i].std(ddof=1)),
                 bootstrap_draws=bootstrap_runs,sign_configurations_or_draws=test_count,
                 exact_test=bool(exact)) for i in range(k)]


def complete_comparisons(rows,task,expected_participants,bootstrap_runs=100000,permutations=1000000):
    assert task in ('faced','seed_binary')
    lookup={};groups={}
    for row in rows:
        key=(row['condition'],row['model'],int(row['subject']))
        assert key not in lookup;lookup[key]=row
        groups.setdefault(key[:2],set()).add(key[2])
    expected={('standard',m) for m in MODELS}|{(c,m) for c in CONDITIONS[1:] for m in MODELS[1:]}
    assert set(groups)==expected
    people=sorted(groups[('standard','ecsi_ce')]);assert len(people)==expected_participants
    assert all(g==set(people) for g in groups.values())
    metadata=[];vectors=[]
    for condition in CONDITIONS:
        for reference in ('ecsi_ce','ecsi_global_adv'):
            reference_condition='standard' if reference=='ecsi_ce' else condition
            for endpoint in ('balanced_accuracy','probe'):
                direction=1 if endpoint=='balanced_accuracy' else -1
                vector=[direction*(float(lookup[(condition,'ecsi_adv',s)][endpoint])-
                                   float(lookup[(reference_condition,reference,s)][endpoint])) for s in people]
                vectors.append(vector)
                metadata.append(dict(task=task,condition=condition,comparator=reference,
                    comparator_condition=reference_condition,endpoint=endpoint,participants=len(people),
                    positive_direction='EC-SI minus reference' if direction==1 else 'reference minus EC-SI'))
    assert len(vectors)==20
    results=paired_matrix_statistics(np.asarray(vectors).T,exact=(task=='seed_binary'),
                                     bootstrap_runs=bootstrap_runs,permutations=permutations)
    adjusted=holm([r['raw_p'] for r in results])
    return [dict(**meta,**result,holm_twenty_p=p,family_size=20)
            for meta,result,p in zip(metadata,results,adjusted)]


def mean_interval(values,draws=100000):
    values=np.asarray(values,dtype=float);rng=np.random.default_rng(20261116);boot=[]
    for start in range(0,draws,1000):
        ids=rng.integers(0,len(values),size=(min(1000,draws-start),len(values)))
        boot.append(values[ids].mean(axis=1))
    low,high=np.quantile(np.concatenate(boot),[.025,.975])
    return float(values.mean()),float(low),float(high)


def binary_auc(labels,scores,positive=1):
    labels=np.asarray(labels);scores=np.asarray(scores)
    d=scores[labels==positive,None]-scores[labels!=positive][None,:]
    assert d.size>0
    return float(np.mean((d>0)+.5*(d==0)))
