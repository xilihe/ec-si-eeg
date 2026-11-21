"""Same gradients with batched diagnostic transfers and head-only backward."""
import math

import torch


def measured_backward(task, adversarial, named_encoder, target_ratio=None,
                      joint_loss=None, joint_parameters=None, zero_adversarial=False):
    named=[(n,p) for n,p in named_encoder if p.requires_grad]
    params=[p for _,p in named]
    gt=torch.autograd.grad(task,params,retain_graph=True,allow_unused=True)
    ga=([None]*len(params) if zero_adversarial else
        torch.autograd.grad(adversarial,params,retain_graph=True,allow_unused=True))
    gt=[torch.zeros_like(p) if g is None else g.detach() for p,g in zip(params,gt)]
    ga=[torch.zeros_like(p) if g is None else g.detach() for p,g in zip(params,ga)]
    # One device-to-host transfer preserves each tensor's double-precision sum.
    stats=torch.stack([torch.stack([(g.double()**2).sum(),(h.double()**2).sum(),
                                   (g.double()*h.double()).sum()]) for g,h in zip(gt,ga)]).cpu().tolist()
    tn=math.sqrt(sum(x[0] for x in stats));an=math.sqrt(sum(x[1] for x in stats))
    dot=sum(x[2] for x in stats)
    if not all(math.isfinite(v) for v in (tn,an,dot)):
        raise FloatingPointError('nonfinite encoder gradient')
    factor=1.
    if target_ratio is not None:
        if target_ratio<0 or not math.isfinite(target_ratio):raise ValueError('Invalid target ratio')
        if tn<=1e-14 or an<=1e-14:
            if target_ratio!=0:raise FloatingPointError('undefined positive-ratio gradient normalization')
            factor=0.
        else:factor=target_ratio*tn/an
    joint=task+adversarial if joint_loss is None else joint_loss
    if target_ratio is not None and joint_parameters is not None:
        # Encoder gradients below are supplied explicitly. The original joint
        # graph still determines every head gradient, without a third encoder VJP.
        joint.backward(inputs=list(joint_parameters))
    else:
        joint.backward()
    if target_ratio is not None:
        for p,g,h in zip(params,gt,ga):p.grad=g+factor*h
    effective2=torch.stack([((factor*h).double()**2).sum() for h in ga]).cpu().tolist()
    effective_ratio=math.sqrt(sum(effective2))/tn if tn>0 else 0.
    if target_ratio is not None and abs(effective_ratio-target_ratio)>1e-5:
        raise FloatingPointError('gradient norm target was not met')
    return dict(task_norm=tn,adversarial_norm=an,raw_ratio=an/tn if tn else 0.,
        factor=factor,effective_ratio=effective_ratio,cosine=dot/(tn*an) if tn*an else 0.,
        layers=[dict(parameter=n,task_norm=math.sqrt(s[0]),adversarial_norm=math.sqrt(s[1]),
                     effective_adversarial_norm=factor*math.sqrt(s[1])) for (n,_),s in zip(named,stats)])
