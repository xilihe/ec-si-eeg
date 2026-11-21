"""Same-forward-graph encoder gradient diagnostics and norm control."""
import math
import torch


def measured_backward(task, adversarial, named_encoder, target_ratio=None):
    named = [(n, p) for n, p in named_encoder if p.requires_grad]
    params = [p for _, p in named]
    gt = torch.autograd.grad(task, params, retain_graph=True, allow_unused=True)
    ga = torch.autograd.grad(adversarial, params, retain_graph=True, allow_unused=True)
    gt = [torch.zeros_like(p) if g is None else g.detach() for p, g in zip(params, gt)]
    ga = [torch.zeros_like(p) if g is None else g.detach() for p, g in zip(params, ga)]
    t2 = sum(float((g.double()*g.double()).sum()) for g in gt)
    a2 = sum(float((g.double()*g.double()).sum()) for g in ga)
    dot = sum(float((g.double()*h.double()).sum()) for g, h in zip(gt, ga))
    tn, an = math.sqrt(t2), math.sqrt(a2)
    if not all(math.isfinite(v) for v in (tn, an, dot)):
        raise FloatingPointError("nonfinite encoder gradient")
    factor = 1.0
    if target_ratio is not None:
        if target_ratio < 0 or not math.isfinite(target_ratio):
            raise ValueError("target ratio must be finite and nonnegative")
        if tn <= 1e-14 or an <= 1e-14:
            if target_ratio != 0:
                raise FloatingPointError("undefined positive-ratio gradient normalization")
            factor = 0.0
        else:
            factor = target_ratio * tn / an
    (task + adversarial).backward()
    if target_ratio is not None:
        for p, g, h in zip(params, gt, ga):
            p.grad = g + factor*h
    effective_norm = math.sqrt(sum(float(((factor*h).double()**2).sum()) for h in ga))
    effective_ratio = effective_norm / tn if tn > 0 else 0.0
    if target_ratio is not None and abs(effective_ratio-target_ratio) > 1e-5:
        raise FloatingPointError("gradient norm target was not met")
    return dict(task_norm=tn, adversarial_norm=an, raw_ratio=an/tn if tn else 0.,
                factor=factor, effective_ratio=effective_ratio,
                cosine=dot/(tn*an) if tn*an else 0.,
                layers=[dict(parameter=n, task_norm=float(g.double().norm()),
                             adversarial_norm=float(h.double().norm()),
                             effective_adversarial_norm=factor*float(h.double().norm()))
                        for (n, _), g, h in zip(named, gt, ga)])
