"""Synthetic software fixtures, never research evidence."""
import copy
from pathlib import Path
import sys
import unittest
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from ecsi_gradient_norm_control_v3_20261116 import measured_backward


class NormControlTests(unittest.TestCase):
    def test_passive_measurement_does_not_change_gradients_or_rng(self):
        torch.manual_seed(33)
        model=nn.Sequential(nn.Linear(3,4),nn.GELU(),nn.Linear(4,2))
        x=torch.randn(8,3); y=torch.arange(8)%2
        reference=copy.deepcopy(model)
        out=reference(x); task=F.cross_entropy(out,y); aux=out.square().mean()
        (task+aux).backward()
        out=model(x); task=F.cross_entropy(out,y); aux=out.square().mean()
        rng=torch.get_rng_state().clone()
        info=measured_backward(task,aux,model.named_parameters())
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for a,b in zip(model.parameters(),reference.parameters()):
            torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)
        self.assertGreater(info['task_norm'],0)

    def test_norm_changes_only_encoder_and_matches_target(self):
        torch.manual_seed(11)
        encoder=nn.Linear(3,4); taskhead=nn.Linear(4,2); advhead=nn.Linear(4,3)
        x=torch.randn(8,3); y=torch.arange(8)%2; a=torch.arange(8)%3
        z=encoder(x); task=F.cross_entropy(taskhead(z),y); aux=-F.cross_entropy(advhead(z),a)
        taskgrad=torch.autograd.grad(task,list(encoder.parameters()),retain_graph=True)
        expected_heads=torch.autograd.grad(task+aux,[*taskhead.parameters(),*advhead.parameters()],retain_graph=True)
        info=measured_backward(task,aux,encoder.named_parameters(),.3)
        tn=torch.sqrt(sum((g.double()**2).sum() for g in taskgrad))
        an=torch.sqrt(sum(((p.grad-g).double()**2).sum() for p,g in zip(encoder.parameters(),taskgrad)))
        self.assertAlmostEqual(float(an/tn),.3,places=6)
        self.assertAlmostEqual(info['effective_ratio'],.3,places=6)
        for p,g in zip([*taskhead.parameters(),*advhead.parameters()],expected_heads):
            torch.testing.assert_close(p.grad,g)

    def test_ce_zero_auxiliary_is_valid_without_control(self):
        model=nn.Linear(2,2); z=model(torch.ones(4,2)); task=z.square().mean()
        info=measured_backward(task,z.sum()*0,model.named_parameters())
        self.assertEqual(info['adversarial_norm'],0)

    def test_zero_gradient_cannot_falsely_pass_positive_norm_control(self):
        model=nn.Linear(2,2); z=model(torch.ones(4,2)); task=z.square().mean()
        with self.assertRaises(FloatingPointError):
            measured_backward(task,z.sum()*0,model.named_parameters(),.3)

    def test_unused_encoder_parameters_supported(self):
        model=nn.Linear(2,2); unused=nn.Parameter(torch.ones(2))
        z=model(torch.ones(4,2)); task=z.square().mean()
        info=measured_backward(task,z.mean(),[*model.named_parameters(),('unused',unused)],.1)
        self.assertEqual(info['layers'][-1]['task_norm'],0.)


class FullJointGraphTests(unittest.TestCase):
    def test_extra_head_term_retains_exact_joint_head_gradient(self):
        torch.manual_seed(2)
        enc=nn.Linear(3,4); th=nn.Linear(4,2); ah=nn.Linear(4,3)
        x=torch.randn(8,3); z=enc(x)
        task=F.cross_entropy(th(z),torch.arange(8)%2)
        aux=-F.cross_entropy(ah(z),torch.arange(8)%3)
        joint=task+aux+.07*th.weight.square().sum()
        expected=torch.autograd.grad(joint,[*th.parameters(),*ah.parameters()],retain_graph=True)
        info=measured_backward(task,aux,enc.named_parameters(),.3,joint_loss=joint)
        for p,g in zip([*th.parameters(),*ah.parameters()],expected):
            torch.testing.assert_close(p.grad,g,atol=0,rtol=0)
        self.assertAlmostEqual(info['effective_ratio'],.3,places=6)

    def test_complete_joint_passive_matches_extra_zero_branches(self):
        torch.manual_seed(8)
        model=nn.Linear(3,4); ref=copy.deepcopy(model); x=torch.randn(8,3)
        out=ref(x); t=out.square().mean(); a=-out.mean()
        (t+a+0*out.sum()+0*out.square().sum()).backward()
        out=model(x); t=out.square().mean(); a=-out.mean()
        measured_backward(t,a,model.named_parameters(),joint_loss=t+a+0*out.sum()+0*out.square().sum())
        for p,q in zip(model.parameters(),ref.parameters()):torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)

    def test_exact_training_instrumentation_anchor(self):
        import run_faced_ecsi_gradient_tradeoff_v3_20261116 as runner
        text=runner.instrument()
        self.assertEqual(text.count('_curve_controlled_step'),1)
        self.assertEqual(text.count('_curve_finish_record'),1)
        self.assertIn('optimizer.step()',text)
        self.assertIn('model.load_state_dict(best_state)',text)

class OptimizationParityTests(unittest.TestCase):
    def test_all_targets_preserve_joint_head_and_encoder_gradients(self):
        from ecsi_gradient_norm_control_v2_20261116 import measured_backward as original
        for ratio in (None,.1,.3,1.,3.):
            torch.manual_seed(91)
            encoder=nn.Linear(3,5); th=nn.Linear(5,2); ah=nn.Linear(5,4); unused=nn.Linear(5,3)
            model=nn.ModuleList([encoder,th,ah,unused]);reference=copy.deepcopy(model);x=torch.randn(9,3)
            def losses(m):
                z=m[0](x);task=F.cross_entropy(m[1](z),torch.arange(9)%2)
                aux=-.7*F.cross_entropy(m[2](z),torch.arange(9)%4)
                joint=task+aux+0*z.sum()+.07*m[1].weight.square().sum()
                return task,aux,joint
            t,a,j=losses(reference);old=original(t,a,reference[0].named_parameters(),ratio,joint_loss=j)
            t,a,j=losses(model);rng=torch.get_rng_state().clone()
            new=measured_backward(t,a,model[0].named_parameters(),ratio,joint_loss=j,
                  joint_parameters=[*model[1].parameters(),*model[2].parameters(),*model[3].parameters()])
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            for left,right in zip(model.parameters(),reference.parameters()):
                if right.grad is None:self.assertIsNone(left.grad)
                else:torch.testing.assert_close(left.grad,right.grad,rtol=0,atol=0)
            for key in ('task_norm','adversarial_norm','raw_ratio','factor','effective_ratio','cosine'):
                self.assertEqual(new[key],old[key])

    def test_known_zero_auxiliary_shortcut_preserves_full_backward(self):
        from ecsi_gradient_norm_control_v2_20261116 import measured_backward as original
        torch.manual_seed(42);model=nn.Linear(3,4);reference=copy.deepcopy(model);x=torch.randn(7,3)
        z=reference(x);t=z.square().mean();a=z.sum()*0
        original(t,a,reference.named_parameters(),joint_loss=t+a)
        z=model(x);t=z.square().mean();a=z.sum()*0
        measured_backward(t,a,model.named_parameters(),joint_loss=t+a,zero_adversarial=True)
        for left,right in zip(model.parameters(),reference.parameters()):torch.testing.assert_close(left.grad,right.grad,rtol=0,atol=0)

if __name__=='__main__': unittest.main()


