#!/usr/bin/env python3
"""Frozen FACED source instrumentation for a complete common-gradient curve."""
import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys

import torch

import run_faced_ecsi_global_control as core
from ecsi_gradient_norm_control_v3_20261116 import measured_backward

RATIO = None
TRANSFORMED_SHA = None
LEARNING_FILES = (
    'run_faced_ecsi_gradient_tradeoff_v3_20261116.py', 'ecsi_gradient_norm_control_v3_20261116.py',
    'run_faced_ecsi_global_control.py', 'run_faced_cross_subject.py', 'run_seed_cross_session.py',
    'seed_pipeline.py', 'ecsi_conditional_hsic.py', 'ecsi_dual_video_mixstyle.py',
    'ecsi_gradient_guard.py', 'ecsi_hierarchical_rex.py', 'ecsi_same_basin_soup.py',
)


def state_sha(model):
    return hashlib.sha256(b''.join(t.detach().cpu().numpy().tobytes()
                                  for t in model.state_dict().values())).hexdigest()


def controlled_step(model, task, auxiliary, joint, variant, progress):
    if not bool(torch.isfinite(joint)):
        raise FloatingPointError('Nonfinite original joint loss')
    encoder_ids = {id(p) for p in model.encoder_parameters()}
    named = [(name, p) for name, p in model.named_parameters() if id(p) in encoder_ids]
    assert len(named) == len(encoder_ids)
    target = None if RATIO is None else RATIO * (2/(1+math.exp(-10*progress))-1)
    assert variant in ('ecsi_ce', 'ecsi_adv', 'ecsi_global_adv')
    if variant == 'ecsi_ce':
        assert RATIO is None, 'CE belongs to the standard policy only'
    head_parameters = [p for p in model.parameters() if id(p) not in encoder_ids and p.requires_grad]
    return measured_backward(task, auxiliary, named, target_ratio=target, joint_loss=joint,
                             joint_parameters=head_parameters, zero_adversarial=(variant == 'ecsi_ce'))


def finish_record(model, train, args, seed, variant, initial_sha, steps):
    people = sorted(set(map(int, train['subject'])))
    videos = sorted(set(map(int, train['trial'])))
    key = core.object_sha256([people, videos, seed, variant])[:24]
    directory = Path(args.bundle_dir)/'gradient_records'
    directory.mkdir(exist_ok=True)
    path = directory/(key+'.json')
    if path.exists():
        raise FileExistsError('Preserve previous completed gradient record; inspect before retry: '+str(path))
    core.write_json_atomic(path, dict(variant=variant, seed=seed, train_subjects=people,
        train_videos=videos, target_ratio=RATIO, initial_state_sha256=initial_sha,
        restored_checkpoint_state_sha256=state_sha(model),
        transformed_train_sha256=TRANSFORMED_SHA, steps=steps))
    return dict(initial_state_sha256=initial_sha, gradient_record=str(path),
                gradient_record_sha256=core.sha256(path),
                gradient_policy='standard' if RATIO is None else 'encoder_norm_ratio',
                encoder_norm_ratio=-1 if RATIO is None else RATIO)


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise AssertionError('The frozen training source no longer has one expected anchor: '+old)
    return source.replace(old, new, 1)


def instrument():
    """Retain the original loop, forward, losses, optimizer and checkpoint rule."""
    source = inspect.getsource(core.train_variant)
    source = replace_once(source, '    set_seed(seed)\n',
        "    assert variant in ('ecsi_ce','ecsi_adv','ecsi_global_adv') and guard_mode == 'none'\n"
        "    initial_state_sha = _curve_state_sha(model)\n"
        "    gradient_steps = []\n"
        '    set_seed(seed)\n')
    source = replace_once(source, '            else:\n                loss.backward()\n',
        '            else:\n'
        '                gradient_info = _curve_controlled_step(model, emotion_loss,\n'
        '                    conditional_adversarial_loss + global_adversarial_loss,\n'
        '                    loss, variant, progress)\n'
        '                gradient_info.update(epoch=epoch, batch=batch_index, progress=progress,\n'
        '                                     grl_strength=reversal_lambda)\n'
        '                gradient_steps.append(gradient_info)\n')
    source = replace_once(source, '    model.load_state_dict(best_state)\n    return model, {\n',
        '    model.load_state_dict(best_state)\n'
        '    gradient_metadata = _curve_finish_record(model, train, args, seed, variant,\n'
        '                                             initial_state_sha, gradient_steps)\n'
        '    return model, {\n'
        '        **gradient_metadata,\n')
    core._curve_state_sha = state_sha
    core._curve_controlled_step = controlled_step
    core._curve_finish_record = finish_record
    exec(compile(source, '<frozen FACED train_variant with gradient records>', 'exec'), core.__dict__)
    return source


def main():
    global RATIO, TRANSFORMED_SHA
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--curve-protocol', type=Path, required=True)
    parser.add_argument('--encoder-norm-ratio', type=float)
    extra, remaining = parser.parse_known_args()
    if not extra.curve_protocol.is_file():
        raise FileNotFoundError(extra.curve_protocol)
    RATIO = extra.encoder_norm_ratio
    if RATIO is not None and RATIO not in (0.1, 0.3, 1.0, 3.0):
        raise ValueError('Only the frozen common curve is authorized')
    transformed = instrument()
    TRANSFORMED_SHA = hashlib.sha256(transformed.encode()).hexdigest()
    sys.argv = [sys.argv[0], *remaining]
    base_parse = core.parse_args

    def parse():
        args = base_parse()
        requested = [args.model_variant] if args.model_variant else args.models.split(',')
        assert set(requested) <= {'ecsi_ce', 'ecsi_adv', 'ecsi_global_adv'}
        assert args.adv_schedule == 'dann' and args.adv_lambda == 1
        if RATIO is not None:
            assert 'ecsi_ce' not in requested
        args.bundle_dir.mkdir(parents=True, exist_ok=True)
        sources = {name:core.sha256(Path(__file__).parent/name) for name in sorted(LEARNING_FILES)}
        policy = dict(wrapper_sha256=core.sha256(Path(__file__)),
            core_sha256=core.sha256(Path(core.__file__)),
            gradient_module_sha256=core.sha256(Path(__file__).with_name('ecsi_gradient_norm_control_v3_20261116.py')),
            protocol_sha256=core.sha256(extra.curve_protocol), encoder_norm_ratio=RATIO,
            transformed_train_sha256=TRANSFORMED_SHA, source_hashes=sources,
            investigation='fixed_complete_FACED_gradient_tradeoff', source_instrumentation=True)
        path = args.bundle_dir/'curve_policy.json'
        if path.exists():
            assert json.loads(path.read_text()) == policy, 'Do not mix code/protocol versions'
        else:
            core.write_json_atomic(path, policy)
            (args.bundle_dir/'instrumented_train_variant.py.txt').write_text(transformed)
        return args

    core.parse_args = parse
    torch.set_num_threads(2)
    core.main()


if __name__ == '__main__':
    main()
