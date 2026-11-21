#!/usr/bin/env python3
"""Release finished-fold arrays; keep frozen V3 numerical learning unchanged."""
import gc
import hashlib
import inspect
import json
from pathlib import Path

import run_faced_ecsi_gradient_tradeoff_v3_20261116 as driver

CLEANUP_NAMES=('train','val','test','probe_test','train_x','val_x','test_x','probe_test_x',
               'train_de','val_de','test_de','probe_test_de')


def memory_event(event):
    values={}
    for name in ('memory.current','memory.peak','memory.max','memory.events'):
        path=Path('/sys/fs/cgroup')/name
        if path.exists():values[name]=path.read_text().strip()
    print(json.dumps(dict(scope='resource diagnostics only',event=event,cgroup=values)),flush=True)


def install():
    original=inspect.getsource(driver.core.main)
    anchor='\n    metrics_path = bundle / "metrics.csv"\n'
    assert original.count(anchor)==1
    insert='\n            del '+', '.join(CLEANUP_NAMES)+'\n'
    insert+='            _ecsi_resource_collect()\n'
    insert+='            _ecsi_memory_event("after_stimulus_fold_cleanup")\n'
    transformed=original.replace(anchor,insert+anchor)
    driver.core.__dict__['_ecsi_resource_collect']=gc.collect
    driver.core.__dict__['_ecsi_memory_event']=memory_event
    driver.core.__dict__['_ecsi_memory_main_source']=transformed
    exec(compile(transformed,'<resource_only_memory_lifecycle_main>','exec'),driver.core.__dict__)
    return dict(status='ACTIVE_RESOURCE_POLICY',scope='Array lifetime only; numerical preprocessing and learning operations unchanged',
                launcher_sha256=driver.core.sha256(Path(__file__)),
                original_core_file_sha256=driver.core.sha256(Path(driver.core.__file__)),
                original_main_sha256=hashlib.sha256(original.encode()).hexdigest(),
                memory_safe_main_sha256=hashlib.sha256(transformed.encode()).hexdigest(),
                cleanup_names=list(CLEANUP_NAMES),learning_wrapper_sha256=driver.core.sha256(Path(driver.__file__)))


def main():
    policy=install();base_parse=driver.core.parse_args
    def parse():
        args=base_parse();args.bundle_dir.mkdir(parents=True,exist_ok=True)
        path=args.bundle_dir/'memory_lifecycle_policy.json'
        if path.exists():assert json.loads(path.read_text())==policy
        else:driver.core.write_json_atomic(path,policy)
        source=args.bundle_dir/'memory_safe_main.py.txt'
        if source.exists():assert hashlib.sha256(source.read_bytes()).hexdigest()==policy['memory_safe_main_sha256']
        else:source.write_text(driver.core.__dict__['_ecsi_memory_main_source'])
        memory_event('before_reference_main')
        return args
    driver.core.parse_args=parse
    driver.main()


if __name__=='__main__':main()
