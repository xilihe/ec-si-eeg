#!/usr/bin/env python3
"""Serial CPU-only independent audits as complete frozen training blocks arrive."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import time

ROOT=Path('/home/student/Mamba_eeg_work')
EXP=ROOT/'artifacts/experiments/ecsi_final_gap_closure_20261116_v2'
EXEC=EXP/'execution/analysis_controller'
IMAGE='lab-pytorch:cuda12.4-py3.11-torch2.5.1'
PY=ROOT/'code/.venv-gpu/bin/python'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p,value):
    temp=p.with_suffix('.partial');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(p)


def state(**value):write(EXEC/'state.json',dict(observed_at=time.time(),**value))


def inspect(uid):
    p=subprocess.run(['lab','job','inspect',uid],capture_output=True,text=True,check=True,timeout=30)
    x=json.loads(p.stdout);return x.get('job',x)


def verify_output(output,proof_name):
    proof_path=output/proof_name
    if not proof_path.exists():return False
    value=json.loads(proof_path.read_text());assert value['status']=='PASS'
    # Output hashes are relative for component certificates and absolute for full task certificates.
    for filename,digest in value.get('output_hashes',{}).items():
        path=Path(filename);path=path if path.is_absolute() else output/path
        assert sha(path)==digest,path
    return True


def run(name,command,output,proof_name,spent):
    receipt_path=EXEC/(name+'.receipt.json');receipt=None
    if receipt_path.exists():receipt=json.loads(receipt_path.read_text())
    script=EXEC/(name+'.sh')
    body='#!/usr/bin/env bash\nset -euo pipefail\nexport OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2\n'+shlex.join(command)+'\n'
    if script.exists():assert script.read_text()==body
    else:script.write_text(body)
    if receipt is None:
        assert not output.exists(),('Unreceipted prior output: preserve and inspect',output)
        remaining=14400-spent
        assert remaining>=300,'Fixed CPU audit budget exhausted; preserve evidence'
        args=['lab','job','run','--gpu','0','--time',str(min(3600,int(remaining))),
              '--image',IMAGE,'--','bash',str(script)]
        uid=None
        with (EXEC/(name+'.log')).open('w',buffering=1) as log:
            proc=subprocess.Popen(args,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            for line in proc.stdout:
                log.write(line)
                match=re.search(r'queued:\s+(\S+)',line)
                if match:
                    assert uid is None;uid=match.group(1)
                    receipt=dict(job_uid=uid,script_sha256=sha(script),submitted_at=time.time())
                    write(receipt_path,receipt);state(status='RUNNING_CPU_AUDIT',phase=name,job_uid=uid,cpu_seconds=spent)
            proc.wait()
        assert uid is not None,'No scheduler UID; inspect before retry'
    assert receipt['script_sha256']==sha(script)
    uid=receipt['job_uid'];missing=0
    while True:
        try:job=inspect(uid)
        except (subprocess.SubprocessError,json.JSONDecodeError):time.sleep(15);continue
        write(EXEC/(name+'.job.json'),job)
        valid=verify_output(output,proof_name)
        if valid and (job['status']=='FINISHED' and job['exit_code']==0 or job['status']=='LOST' and job['exit_code']==125):
            start=datetime.fromisoformat(job['started_at'].replace('Z','+00:00')).timestamp()
            end=datetime.fromisoformat(job['finished_at'].replace('Z','+00:00')).timestamp() if job.get('finished_at') else (output/proof_name).stat().st_mtime
            seconds=max(0,end-start)
            write(EXEC/(name+'.complete.json'),dict(status='PASS',job_uid=uid,cpu_seconds=seconds,
                proof_sha256=sha(output/proof_name),official_job_status=job['status'],official_exit_code=job['exit_code']))
            return seconds
        if job['status'] in ('FAILED','CANCELLED') or job['status']=='FINISHED' and job['exit_code']!=0:
            raise RuntimeError(('CPU audit failed: preserve outputs, no automatic retry',name,job))
        if job['status']=='FINISHED' and not valid:
            missing+=1
            if missing>=4:raise RuntimeError(('Terminal audit missing required certificate',name,uid))
        time.sleep(15)


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);a=p.parse_args()
    EXEC.mkdir(parents=True,exist_ok=True)
    frozen=json.loads((a.release/'source_freeze.json').read_text())
    for name,digest in frozen.items():assert sha(a.release/name)==digest
    import sys
    sys.path.insert(0,str(a.release))
    from aggregate_ecsi_complete_curve_20261116 import task_components
    blocks=json.loads((EXP/'code/blocks.json').read_text());spent=0.;complete=[]
    for f in EXEC.glob('*.complete.json'):
        value=json.loads(f.read_text());assert value['status']=='PASS';spent+=value['cpu_seconds'];complete.append(f.name.removesuffix('.complete.json'))
    while True:
        progressed=False
        for task in ('seed_binary','faced'):
            for condition,bundle,output in task_components(EXP,task):
                name='component_'+output.name
                proof='independent_artifact_validation.json' if task=='seed_binary' else 'component_validation.json'
                if name in complete:
                    assert verify_output(output,proof);continue
                relevant=[b for b in blocks if b['bundle']==bundle.name]
                ready=bool(relevant) and all((EXP/'execution/block_certificates'/(b['block']+'.json')).exists() for b in relevant)
                if task=='seed_binary' and condition in ('standard','rho0.3'):ready=True
                if not ready:continue
                script='analyze_seed_curve_component_20261116.py' if task=='seed_binary' else 'audit_faced_curve_bundle_20261116.py'
                command=[str(PY),str(a.release/script),'--bundle',str(bundle),'--condition',condition,'--output',str(output)]
                spent+=run(name,command,output,proof,spent);complete.append(name);progressed=True
            name='aggregate_'+task;output=EXP/('analysis_'+task)
            needed=['component_'+directory.name for _,_,directory in task_components(EXP,task)]
            if name not in complete and all(n in complete for n in needed):
                command=[str(PY),str(a.release/'aggregate_ecsi_complete_curve_20261116.py'),'--experiment',str(EXP),'--task',task,'--output',str(output)]
                spent+=run(name,command,output,'validation.json',spent);complete.append(name);progressed=True
        if all('aggregate_'+task in complete for task in ('seed_binary','faced')):
            name='aggregate_combined';output=EXP/'analysis_combined'
            if name not in complete:
                command=[str(PY),str(a.release/'aggregate_ecsi_complete_curve_20261116.py'),'--experiment',str(EXP),'--task','combined','--output',str(output)]
                spent+=run(name,command,output,'validation.json',spent);complete.append(name)
            state(status='ANALYSIS_COMPLETE_MANUSCRIPT_PENDING',complete=complete,cpu_seconds=spent);return
        if not progressed:state(status='WAITING_FOR_COMPLETE_TRAINING_BLOCKS',complete=complete,cpu_seconds=spent)
        assert spent<=14400
        time.sleep(30)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        EXEC.mkdir(parents=True,exist_ok=True)
        state(status='NEEDS_ATTENTION',error=repr(error));raise
