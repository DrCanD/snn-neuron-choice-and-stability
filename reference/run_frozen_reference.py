"""Prospective DIAT cohort: call unchanged reference train_one; no replacement loop."""
import sys
sys.dont_write_bytecode=True
import argparse, ast, copy, hashlib, importlib.util, json, math, os, platform, random, subprocess, time, traceback
from pathlib import Path
from types import SimpleNamespace
from dataclasses import dataclass,field
from datetime import datetime,timezone
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset

BUNDLE=Path(__file__).resolve().parent
from runtime import sources as pinned_sources
CLI_ARGS=None
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def thash(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def shash(state):
    h=hashlib.sha256()
    for name,t in sorted(state.items()):
        h.update(name.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False));tmp.replace(p)
def require(ok,message):
    if not ok:raise RuntimeError(message)
def extract(path,names):
    s=Path(path).read_text();lines=s.splitlines(keepends=True);out={}
    for n in ast.parse(s).body:
        start=min([d.lineno for d in getattr(n,'decorator_list',[])]+[n.lineno])
        targets=[n.name] if isinstance(n,(ast.FunctionDef,ast.ClassDef)) else [t.id for t in n.targets if isinstance(t,ast.Name)] if isinstance(n,ast.Assign) else []
        for name in targets:
            if name in names:out[name]=''.join(lines[start-1:n.end_lineno])
    require(set(out)==set(names),'Missing pinned definitions')
    return out
def setup():
    import sklearn,scipy,h5py
    contract=read(BUNDLE/'reference_contract.json');plan=read(BUNDLE/'experiment_plan.json')
    require(sha(BUNDLE/'reference_contract.json')==plan['contract_sha256'],'Contract differs from fixed plan')
    for name,digest in read(BUNDLE/'bundle_manifest.json')['files'].items():require(sha(BUNDLE/name)==digest,'Bundle hash differs: '+name)
    env=dict(python=platform.python_version(),numpy=np.__version__,torch=str(torch.__version__),
        cuda_runtime=torch.version.cuda,cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        scikit_learn=sklearn.__version__,scipy=scipy.__version__,h5py=h5py.__version__)
    require(env==contract['environment'],'Unvalidated environment: '+str(env))
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.benchmark=True;torch.use_deterministic_algorithms(False)
    repo,lib,core=pinned_sources(CLI_ARGS.sources)
    from dikmen_neurons import NeuronRegistry
    from dikmen_neurons.ms_if import CrossInhibLIF
    require(Path(core.__file__).resolve()==(repo/'src/mstask/core.py').resolve(),'Wrong imported core')
    definitions=read(BUNDLE/'definitions.json')
    require(sha(BUNDLE/'definitions.json')==contract['definitions_sha256'],'Definitions changed')
    require(sha(BUNDLE/'reference_observer.py')==contract['observer_sha256'],'Observer changed')
    require(extract(repo/'experiments/02_diat_ablation.py',list(definitions['source']))==definitions['source'],'Extracted functions differ from tested reference')
    spec=importlib.util.spec_from_file_location('frozen_observer',BUNDLE/'reference_observer.py')
    observer=importlib.util.module_from_spec(spec);sys.modules[spec.name]=observer;spec.loader.exec_module(observer)
    source={k:v for k,v in definitions['source'].items() if k!='load_diat'}
    context=dict(torch=torch,np=np,copy=copy,math=math,audit=SimpleNamespace(os=os,R1_DEFS=definitions['R1_DEFS']),
        Path=Path,dataclass=dataclass,field=field,DataLoader=DataLoader,TensorDataset=TensorDataset,
        core=core,suite=SimpleNamespace(device=torch.device('cuda')),CrossInhibLIF=CrossInhibLIF,
        NeuronRegistry=NeuronRegistry,STABLE_CLASS_SOURCE=definitions['STABLE_CLASS_SOURCE'],source=source)
    exec(compile(definitions['factory'],'tested_namespace_factory','exec'),context)
    factory=context['namespace'];ns=factory(200)
    require(all(getattr(ns['cfg'],k)==v for k,v in contract['config'].items()),'Config changed')
    ns['get_dataset_path']=lambda name: CLI_ARGS.data
    exec(compile(definitions['source']['load_diat'],'pinned_02:load_diat','exec'),ns)
    data_path=CLI_ARGS.data
    for name,digest in contract['raw_data_sha256'].items():require(sha(data_path/name)==digest,'Raw data changed')
    data=ns['load_diat']()
    hashes={n:thash(t) for n,t in zip(contract['six_array_hashes'],data)}
    require(hashes==contract['six_array_hashes'],'Split or normalization changed')
    return contract,plan,factory,observer,data,env

def run(seed,preflight_only=False):
    started=time.perf_counter();contract,plan,factory,observer,data,env=setup()
    root=CLI_ARGS.output;root.mkdir(parents=True,exist_ok=True)
    worker_root=root;worker_root.mkdir(exist_ok=True)
    # Each seed has exactly one assigned Colab. Local flock also rejects duplicate launch in that VM.
    import fcntl
    lock=open(worker_root/'.reference.lock','a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    jobs=[j for j in plan['jobs'] if j['seed']==seed];require(len(jobs)==2,'Seed outside fixed plan')
    gate={'status':'PREFLIGHT_PASSED','environment':env,'data_verified':True,'contract_sha256':plan['contract_sha256'],
          'runner_sha256':sha(__file__),'time_utc':datetime.now(timezone.utc).isoformat(),'seed':seed,'training_started':False}
    write(worker_root/'preflight.json',gate);print(json.dumps(gate),flush=True)
    if preflight_only:return
    for job in jobs:
        out=worker_root/job['id'];out.mkdir(exist_ok=True)
        result_path=out/'result.json'
        if result_path.exists():
            old=read(result_path);require(old['contract_sha256']==plan['contract_sha256'] and old['selected_replay_exact'],'Existing result failed integrity check')
            print('SKIP_VERIFIED',job['id'],flush=True);continue
        ns=factory(200);state={};build=ns['build_model']
        def builder(*args,**kwargs):
            model=build(*args,**kwargs);state['model']=model;state['initial_state_sha256']=shash(model.state_dict())
            return model
        ns['build_model']=builder
        run_contract={'reference_contract_sha256':plan['contract_sha256'],'job':job,'environment':env,'loader_contract':contract['loader_contract']}
        obs=observer.ReferenceObserver(ns,out,run_contract)
        original_save=ns['save_ckpt'];job_start=time.perf_counter();last=[job_start];timings=[]
        def progress_save(model,opt,epoch,metrics,path):
            ret=original_save(model,opt,epoch,metrics,path)
            now=time.perf_counter();timings.append(now-last[0]);last[0]=now
            write(out/'progress.json',{'status':'RUNNING','job':job,'epoch_zero_based':epoch,'selected_epoch_zero_based':metrics['best_epoch'],
                'patience_counter':metrics['patience_ctr'],'session_elapsed_seconds':now-job_start,'epoch_seconds':timings[-10:],
                'updated_utc':datetime.now(timezone.utc).isoformat(),'contract_sha256':plan['contract_sha256']})
            return ret
        ns['save_ckpt']=progress_save
        random.seed(9001);np.random.seed(9001);torch.manual_seed(777)
        tr,va,te=ns['make_loaders'](data);warm=iter(tr);next(warm)
        try:
            # A crash after the terminal epoch's durable save must not add another epoch.
            ckpt=out/(job['method']+'_seed'+str(seed)+'.pt');ledger=ckpt.with_suffix('.observer.json')
            terminal=False
            if ledger.exists():
                ld=read(ledger);side=Path(ld['recovery_sidecar'])
                require(side.resolve().is_relative_to(out.resolve()),'Sidecar outside this job')
                require(sha(side)==ld['recovery_sidecar_sha256'],'Recovery sidecar changed')
                payload=torch.load(side,map_location='cpu',weights_only=False)
                terminal=payload['metrics']['patience_ctr']>=ns['cfg'].patience or payload['epoch']+1>=ns['cfg'].max_epochs
            if terminal:
                model=builder(job['method'],64,6,contract['neuron_kwargs'],contract['specs']).cuda()
                opt=torch.optim.Adam(model.parameters(),lr=ns['cfg'].lr);_,metrics=obs.load(model,opt,ckpt)
                best,fire_log=metrics['best'],metrics['fire_log']
            else:
                best,fire_log=ns['train_one'](job['method'],seed,64,6,contract['neuron_kwargs'],contract['specs'],tr,va,te,data[0][:128],out)
            require(obs.selected is not None,'No selected full-state snapshot')
            replay=build(job['method'],64,6,contract['neuron_kwargs'],contract['specs']).cuda()
            replay.load_state_dict(obs.selected['model_state_dict']);modules=dict(replay.named_modules())
            for name,value in obs.selected['thresholds'].items():modules[name].threshold=value
            pred,true=obs.original_evaluate(replay,te)
            require(pred.tolist()==best['preds'] and true.tolist()==best['trues'],'Selected snapshot does not replay test predictions')
            selected_path=out/'verified_selected.pt'
            tmp=selected_path.with_suffix('.pt.tmp');torch.save(obs.selected,tmp);tmp.replace(selected_path)
            result={'status':'COMPLETED','job':job,'contract_sha256':plan['contract_sha256'],'environment':env,
                'best':best,'fire_log':fire_log,'selected_epoch_zero_based':obs.selected['epoch'],'epochs_trained':len(fire_log),
                'initial_state_sha256':state['initial_state_sha256'],'selected_replay_exact':True,
                'selected_checkpoint':str(selected_path),'selected_checkpoint_sha256':sha(selected_path),'resume':obs.resume,
                'session_seconds':time.perf_counter()-job_start,'completed_utc':datetime.now(timezone.utc).isoformat()}
            write(result_path,result);write(out/'progress.json',{'status':'COMPLETED','job':job,'epochs_trained':len(fire_log),'result_sha256':sha(result_path)})
            print('COMPLETED',job['id'],'epochs',len(fire_log),'seconds',result['session_seconds'],flush=True)
            del replay
        finally:
            if tr._iterator is not None:tr._iterator._shutdown_workers()
        del ns,obs,state,tr,va,te;torch.cuda.empty_cache()
    write(worker_root/'complete.json',{'status':'COMPLETED','seed':seed,'jobs':jobs,'seconds':time.perf_counter()-started,'contract_sha256':plan['contract_sha256']})

def run_cli(args, jobs):
    global CLI_ARGS
    CLI_ARGS = args
    run(args.seed)
