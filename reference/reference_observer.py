"""Observation/recovery sidecars around the unchanged pinned reference train_one.

No optimizer, training loop, sampler, model, or loss implementation lives here.
The caller must warm the deterministic TensorDataset loader before train_one.
"""
from pathlib import Path
import copy, hashlib, json, random, shutil
import numpy as np
import torch


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def thresholds(model):
    return {n:float(m.threshold) for n,m in model.named_modules() if hasattr(m,'threshold')}


def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),
            'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def cpu_state(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


class ReferenceObserver:
    def __init__(self, namespace, output_dir, contract):
        self.ns=namespace
        self.root=Path(output_dir).resolve()
        self.root.mkdir(parents=True,exist_ok=True)
        self.contract=copy.deepcopy(contract)
        self.original_save=namespace['save_ckpt']
        self.original_load=namespace['load_ckpt']
        self.original_evaluate=namespace['evaluate']
        self.evaluations=[]
        self.saved=[]
        self.resume=None
        self.selected=None
        namespace['save_ckpt']=self.save
        namespace['load_ckpt']=self.load
        namespace['evaluate']=self.evaluate

    def checked_path(self,path):
        path=Path(path).resolve()
        if not path.is_relative_to(self.root):raise RuntimeError('Checkpoint outside observer output directory')
        return path

    def evaluate(self,model,loader):
        result=self.original_evaluate(model,loader)
        self.evaluations.append({'predictions':np.asarray(result[0]).copy().tolist(),
                                 'labels':np.asarray(result[1]).copy().tolist()})
        return result

    def save(self,model,opt,epoch,metrics,path):
        path=self.checked_path(path)
        result=self.original_save(model,opt,epoch,metrics,path)
        selected=metrics.get('best_epoch')==epoch
        payload={'contract':self.contract,'epoch':int(epoch),
            'base_checkpoint_sha256':file_hash(path),'thresholds':thresholds(model),
            'rng':rng_state(),'metrics':copy.deepcopy(metrics),
            'model_state_dict':cpu_state(model)}
        if selected:
            self.selected={'epoch':int(epoch),'model_state_dict':payload['model_state_dict'],
                           'thresholds':payload['thresholds'],'selected_metrics':copy.deepcopy(metrics['best'])}
        if self.selected is None:raise RuntimeError('Selected model snapshot is missing')
        payload['selected_checkpoint']=self.selected
        # Two recovery slots: publish a pair only after both files are complete.
        # The prior ledger's slot stays untouched if the process crashes midway.
        slot='A' if int(epoch)%2==0 else 'B'
        recovery=path.with_suffix('.recovery_'+slot+'.pt')
        temporary=recovery.with_suffix('.pt.tmp')
        shutil.copyfile(path,temporary);temporary.replace(recovery)
        recovery_side=path.with_suffix('.recovery_'+slot+'.observer.pt')
        temporary=recovery_side.with_suffix('.pt.tmp')
        torch.save(payload,temporary);temporary.replace(recovery_side)
        side=path.with_suffix('.observer.pt')
        temporary=side.with_suffix('.pt.tmp')
        torch.save(payload,temporary);temporary.replace(side)
        if selected:
            best=path.with_suffix('.selected.pt')
            temporary=best.with_suffix('.pt.tmp')
            torch.save(payload,temporary);temporary.replace(best)
        self.saved.append({'epoch':int(epoch),'selected':selected,'base_sha256':payload['base_checkpoint_sha256'],
                           'sidecar':str(side),'sidecar_sha256':file_hash(side)})
        ledger={'contract':self.contract,'saved':self.saved,'evaluations':self.evaluations,
                'resume':self.resume,'latest_checkpoint':str(path),
                'recovery_checkpoint':str(recovery),'recovery_sidecar':str(recovery_side),
                'recovery_sidecar_sha256':file_hash(recovery_side),
                'selected_checkpoint':str(path.with_suffix('.selected.pt'))}
        target=path.with_suffix('.observer.json')
        temporary=target.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(ledger,indent=2));temporary.replace(target)
        return result

    def load(self,model,opt,path):
        path=self.checked_path(path)
        ledger_path=path.with_suffix('.observer.json')
        if not ledger_path.is_file():
            if path.exists():raise RuntimeError('Exact resume requires completed observer ledger')
            return self.original_load(model,opt,path)
        ledger=json.loads(ledger_path.read_text())
        if ledger['contract']!=self.contract:raise RuntimeError('Recovery ledger contract differs')
        recovery=self.checked_path(ledger['recovery_checkpoint'])
        side=self.checked_path(ledger['recovery_sidecar'])
        if file_hash(side)!=ledger['recovery_sidecar_sha256']:raise RuntimeError('Recovery sidecar hash differs')
        payload=torch.load(side,map_location='cpu',weights_only=False)
        if payload['contract']!=self.contract:raise RuntimeError('Observer contract differs')
        if payload['base_checkpoint_sha256']!=file_hash(recovery):raise RuntimeError('Recovery checkpoint hash differs')
        result=self.original_load(model,opt,recovery)
        if result[0]!=payload['epoch']+1:raise RuntimeError('Resume epoch differs')
        current=cpu_state(model)
        if set(current)!=set(payload['model_state_dict']) or not all(torch.equal(current[k],payload['model_state_dict'][k]) for k in current):
            raise RuntimeError('Base and sidecar model states differ')
        modules=dict(model.named_modules())
        for name,value in payload['thresholds'].items():modules[name].threshold=value
        restore_rng(payload['rng'])
        self.selected=payload['selected_checkpoint']
        self.saved=ledger['saved']
        self.evaluations=ledger['evaluations']
        self.resume={'epoch':payload['epoch'],'checkpoint':str(recovery),
                     'base_sha256':payload['base_checkpoint_sha256'],'sidecar_sha256':file_hash(side)}
        return result
