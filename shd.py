"""Bounded SHD factorial follow-up. --preflight never launches scientific training."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys
sys.dont_write_bytecode=True
import argparse, ast, copy, hashlib, json, math, platform, random, subprocess, time
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score

BUNDLE=Path(__file__).resolve().parent
from runtime import sources as pinned_sources, verify_package

def require(ok,message):
    if not ok:raise RuntimeError(message)
def utc():return datetime.now(timezone.utc).isoformat()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def ahash(a,metadata=False):
    if torch.is_tensor(a):a=a.detach().cpu().contiguous().numpy()
    a=np.ascontiguousarray(a);h=hashlib.sha256()
    if metadata:h.update(str(a.dtype).encode());h.update(str(a.shape).encode())
    h.update(a.tobytes());return h.hexdigest()
def statehash(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):h.update(k.encode());h.update(ahash(v,True).encode())
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False,ensure_ascii=False),encoding='utf-8');tmp.replace(path)
def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');torch.save(obj,tmp);tmp.replace(path)
    write(path.with_suffix('.sha256.json'),{'sha256':sha(path)})
def load(path):
    require(sha(path)==read(path.with_suffix('.sha256.json'))['sha256'],'Checkpoint hash mismatch')
    return torch.load(path,map_location='cuda',weights_only=False)
def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),
            'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}
def restore_rng(r):
    random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['cpu'].cpu())
    torch.cuda.set_rng_state_all([v.cpu() for v in r['cuda']])
def rng_hash():
    r=rng_state()
    return {'python':repr(r['python']),'numpy':repr(r['numpy']),
            'cpu':ahash(r['cpu']),'cuda':[ahash(x) for x in r['cuda']]}
def permutation(seed,epoch,n):
    return torch.randperm(n,generator=torch.Generator().manual_seed(1000000+seed+10000*epoch))
def extract_function(path,name):
    s=path.read_text();lines=s.splitlines(keepends=True)
    node=next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name==name)
    return ''.join(lines[node.lineno-1:node.end_lineno])


class Study:
    def __init__(self,args):
        seed=args.seed;self.args=args
        import scipy,sklearn,h5py
        self.seed=seed;self.contract=read(BUNDLE/'configs/shd.json');self.chash=sha(BUNDLE/'configs/shd.json')
        verify_package()
        self.runner_sha=sha(__file__)
        require(seed in self.contract['seeds'],'Unexpected seed')
        require(torch.cuda.is_available(),'GPU required; no training started')
        self.env={'python':platform.python_version(),'numpy':np.__version__,'torch':str(torch.__version__),
                  'cuda_runtime':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
                  'gpu':torch.cuda.get_device_name(0),'scikit_learn':sklearn.__version__,
                  'scipy':scipy.__version__,'h5py':h5py.__version__}
        require(self.env==self.contract['environment'],'Unvalidated environment: '+str(self.env))
        torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
        torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
        self.cfg=self.contract['config'];self.root=args.output
        self.worker=self.root;self.worker.mkdir(parents=True,exist_ok=True)
        import fcntl
        self.lock=open(self.worker/'.shd.lock','a')
        fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (self.root/'contract.json').exists():require(sha(self.root/'contract.json')==self.chash,'Existing campaign differs')
        else:(self.root/'contract.json').write_bytes((BUNDLE/'configs/shd.json').read_bytes())
        self.source_setup();self.data_setup()
        self.initial={}
        print('[SETUP_READY]',json.dumps({'seed':seed,'environment':self.env,'contract':self.chash}),flush=True)

    def source_setup(self):
        repo,lib,core=pinned_sources(self.args.sources)
        from dikmen_neurons import BaseNeuron,NeuronRegistry
        from dikmen_neurons.ms_if import CrossInhibLIF
        require(Path(core.__file__).resolve()==(repo/'src/mstask/core.py').resolve(),'Wrong core binding')
        coupling=self.cfg['stable_coupling']
        class Stable(CrossInhibLIF):
            def __init__(self,size,beta=.95,threshold=1.):
                super().__init__(size,beta,threshold)
                with torch.no_grad():
                    self.w_inh.fill_(math.log(math.expm1(coupling)));self.w_exc.fill_(math.log(math.expm1(coupling)))
        NeuronRegistry._all['SHDFollowupStable']=Stable
        class AnchorNet(core.VerticalNet):
            def forward(self,x):
                require(x.shape[1] in (48,96),'Unexpected number of updates')
                features,spikes=self.stack(x)
                if x.shape[1]==96:features=spikes[:,1::2,:].mean(dim=1)
                return self.readout(features)
        self.core,self.BaseNeuron,self.Net=core,BaseNeuron,AnchorNet
        ns={'np':np};exec(extract_function(repo/'experiments/07_shd_disjoint.py','_bin_h5'),ns)
        self.bin_h5=ns['_bin_h5']

    def data_setup(self):
        from sklearn.model_selection import train_test_split
        arrays=[]
        for split in ('train','test'):
            path=self.args.data/f'shd_{split}.h5'
            require(path.is_file(),'Official SHD file missing')
            require(sha(path)==self.contract['raw_sha256'][split],'Raw SHD changed')
            print('[DATA]',split,'raw hash verified',flush=True)
            arrays.append(self.bin_h5(path,48,700,1.))
        (x,y),(xt,yt)=arrays
        it,iv=train_test_split(np.arange(len(y)),test_size=.15,stratify=y,random_state=0)
        values={'train_index_sha256':it,'validation_index_sha256':iv,'train_labels_sha256':y,'test_labels_sha256':yt}
        for key,v in values.items():require(ahash(v,True)==self.contract['split_hashes'][key],'Split differs: '+key)
        require((len(it),len(iv),len(yt))==(6932,1224,2264),'Split size differs')
        # Permutations depend only on sample index and a fixed RNG, never labels or model seeds.
        gen=np.random.default_rng(self.contract['shuffle_seed'])
        def make_perms(n):
            return np.argsort(gen.random((n,12,4)),axis=2).astype(np.int64)
        p=make_perms(len(y));pt=make_perms(len(yt))
        self.data={k:(torch.from_numpy(xx.copy()),torch.from_numpy(yy.copy()).long(),torch.from_numpy(pp.copy()))
                   for k,xx,yy,pp in [('train',x[it],y[it],p[it]),('val',x[iv],y[iv],p[iv]),('test',xt,yt,pt)]}
        self.data_hashes={k:{'x':ahash(v[0],True),'y':ahash(v[1],True),'shuffle':ahash(v[2],True)} for k,v in self.data.items()}
        write(self.worker/'data_fingerprints.json',{'contract':self.chash,'arrays':self.data_hashes})

    def model(self,method):
        seed_all(self.seed)
        neuron={'doppler':'Doppler-LIF','ei_stable':'SHDFollowupStable'}[method]
        m=self.Net(700,20,[neuron]*4,128,{},'none')
        g=torch.Generator().manual_seed(2000000+self.seed)
        for module in m.modules():
            if isinstance(module,torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(module.weight,a=math.sqrt(5),generator=g)
                torch.nn.init.uniform_(module.bias,-1/math.sqrt(module.in_features),1/math.sqrt(module.in_features),generator=g)
        return m.to('cuda')

    def thresholds(self,m):return {n:float(v.threshold) for n,v in m.named_modules() if isinstance(v,self.BaseNeuron)}
    def restore(self,m,ck):
        m.load_state_dict(ck['model'],strict=True)
        require(set(ck['thresholds'])==set(self.thresholds(m)),'Threshold keys differ')
        for n,t in ck['thresholds'].items():m.get_submodule(n).threshold=float(t)

    def initial_model(self,method):
        if method not in self.initial:
            m=self.model(method);before=rng_hash()
            with torch.no_grad():self.core.calibrate_thresholds(m,self.data['train'][0][:128].cuda(),.15)
            require(rng_hash()==before,'Calibration consumed RNG')
            self.initial[method]={'model':copy.deepcopy(m.state_dict()),'thresholds':self.thresholds(m)}
            require(len(self.initial[method]['thresholds'])==4 and all(math.isfinite(v) and v>0 for v in self.thresholds(m).values()),'Calibration invalid')
            p=self.worker/f'initial_{method}.pt'
            if p.exists():
                old=load(p)
                require(statehash(old['model'])==statehash(self.initial[method]['model']) and old['thresholds']==self.thresholds(m),'Initial state changed across invocation')
            else:save(p,self.initial[method])
            del m
        m=self.model(method);self.restore(m,self.initial[method]);return m

    def transformed(self,split,idx,condition):
        x,y,p=self.data[split];v=x[idx].cuda();labels=y[idx].cuda()
        if condition['shuffle']:
            inds=p[idx].cuda().unsqueeze(-1).expand(-1,-1,-1,700)
            v=v.reshape(-1,12,4,700).gather(2,inds).reshape(-1,48,700)
        r,n=condition['resolution'],condition['updates']
        if r==12:v=v.reshape(-1,12,4,700).sum(2)
        if n!=r:
            z=v.new_zeros((len(v),n,700));z[:,n//r-1::n//r,:]=v;v=z
        return v,labels

    def infer(self,m,condition,split,limit=None,return_logits=False):
        m.eval();preds=[];logits=[];n=len(self.data[split][1]) if limit is None else limit
        with torch.no_grad():
            for start in range(0,n,256):
                x,y=self.transformed(split,torch.arange(start,min(start+256,n)),condition)
                out=m(x);require(bool(torch.isfinite(out).all()),'Nonfinite logits')
                preds.extend(out.argmax(1).cpu().tolist())
                if return_logits:logits.append(out.cpu())
        truth=self.data[split][1][:n].tolist()
        return (preds,truth,torch.cat(logits)) if return_logits else (preds,truth)

    def step(self,m,opt,condition,idx):
        m.train();x,y=self.transformed('train',idx,condition)
        opt.zero_grad(set_to_none=True);loss=F.cross_entropy(m(x),y)
        require(bool(torch.isfinite(loss)),'Nonfinite training loss');loss.backward()
        require(all(bool(torch.isfinite(p.grad).all()) for p in m.parameters() if p.grad is not None),'Nonfinite gradient')
        opt.step();return float(loss.detach())

    def preflight(self):
        started=time.perf_counter();checks={};conditions=self.contract['conditions'];idx=torch.arange(8)
        native=self.data['train'][0][idx].cuda();views={c['name']:self.transformed('train',idx,c)[0] for c in conditions}
        require(torch.equal(views['fine48_N48'],native),'Baseline transformation changed')
        for c in conditions:require(torch.equal(views[c['name']].sum(1),native.sum(1)),'Event count lost')
        for r in ('fine48','coarse12'):
            require(torch.equal(views[r+'_N96'][:,1::2,:],views[r+'_N48']),'Nonzero input values/times differ with update count')
            require(torch.count_nonzero(views[r+'_N96'][:,::2,:])==0,'Extra updates have external input')
        a=native.reshape(-1,12,4,700);b=views['shuffled48_N48'].reshape(-1,12,4,700)
        require(torch.equal(a.sum(2),b.sum(2)) and torch.equal(a.sort(2).values,b.sort(2).values),'Shuffle changes counts/amplitudes')
        require(not torch.equal(a,b),'Shuffle is identity on probe')
        checks['event_mass_injection_times_amplitude_shuffle']=True
        timings=[];affines=[]
        for method in self.contract['models']:
            m=self.initial_model(method)
            affines.append(statehash({n:p for n,p in m.state_dict().items() if 'projs.' in n or n.startswith('readout.')}))
            # Repeated calibration must reproduce exactly without changing parameters.
            repeated=self.model(method)
            with torch.no_grad():self.core.calibrate_thresholds(repeated,self.data['train'][0][:128].cuda(),.15)
            require(self.thresholds(repeated)==self.thresholds(m) and statehash(repeated.state_dict())==statehash(m.state_dict()),'Calibration repeat differs')
            # N48 wrapper must be exactly the original forward, loss and gradients.
            m.zero_grad(set_to_none=True);repeated.zero_grad(set_to_none=True)
            z=m(native);zr=self.core.VerticalNet.forward(repeated,native)
            require(torch.equal(z,zr),'N48 original forward differs')
            z.square().sum().backward();zr.square().sum().backward()
            for (n,p),(nr,pr) in zip(m.named_parameters(),repeated.named_parameters()):
                require(n==nr and ((p.grad is None and pr.grad is None) or (p.grad is not None and pr.grad is not None and torch.equal(p.grad,pr.grad))),'N48 original gradient differs')
            with torch.no_grad():
                m(native.flip(1));require(torch.equal(m(native),z.detach()),'State leaked between batches')
            del m,repeated,z,zr
            for c in conditions:
                m=self.initial_model(method);opt=torch.optim.Adam(m.parameters(),lr=.001)
                order=permutation(self.seed,0,len(self.data['train'][1]))
                # Warmup and short benchmark on disposable models only.
                self.step(m,opt,c,order[:128]);torch.cuda.synchronize();t=time.perf_counter()
                self.step(m,opt,c,order[128:256]);torch.cuda.synchronize()
                timings.append({'model':method,'condition':c['name'],'seconds_per_128_train_batch':time.perf_counter()-t})
                del m,opt
            # One update + disk checkpoint + second update equals continuous two updates.
            c=conditions[2];order=permutation(self.seed,0,len(self.data['train'][1]));m=self.initial_model(method)
            opt=torch.optim.Adam(m.parameters(),lr=.001);self.step(m,opt,c,order[:128])
            cp={'model':copy.deepcopy(m.state_dict()),'thresholds':self.thresholds(m),'optimizer':copy.deepcopy(opt.state_dict()),'rng':rng_state()}
            p=self.worker/f'diagnostic_resume_{method}.pt';save(p,cp)
            loss=self.step(m,opt,c,order[128:256]);final=statehash(m.state_dict());rng_end=rng_hash()
            restored=self.model(method);saved=load(p);self.restore(restored,saved)
            oo=torch.optim.Adam(restored.parameters(),lr=.001);oo.load_state_dict(saved['optimizer']);restore_rng(saved['rng'])
            loss2=self.step(restored,oo,c,order[128:256])
            require(loss==loss2 and statehash(restored.state_dict())==final and rng_hash()==rng_end,'Resume chain differs')
            _,_,l1=self.infer(restored,c,'val',limit=32,return_logits=True)
            _,_,l2=self.infer(restored,c,'val',limit=32,return_logits=True)
            require(torch.equal(l1,l2),'GPU inference replay differs')
            del m,opt,restored,oo,cp,saved
        require(len(set(affines))==1,'Cross-model affine initialization differs')
        # Permutation covers every sample once and is independent of global RNG consumption.
        order=permutation(self.seed,3,6932);torch.rand(711);random.random();np.random.rand(19)
        require(torch.equal(order,permutation(self.seed,3,6932)) and torch.equal(order.sort().values,torch.arange(6932)),'Sampler invariant failed')
        checks.update(original_N48_forward_and_gradients_exact=True,calibration_repeat_exact=True,
                      calibration_preserves_rng=True,common_affines=True,state_reset_between_batches=True,
                      private_full_epoch_permutation=True,checkpoint_resume_exact=True,gpu_replay_exact=True)
        report={'status':'PASSED','scientific_training_started':False,'diagnostic_optimizer_steps':26,
                'contract_sha256':self.chash,'runner_sha256':self.runner_sha,'environment':self.env,
                'checks':checks,'data_hashes':self.data_hashes,'timings':timings,
                'elapsed_seconds':time.perf_counter()-started,'finished_utc':utc()}
        write(self.worker/'preflight.json',report)
        print('[PREFLIGHT_PASSED]',json.dumps(report),flush=True)

    def run_one(self,method,c):
        job=f"{method}_{c['name']}_s{self.seed}";out=self.worker/job;out.mkdir(exist_ok=True)
        if (out/'result.json').exists():
            r=read(out/'result.json');require(r['status']=='COMPLETED' and r['contract_sha256']==self.chash and sha(out/'selected.pt')==r['selected_sha256'],'Invalid completed job')
            print('[ALREADY_COMPLETED]',job,flush=True);return r
        m=self.initial_model(method);opt=torch.optim.Adam(m.parameters(),lr=.001)
        initial_hash=statehash(m.state_dict());initial_thresholds=self.thresholds(m)
        start=0;history=[];orders=[];best=-1.;selected=-1;patience=0;elapsed_prior=0.;epoch_seconds=[]
        if (out/'last.pt').exists():
            ck=load(out/'last.pt');require(ck['contract_sha256']==self.chash and ck['job']==job,'Resume contract differs')
            self.restore(m,ck);opt.load_state_dict(ck['optimizer']);restore_rng(ck['rng'])
            start=ck['epoch']+1;history=ck['history'];orders=ck['orders'];best=ck['best'];selected=ck['selected'];patience=ck['patience']
            elapsed_prior=ck['elapsed'];epoch_seconds=ck['epoch_seconds']
        started=time.perf_counter();n=len(self.data['train'][1])
        for epoch in range(start,self.cfg['max_epochs']):
            if patience>=self.cfg['patience']:break
            t=time.perf_counter();order=permutation(self.seed,epoch,n);orders.append(ahash(order))
            for off in range(0,n,128):self.step(m,opt,c,order[off:off+128])
            vp,vt=self.infer(m,c,'val');val=float(f1_score(vt,vp,labels=np.arange(20),average='macro',zero_division=0))
            history.append(val);smooth=float(np.mean(history[-5:]));improved=smooth>best+1e-4
            if improved:
                best,selected,patience=smooth,epoch,0
                save(out/'selected.pt',{'job':job,'model':copy.deepcopy(m.state_dict()),'thresholds':self.thresholds(m),
                     'epoch':epoch,'validation_preds':vp,'validation_f1':val,'contract_sha256':self.chash})
            else:patience+=1
            epoch_seconds.append(time.perf_counter()-t)
            elapsed=elapsed_prior+time.perf_counter()-started
            save(out/'last.pt',{'job':job,'contract_sha256':self.chash,'epoch':epoch,
                 'model':copy.deepcopy(m.state_dict()),'thresholds':self.thresholds(m),'optimizer':opt.state_dict(),'rng':rng_state(),
                 'history':history,'orders':orders,'best':best,'selected':selected,'patience':patience,
                 'elapsed':elapsed,'epoch_seconds':epoch_seconds})
            write(out/'progress.json',{'status':'RUNNING','job':job,'epoch':epoch+1,'selected_epoch':selected+1,
                 'patience':patience,'val_raw':val,'val_smoothed':smooth,'elapsed_seconds':elapsed,
                 'epoch_seconds':epoch_seconds[-10:],'updated_utc':utc()})
            print(f'[EPOCH] {job} {epoch+1}/200 val={val:.5f} selected={selected+1} patience={patience} sec={epoch_seconds[-1]:.2f}',flush=True)
        ck=load(out/'selected.pt');self.restore(m,ck)
        vp,vt=self.infer(m,c,'val');require(vp==ck['validation_preds'],'Selected validation does not replay')
        tp,tt,logits=self.infer(m,c,'test',return_logits=True)
        replay=self.model(method);self.restore(replay,load(out/'selected.pt'))
        tp2,tt2,logits2=self.infer(replay,c,'test',return_logits=True)
        require(tp==tp2 and tt==tt2 and torch.equal(logits,logits2),'Selected test checkpoint does not replay exactly')
        r={'status':'COMPLETED','job':job,'model':method,'condition':c,'seed':self.seed,
           'contract_sha256':self.chash,'runner_sha256':self.runner_sha,'environment':self.env,
           'initial_state_sha256':initial_hash,'initial_thresholds':initial_thresholds,
           'epochs':len(history),'selected_epoch_zero_based':selected,'validation_history':history,
           'validation_selected_preds':vp,'minibatch_permutation_sha256':orders,'epoch_seconds':epoch_seconds,
           'test_macro_f1':float(f1_score(tt,tp,labels=np.arange(20),average='macro',zero_division=0)),
           'test_accuracy':float(accuracy_score(tt,tp)),'trues':tt,'preds':tp,
           'selected_sha256':sha(out/'selected.pt'),'selected_logits_sha256':ahash(logits,True),
           'selected_replay_exact':True,'data_hashes':self.data_hashes,
           'elapsed_seconds':elapsed_prior+time.perf_counter()-started,'completed_utc':utc()}
        write(out/'result.json',r);write(out/'progress.json',{'status':'COMPLETED','job':job,'epochs':len(history),'updated_utc':utc()})
        print('[DONE]',job,'F1',r['test_macro_f1'],'epochs',len(history),flush=True)
        del m,opt,replay,ck,logits,logits2;torch.cuda.empty_cache();return r

    def run(self):
        gate=read(self.worker/'preflight.json')
        require(gate['status']=='PASSED' and gate['contract_sha256']==self.chash and gate['runner_sha256']==self.runner_sha and gate['environment']==self.env,'Preflight required')
        require(gate['data_hashes']==self.data_hashes,'Data changed since preflight')
        rows=[]
        for method in self.contract['models']:
            for c in self.contract['conditions']:rows.append(self.run_one(method,c))
        require(len(rows)==10,'Wrong job count')
        for method in self.contract['models']:
            rr=[r for r in rows if r['model']==method]
            require(len({r['initial_state_sha256'] for r in rr})==1 and all(r['initial_thresholds']==rr[0]['initial_thresholds'] for r in rr),'Condition initialization/calibration confound')
        for epoch in range(max(r['epochs'] for r in rows)):
            require(len({r['minibatch_permutation_sha256'][epoch] for r in rows if r['epochs']>epoch})==1,'Minibatch order differs across cells/models')
        write(self.worker/'complete.json',{'status':'COMPLETED','seed':self.seed,'jobs':[r['job'] for r in rows],
              'contract_sha256':self.chash,'all_replays_exact':True,'minibatch_chains_paired':True,'completed_utc':utc()})
        print('[SEED_COMPLETE]',self.seed,'10/10',flush=True)


def run(args, jobs):
    study = Study(args)
    gate = study.worker / 'preflight.json'
    if not gate.exists():
        study.preflight()
    study.run()
