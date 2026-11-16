"""DIAT capacity, representation and E-I initialization experiments."""
import copy
import hashlib
import json
import math
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

from runtime import ROOT, definition, environment, read, sha, sources


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def array_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def evaluate(model, loader):
    preds, trues = [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            preds.extend(model(x.cuda()).argmax(1).cpu().tolist())
            trues.extend(y.tolist())
    return preds, trues


def build_model(core, registry, CrossInhibLIF, spec, cfg):
    class Stable(CrossInhibLIF):
        def __init__(self, size, beta=0.95, threshold=1.0):
            super().__init__(size, beta, threshold)
            with torch.no_grad():
                raw = math.log(math.expm1(spec['coupling'] if spec['coupling'] is not None else 0.20))
                self.w_inh.fill_(raw)
                self.w_exc.fill_(raw)
    registry._all['PipelineStableEI'] = Stable
    name, width = spec['model'], spec['width']
    nk = {'STFT-IF': {'window_len': cfg['stft_window_diat']}}
    if name.startswith('bank_'):
        paths = ['Doppler-LIF', 'Chirp-LIF', 'STFT-IF', 'Dual-tau-LIF',
                 'PipelineStableEI' if name == 'bank_stable' else 'CrossInhib-LIF']
        model = core.PathBankNet(64, 6, paths, {64: 26, 128: 51, 256: 100}[width], 4, 'concat', nk, 'none', True)
        expected = {64: 18180, 128: 61155, 256: 217906}[width]
    else:
        neuron = {'doppler': 'Doppler-LIF', 'vanilla': 'Vanilla-LIF', 'parametric': 'Parametric-LIF',
                  'ei': 'CrossInhib-LIF' if spec['coupling'] is None else 'PipelineStableEI'}[name]
        model = core.VerticalNet(64, 6, [neuron] * 4, width, nk, 'none')
        expected = {64: 17798, 128: 60166, 256: 218630}[width] if name == 'doppler' else None
    if expected is not None and core.count_params(model) != expected:
        raise RuntimeError('Parameter count differs from the recorded model.')
    return model


def run(args, jobs):
    contract = read(ROOT / 'configs/diat.json')
    env = environment(contract['environment'], contract['flags'])
    repo, lib, core = sources(args.sources)
    from dikmen_neurons import BaseNeuron, NeuronRegistry, spike_hard
    from dikmen_neurons.ms_if import CrossInhibLIF
    cfg = contract['config']
    scope = dict(np=np, torch=torch, nn=nn, F=F, math=math, BaseNeuron=BaseNeuron,
                 spike_hard=spike_hard, cfg=SimpleNamespace(**cfg),
                 get_dataset_path=lambda name: args.data)
    for filename, name in [('02_diat_ablation.py', 'load_diat'),
                           ('07_shd_disjoint.py', 'VanillaLIF'),
                           ('04_best_model_search_full.py', 'ParametricLIF')]:
        exec(compile(definition(repo / 'experiments' / filename, name), filename + ':' + name, 'exec'), scope)
    NeuronRegistry._all['Vanilla-LIF'] = scope['VanillaLIF']
    NeuronRegistry._all['Parametric-LIF'] = scope['ParametricLIF']
    data = scope['load_diat']()
    hashes = {key: array_hash(x) for key, x in zip(contract['split_sha256'], data)}
    if hashes != contract['split_sha256']:
        raise RuntimeError('DIAT split/normalization differs from the recorded inputs.')
    write(args.output / 'protocol.json', dict(contract=contract, environment=env, split_sha256=hashes))
    for spec in jobs:
        coupling = 'library' if spec['coupling'] is None else str(spec['coupling'])
        name = f"{spec['model']}_W{spec['width']}_{spec['view']}_T{spec['steps']}_{coupling}_s{spec['seed']}"
        folder = args.output / name
        result = folder / 'result.json'
        if result.exists():
            old = read(result)
            if old['spec'] != spec or sha(folder / 'selected.pt') != old['selected_sha256']:
                raise RuntimeError('Completed result/checkpoint differs.')
            print('[COMPLETED]', name, flush=True)
            continue
        # The recorded DIAT control campaigns restarted interrupted individual runs.
        # Require a new output directory rather than overwrite a partial attempt.
        folder.mkdir(exist_ok=False)
        tensors = list(data)
        for i in (0, 2, 4):
            x = data[i]
            if spec['view'] == 'difference':
                x = torch.cat((x[:, :1], x[:, 1:] - x[:, :-1]), dim=1)
            if spec['steps'] != 64:
                x = x.reshape(len(x), spec['steps'], 64 // spec['steps'], 64).mean(dim=2)
            tensors[i] = x
        xtr, ytr, xva, yva, xte, yte = tensors
        va = DataLoader(TensorDataset(xva.cpu(), yva.cpu().long()), batch_size=256, shuffle=False)
        te = DataLoader(TensorDataset(xte.cpu(), yte.cpu().long()), batch_size=256, shuffle=False)
        seed = spec['seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        tr = DataLoader(TensorDataset(xtr.cpu(), ytr.cpu().long()), batch_size=cfg['batch_size'],
                        shuffle=True, generator=torch.Generator().manual_seed(seed), num_workers=0)
        model = build_model(core, NeuronRegistry, CrossInhibLIF, spec, cfg).cuda()
        with torch.no_grad():
            core.calibrate_thresholds(model, xtr[:cfg['batch_size']].cuda(), cfg['target_firing'])
        opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
        best, best_epoch, patience, history = -1., -1, 0, []
        for epoch in range(cfg['max_epochs']):
            model.train()
            for xb, yb in tr:
                opt.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(xb.cuda()), yb.cuda())
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('Nonfinite loss')
                loss.backward()
                opt.step()
            vp, vt = evaluate(model, va)
            value = float(f1_score(vt, vp, labels=np.arange(6), average='macro', zero_division=0))
            history.append(value)
            smooth = float(np.mean(history[-cfg['val_smooth']:]))
            if smooth > best + 1e-4:
                best, best_epoch, patience = smooth, epoch, 0
                thresholds = {n: float(m.threshold) for n, m in model.named_modules() if isinstance(m, BaseNeuron)}
                temporary = folder / 'selected.tmp'
                torch.save(dict(model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                                thresholds=thresholds, epoch=epoch, spec=spec), temporary)
                temporary.replace(folder / 'selected.pt')
            else:
                patience += 1
            write(folder / 'progress.json', dict(epoch=epoch, selected_epoch=best_epoch, validation_history=history))
            print('[EPOCH]', name, epoch + 1, value, flush=True)
            if patience >= cfg['patience']:
                break
        checkpoint = torch.load(folder / 'selected.pt', map_location='cuda', weights_only=True)
        model.load_state_dict(checkpoint['model'], strict=True)
        for n, threshold in checkpoint['thresholds'].items():
            model.get_submodule(n).threshold = threshold
        tp, tt = evaluate(model, te)
        write(result, dict(status='COMPLETED', spec=spec, environment=env, epochs=epoch + 1,
                           selected_epoch_zero_based=best_epoch, validation_history=history,
                           validation_best_smoothed_f1=best,
                           test_macro_f1=float(f1_score(tt, tp, labels=np.arange(6), average='macro', zero_division=0)),
                           test_accuracy=float(accuracy_score(tt, tp)), preds=tp, trues=tt,
                           selected_sha256=sha(folder / 'selected.pt')))
        del model, opt, checkpoint
        torch.cuda.empty_cache()
