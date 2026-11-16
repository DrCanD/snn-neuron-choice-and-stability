# Adapted from neuron-task-matching 96c2c484: local real-data execution; plots/setup omitted.
if __name__ != '__main__':
    raise RuntimeError('Use run.py train; importing an experiment must not start training.')
import mstask
from mstask.core import FeatureStack, VerticalNet, _calibrate_stack, _make_norm, _strat_split, accuracy, calibrate_thresholds, firing_snapshot, load_registry, macro_f1, register_dataset, spearman
import os, sys, json, gzip, shutil, time, copy, subprocess, warnings
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
warnings.filterwarnings('ignore')
IN_COLAB = True
ON_COLAB = True

@dataclass
class Config:
    # architecture (LOCKED — identical to the ablation/dissociation harness)
    L: int = 4
    W: int = 128
    norm: str = "none"                 # PURE SNN: no normalization anywhere
    target_firing: float = 0.15        # init threshold calibration target
    stft_window: int = 8               # SHD T=12 -> STFT window must be <= T

    # SHD binning
    T: int = 12                        # frames (matches the paper's cached SHD)
    F: int = 700                       # cochlear channels
    duration_s: float = 1.0            # bin spike times over [0, duration] into T frames

    # training / selection (LOCKED)
    max_epochs: int = 200
    patience: int = 20                 # early stop on smoothed-val plateau
    val_smooth: int = 5                # mean over last 5 epochs (paper's wording)
    val_frac: float = 0.15             # val carved from official train (selection ONLY)
    select_metric: str = "macro_f1"
    lr: float = 1e-3
    batch_size: int = 128
    seeds: list = field(default_factory=lambda: [42, 123, 999])
    nboot: int = 2000

    # the 8 neurons of Table 4 (Vanilla-LIF registered below; rest are MS-IF)
    methods: list = field(default_factory=lambda: [
        "CrossInhib-LIF", "Vanilla-LIF", "Dual-tau-LIF", "Chirp-LIF",
        "Beam-IF", "Doppler-LIF", "STFT-IF", "Phase-LIF"])

    # locked speaker-MIXED ranking (Table 4) — reference for the Spearman check.
    # (values are the paper's reported macro-F1; used ONLY for rank correlation)
    mixed_ranking: dict = field(default_factory=lambda: {
        "CrossInhib-LIF": 0.741, "Vanilla-LIF": 0.703, "Dual-tau-LIF": 0.694,
        "Chirp-LIF": 0.662, "Beam-IF": 0.639, "Doppler-LIF": 0.601,
        "STFT-IF": 0.244, "Phase-LIF": 0.143})

    # experiment manifest (single source of truth)
    exp_index: int = 2
    exp_total: int = 3
    exp_name: str = "SHD speaker-disjoint per-neuron ranking"
    exp_manifest: list = field(default_factory=lambda: [
        (1, "DIAT ablation + dissociation (mixed-split) [paper]", "done"),
        (2, "SHD speaker-DISJOINT ranking re-run", "THIS"),
        (3, "parameter-matched CNN baseline", "pending"),
    ])
cfg = Config()
RESEARCH = Path(os.environ['RESEARCH_ROOT'])
RESEARCH.mkdir(parents=True, exist_ok=True)
PROJECT = RESEARCH / "NISAC_DeepHetero"
PROJECT.mkdir(parents=True, exist_ok=True)
CACHE = Path(os.environ['STUDY_SHD_DATA'])
CACHE.mkdir(parents=True, exist_ok=True)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import dikmen_neurons as D
from dikmen_neurons import BaseNeuron, spike_hard, NeuronRegistry
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

class VanillaLIF(BaseNeuron):
    _family = "lif"
    _description = "Standard leaky integrate-and-fire baseline (control)."
    def single_step(self, x_t, state):
        mem = self.beta * state["mem"] + x_t
        spk = spike_hard(mem, self.threshold)
        mem = mem * (1.0 - spk)
        return spk, {"mem": mem}
NeuronRegistry._all["Vanilla-LIF"] = VanillaLIF
REGISTRY_PATH = RESEARCH / "datasets.json"

def fast_download(url, dest, desc='file'):
    raise RuntimeError('Provide the official decompressed SHD files via --data.')
SHD_URLS = {
    "train": ["https://zenkelab.org/datasets/shd_train.h5.gz",
              "https://compneuro.net/datasets/shd_train.h5.gz"],
    "test":  ["https://zenkelab.org/datasets/shd_test.h5.gz",
              "https://compneuro.net/datasets/shd_test.h5.gz"],
}

def _download_shd(split):
    p = CACHE / f'shd_{split}.h5'
    if not p.is_file():
        raise FileNotFoundError(p)
    return p

def _bin_h5(h5_path, T, F, duration):
    """SHD HDF5 -> [N, T, F] spike-count frames. Keys: spikes/times, spikes/units, labels."""
    import h5py
    with h5py.File(h5_path, "r") as f:
        times = f["spikes"]["times"]
        units = f["spikes"]["units"]
        labels = np.asarray(f["labels"], dtype=np.int64)
        N = len(labels)
        X = np.zeros((N, T, F), dtype=np.float32)
        for i in range(N):
            t = np.asarray(times[i], dtype=np.float32)
            u = np.asarray(units[i], dtype=np.int64)
            if t.size == 0:
                continue
            b = np.clip((t / duration * T).astype(np.int64), 0, T - 1)
            u = np.clip(u, 0, F - 1)
            np.add.at(X, (np.full(b.shape, i), b, u), 1.0)
    return X, labels

def load_shd_disjoint():
    """Returns train/val/test tensors. test = OFFICIAL disjoint test; val carved from train."""
    tr_h5, te_h5 = (_download_shd('train'), _download_shd('test'))
    Xtr_full, ytr_full = _bin_h5(tr_h5, cfg.T, cfg.F, cfg.duration_s)
    Xte, yte = _bin_h5(te_h5, cfg.T, cfg.F, cfg.duration_s)
    Xtr, ytr, Xva, yva = _strat_split(Xtr_full, ytr_full, cfg.val_frac, seed=0)
    register_dataset('shd_official', path=str(CACHE), source='zenkelab.org', format='hdf5', split='speaker-disjoint', classes=int(max(ytr_full.max(), yte.max()) + 1), T=cfg.T, features=cfg.F, notes='official train/test; 2 of 12 speakers test-only; binned to T frames')
    to_t = lambda a: torch.as_tensor(a).float()
    to_y = lambda a: torch.as_tensor(a).long()
    return (to_t(Xtr), to_y(ytr), to_t(Xva), to_y(yva), to_t(Xte), to_y(yte))

def _make_synth_h5(split):
    raise RuntimeError('Synthetic data is not a study input.')
METRICS = {"accuracy": lambda p, t, C: accuracy(p, t),
           "macro_f1": lambda p, t, C: macro_f1(p, t, C)[0]}

@torch.no_grad()
def evaluate(model, loader):
    model.eval(); preds, trues = [], []
    for xb, yb in loader:
        preds.append(model(xb.to(device)).argmax(1).cpu()); trues.append(yb)
    return torch.cat(preds).numpy(), torch.cat(trues).numpy()

def paired_bootstrap(preds_a, preds_b, trues, metric_fn, C, nboot=2000, seed=0):
    """Paired bootstrap over test examples; resamples preds AND trues by the same indices."""
    rng = np.random.default_rng(seed); n = len(trues)
    base = metric_fn(preds_a, trues, C) - metric_fn(preds_b, trues, C)
    boots = np.empty(nboot)
    for k in range(nboot):
        i = rng.integers(0, n, n)
        boots[k] = metric_fn(preds_a[i], trues[i], C) - metric_fn(preds_b[i], trues[i], C)
    return float(base), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

def build_model(method, Fin, C, NK):
    return VerticalNet(Fin, C, [method] * cfg.L, cfg.W, NK, cfg.norm)

def save_ckpt(model, opt, epoch, metrics, path):
    torch.save({"epoch": epoch, "model_state_dict": copy.deepcopy(model.state_dict()),
                "optimizer_state_dict": opt.state_dict(), "metrics": metrics}, path)

def load_ckpt(model, opt, path):
    if not path.exists():
        return 0, {}
    ck = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"]); opt.load_state_dict(ck["optimizer_state_dict"])
    print(f"    [RESUME] from epoch {ck['epoch']}")
    return ck["epoch"] + 1, ck["metrics"]

def train_one(method, seed, Fin, C, NK, tr_loader, va_loader, te_loader, fire_batch, ckpt_dir):
    """Early stop on SMOOTHED validation; report TEST once at best-smoothed-val epoch."""
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = build_model(method, Fin, C, NK).to(device)
    calibrate_thresholds(model, fire_batch.to(device), cfg.target_firing)   # init only
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    ckpt = ckpt_dir / f"{method}_seed{seed}.pt"
    start, m = load_ckpt(model, opt, ckpt)
    sel = METRICS[cfg.select_metric]
    fire_log     = m.get("fire_log", [])
    val_hist     = m.get("val_hist", [])
    best_val     = m.get("best_val", -1.0)
    best_epoch   = m.get("best_epoch", -1)
    patience_ctr = m.get("patience_ctr", 0)
    best = m.get("best", None)
    for ep in range(start, cfg.max_epochs):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb)
            loss.backward(); opt.step()
        vp, vt = evaluate(model, va_loader); v_raw = sel(vp, vt, C)
        val_hist.append(v_raw)
        v_smooth = float(np.mean(val_hist[-cfg.val_smooth:]))
        tp, tt = evaluate(model, te_loader)
        t_acc = accuracy(tp, tt); t_f1, t_f1c = macro_f1(tp, tt, C)
        fr = firing_snapshot(model, fire_batch.to(device))
        improved = v_smooth > best_val + 1e-4
        if improved:
            best_val, best_epoch, patience_ctr = v_smooth, ep, 0
            best = {"acc": t_acc, "f1": t_f1, "f1_per_class": t_f1c,
                    "firing": [round(r, 4) for r in fr],
                    "preds": tp.tolist(), "trues": tt.tolist()}
        else:
            patience_ctr += 1
        fire_log.append({"epoch": ep, "val_smooth": round(v_smooth, 4), "val_raw": round(v_raw, 4),
                         "test_acc": round(t_acc, 4), "test_f1": round(t_f1, 4),
                         "firing": [round(r, 4) for r in fr]})
        save_ckpt(model, opt, ep, {"fire_log": fire_log, "val_hist": val_hist, "best_val": best_val,
                                   "best_epoch": best_epoch, "patience_ctr": patience_ctr,
                                   "best": best}, ckpt)
        star = " *" if improved else ""
        print(f"    ep {ep:>3} val={v_smooth:.4f}(raw {v_raw:.4f}) test_acc={t_acc:.4f} test_f1={t_f1:.4f}{star}")
        if patience_ctr >= cfg.patience:
            print(f"    [EARLY STOP] no smoothed-val gain {cfg.patience} ep; best @ep{best_epoch} "
                  f"-> test f1={best['f1']:.4f}")
            break
    return best, fire_log

def main():
    RUN_ID = os.environ.get('RUN_ID', 'main')
    RUN_DIR = PROJECT / 'runs_shd_disjoint' / RUN_ID
    CKPT_DIR = RUN_DIR / 'checkpoints'
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    print(f'[PERSIST] project: {PROJECT}')
    print(f'[HW] device={device}  gpu={(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')}')
    print(f'[PROGRESS] Experiment {cfg.exp_index}/{cfg.exp_total}: {cfg.exp_name}')
    for idx, name, status in cfg.exp_manifest:
        mark = '->' if status == 'THIS' else 'x' if status == 'done' else '  '
        print(f'   {mark} {idx}/{cfg.exp_total}: {name} [{status}]')
    Xtr, ytr, Xva, yva, Xte, yte = load_shd_disjoint()
    Fin = Xtr.shape[-1]
    C = int(max(ytr.max(), yva.max(), yte.max()) + 1)
    print(f'[DATA] SHD-disjoint train={tuple(Xtr.shape)} val={tuple(Xva.shape)} test={tuple(Xte.shape)}  C={C}')
    print(f'[DATA] spike-count range [{Xtr.min():.1f}, {Xtr.max():.1f}]  chance={1 / C:.3f}')
    NK = {'STFT-IF': {'window_len': cfg.stft_window}}
    fire_batch = Xtr[:cfg.batch_size]
    pin = device.type == 'cuda'
    tr_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=cfg.batch_size, shuffle=True, num_workers=2, pin_memory=pin, persistent_workers=pin)
    va_loader = DataLoader(TensorDataset(Xva, yva), batch_size=256, shuffle=False)
    te_loader = DataLoader(TensorDataset(Xte, yte), batch_size=256, shuffle=False)
    n_runs = len(cfg.methods) * len(cfg.seeds)
    print(f'[WORKLOAD] {len(cfg.methods)} neurons x {len(cfg.seeds)} seeds = {n_runs} runs x <= {cfg.max_epochs} ep')
    results = {}
    run_counter = 0
    first_t = None
    for method in cfg.methods:
        f1s, accs, preds_seeds, trues_seeds = ([], [], [], [])
        for seed in cfg.seeds:
            run_counter += 1
            print(f'\n  Run {run_counter}/{n_runs}: {method} | seed={seed}')
            t0 = time.time()
            best, flog = train_one(method, seed, Fin, C, NK, tr_loader, va_loader, te_loader, fire_batch, CKPT_DIR)
            dt = time.time() - t0
            if first_t is None:
                first_t = dt
                print(f'    [ETA] ~{first_t * (n_runs - 1) / 60:.0f} min for remaining {n_runs - 1} runs')
            f1s.append(best['f1'])
            accs.append(best['acc'])
            preds_seeds.append(best['preds'])
            trues_seeds.append(best['trues'])
            with open(RUN_DIR / f'results_{method}_seed{seed}.json', 'w') as f:
                json.dump({'method': method, 'seed': seed, 'f1': best['f1'], 'acc': best['acc'], 'f1_per_class': best['f1_per_class'], 'firing': best['firing'], 'fire_log': flog, 'time_s': dt}, f, indent=2)
        results[method] = {'f1_mean': float(np.mean(f1s)), 'f1_std': float(np.std(f1s)), 'acc_mean': float(np.mean(accs)), 'f1s': f1s, 'preds': preds_seeds, 'trues': trues_seeds}
        print(f'  => {method}: macro-F1 {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}')
    disjoint = {m: results[m]['f1_mean'] for m in cfg.methods}
    order_dis = sorted(cfg.methods, key=lambda m: disjoint[m], reverse=True)
    order_mix = sorted(cfg.methods, key=lambda m: cfg.mixed_ranking[m], reverse=True)
    rho = spearman(cfg.methods, cfg.mixed_ranking, disjoint)
    print('\n' + '=' * 74)
    print(' SHD per-neuron ranking: locked MIXED (Table 4)  vs  OFFICIAL DISJOINT')
    print('=' * 74)
    print(f'  {'neuron':<16}{'mixed F1':>10}{'mixed rk':>9}   {'disjoint F1':>12}{'disjoint rk':>12}')
    for m in order_mix:
        print(f'  {m:<16}{cfg.mixed_ranking[m]:>10.3f}{order_mix.index(m) + 1:>9}   {disjoint[m]:>12.4f}{order_dis.index(m) + 1:>12}')
    print(f'\n  Spearman rank correlation (mixed vs disjoint) = {rho:+.3f}')
    print(f'  mixed   order: {' > '.join(order_mix)}')
    print(f'  disjoint order: {' > '.join(order_dis)}')
    boot = None
    a, b = ('CrossInhib-LIF', 'Doppler-LIF')
    if a in results and b in results:
        per_seed = []
        for si in range(len(cfg.seeds)):
            pa = np.array(results[a]['preds'][si])
            pb = np.array(results[b]['preds'][si])
            tr = np.array(results[a]['trues'][si])
            per_seed.append(paired_bootstrap(pa, pb, tr, METRICS['macro_f1'], C, nboot=cfg.nboot, seed=cfg.seeds[si]))
        diffs = [d for d, _, _ in per_seed]
        consistent = all((lo > 0 for _, lo, _ in per_seed)) or all((hi < 0 for _, _, hi in per_seed))
        boot = {'pair': f'{a} - {b}', 'mean_diff': float(np.mean(diffs)), 'per_seed_ci': [(round(l, 3), round(h, 3)) for _, l, h in per_seed], 'consistent_sign': consistent}
        print(f'\n[BOOTSTRAP audio] {a} - {b}: mean macro-F1 diff {np.mean(diffs):+.4f}  per-seed CIs {boot['per_seed_ci']}  consistent_sign={consistent}')
        print('  (paper Table 5: CrossInhib beats Doppler on audio by ~0.140, consistent)')
    summary = {'run_id': RUN_ID, 'split': 'speaker-disjoint (official zenkelab.org)', 'config': {k: v for k, v in vars(cfg).items()}, 'results': {m: {'f1_mean': results[m]['f1_mean'], 'f1_std': results[m]['f1_std'], 'acc_mean': results[m]['acc_mean'], 'f1s': results[m]['f1s']} for m in cfg.methods}, 'disjoint_ranking': order_dis, 'mixed_ranking': order_mix, 'spearman_mixed_vs_disjoint': rho, 'crossinhib_vs_doppler_audio': boot, 'timestamp': datetime.now().isoformat()}
    for nm, base in [('summary.json', RUN_DIR), ('latest_shd_disjoint.json', PROJECT)]:
        with open(base / nm, 'w') as f:
            json.dump(summary, f, indent=2)
    print(f'\n[DONE] summary -> {RUN_DIR / 'summary.json'}')
    return summary
if __name__ == '__main__':
    summary = main()
