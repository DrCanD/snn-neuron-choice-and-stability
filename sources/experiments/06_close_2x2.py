# Adapted from neuron-task-matching 96c2c484: local real-data execution; plots/setup omitted.
if __name__ != '__main__':
    raise RuntimeError('Use run.py train; importing an experiment must not start training.')
import mstask
from mstask.core import FeatureStack, VerticalNet, _make_norm, accuracy, calibrate_thresholds, count_params, macro_f1
import os, json, math, copy, warnings
from pathlib import Path
from datetime import datetime
warnings.filterwarnings('ignore')
IN_COLAB = True
ON_COLAB = True

class CFG:
    NEURONS = ["Doppler-LIF", "CrossInhib-LIF", "Phase-LIF"]   # Doppler re-run for paired bootstrap
    L            = 4
    W            = 128
    TARGET_FIRING = 0.15
    NORM         = "none"
    MAX_EPOCHS   = 200
    PATIENCE     = 20
    VAL_SMOOTH   = 5
    VAL_FRAC     = 0.15
    LR           = 1e-3
    BATCH        = 128
    SELECT_METRIC = "macro_f1"
    SEEDS        = [42, 123, 999]
    USE_TQDM     = True
    HEARTBEAT_EVERY = 5
    PROJECT = os.environ.get("RESEARCH_ROOT", "./data") + "/NISAC_DeepHetero"
cfg = CFG()
DIAT_HOMOG = {"Doppler-LIF": 0.8719, "Dual-tau-LIF": 0.8598, "Chirp-LIF": 0.6971, "STFT-IF": 0.1850}
SHD_HOMOG = {"CrossInhib-LIF": 0.7405, "Vanilla-LIF": 0.7030, "Dual-tau-LIF": 0.6943,
             "Chirp-LIF": 0.6624, "Beam-IF": 0.6385, "Doppler-LIF": 0.6005,
             "STFT-IF": 0.2439, "Phase-LIF": 0.1430}
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import dikmen_neurons as D
from dikmen_neurons import BaseNeuron, spike_hard, NeuronRegistry
try:
    from tqdm.auto import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False
device = "cuda" if torch.cuda.is_available() else "cpu"
RUN_ID  = os.environ.get("RUN_ID", "diat_2x2")
RUN_DIR = Path(cfg.PROJECT) / "dissoc" / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = RUN_DIR / "ckpt"; CKPT_DIR.mkdir(exist_ok=True)
CKPT_DIR.mkdir(exist_ok=True)
print(f'[HW] device={device}' + (f'  gpu={torch.cuda.get_device_name(0)}' if device == 'cuda' else ''), flush=True)
print(f'[RUN] {RUN_DIR}', flush=True)

class VanillaLIF(BaseNeuron):
    _family = "lif"
    def single_step(self, x_t, state):
        mem = self.beta * state["mem"] + x_t
        spk = spike_hard(mem, self.threshold)
        return spk, {"mem": mem * (1.0 - spk)}
NeuronRegistry._all["Vanilla-LIF"] = VanillaLIF
DIAT_PATH = os.environ.get("DIAT_DATA", "/content/drive/MyDrive/NISAC/data/DIAT_uSAT/processed")

def _diat_xy():
    X = np.load(Path(DIAT_PATH) / 'X.npy').astype(np.float32)
    y = np.load(Path(DIAT_PATH) / 'y.npy').astype(np.int64)
    return (X, y)

def load_diat():
    X, y = _diat_xy()
    assert X.ndim == 3, f"expected [N,64,64], got {X.shape}"
    def strat_split(Xa, ya, frac, seed):
        try:
            from sklearn.model_selection import train_test_split
            A, B, ya2, yb2 = train_test_split(Xa, ya, test_size=frac, stratify=ya, random_state=seed)
            return A, ya2, B, yb2
        except Exception:
            rng = np.random.default_rng(seed); ia, ib = [], []
            for c in np.unique(ya):
                ci = np.where(ya == c)[0]; rng.shuffle(ci); k = int((1 - frac) * len(ci))
                ia += list(ci[:k]); ib += list(ci[k:])
            return Xa[ia], ya[ia], Xa[ib], ya[ib]
    Xtv, ytv, Xte, yte = strat_split(X, y, 0.20, 0)
    Xtr, ytr, Xva, yva = strat_split(Xtv, ytv, cfg.VAL_FRAC / 0.80, 0)
    mu, sd = float(Xtr.mean()), float(Xtr.std() + 1e-6)
    norm = lambda a: torch.as_tensor((a - mu) / sd).float()
    yy   = lambda a: torch.as_tensor(a).long()
    return norm(Xtr), yy(ytr), norm(Xva), yy(yva), norm(Xte), yy(yte)
Xtr, ytr, Xva, yva, Xte, yte = load_diat()
C = int(max(ytr.max(), yte.max()) + 1); Fin = Xtr.shape[-1]; Tlen = Xtr.shape[1]
C = int(max(ytr.max(), yte.max()) + 1); Fin = Xtr.shape[-1]; Tlen = Xtr.shape[1]
C = int(max(ytr.max(), yte.max()) + 1); Fin = Xtr.shape[-1]; Tlen = Xtr.shape[1]
print(f'[DATA] DIAT train={tuple(Xtr.shape)} val={tuple(Xva.shape)} test={tuple(Xte.shape)} T={Tlen} F={Fin} C={C} chance={1 / C:.3f}', flush=True)
METRICS = {"macro_f1": lambda p,t,C: macro_f1(p,t,C), "accuracy": lambda p,t,C: accuracy(p,t)}

@torch.no_grad()
def evaluate(model, X, Y, batch):
    model.eval(); preds = []
    for i in range(0, len(X), batch):
        preds.append(model(X[i:i+batch].to(device)).argmax(1).cpu())
    return torch.cat(preds).numpy(), Y.numpy()

def firing_mean(model, xb): return float(np.mean(model.firing_rates(xb)))

def paired_bootstrap(pa, pb, trues, mfn, C, nboot=2000, seed=0):
    rng = np.random.default_rng(seed); n = len(trues); boots = np.empty(nboot)
    base = mfn(pa, trues, C) - mfn(pb, trues, C)
    for k in range(nboot):
        i = rng.integers(0, n, n)
        boots[k] = mfn(pa[i], trues[i], C) - mfn(pb[i], trues[i], C)
    return float(base), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

def train_one(neuron, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    fire_batch = Xtr[:256].to(device)
    model = VerticalNet(Fin, C, [neuron]*cfg.L, cfg.W, {}, cfg.NORM).to(device)
    calibrate_thresholds(model, fire_batch, cfg.TARGET_FIRING)
    n_params = count_params(model)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.LR)
    sel = METRICS[cfg.SELECT_METRIC]

    ckpt = CKPT_DIR / f"{neuron.replace('/','_')}_s{seed}.pt"
    start, m = 0, {}
    if ckpt.exists():
        ck = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); start, m = ck["epoch"]+1, ck["m"]
    val_hist = m.get("val_hist", []); best_val = m.get("best_val", -1.0)
    best_epoch = m.get("best_epoch", -1); pc = m.get("pc", 0); best = m.get("best", None)

    use_bar = cfg.USE_TQDM and _HAS_TQDM
    it = tqdm(range(start, cfg.MAX_EPOCHS), desc=f"{neuron} s{seed}", leave=False,
              initial=start, total=cfg.MAX_EPOCHS) if use_bar else range(start, cfg.MAX_EPOCHS)
    for ep in it:
        model.train(); perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), cfg.BATCH):
            b = perm[i:i+cfg.BATCH]
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(model(Xtr[b].to(device)), ytr[b].to(device)).backward(); opt.step()
        vp, vt = evaluate(model, Xva, yva, cfg.BATCH); val_hist.append(sel(vp, vt, C))
        v_smooth = float(np.mean(val_hist[-cfg.VAL_SMOOTH:]))
        tp, tt = evaluate(model, Xte, yte, cfg.BATCH)
        t_f1 = macro_f1(tp, tt, C); t_acc = accuracy(tp, tt)
        improved = v_smooth > best_val + 1e-4
        if improved:
            best_val, best_epoch, pc = v_smooth, ep, 0
            best = {"f1": t_f1, "acc": t_acc, "firing": firing_mean(model, fire_batch),
                    "preds": tp.tolist(), "trues": tt.tolist()}
        else:
            pc += 1
        torch.save({"epoch": ep, "model": copy.deepcopy(model.state_dict()), "opt": opt.state_dict(),
                    "m": {"val_hist": val_hist, "best_val": best_val, "best_epoch": best_epoch,
                          "pc": pc, "best": best}}, ckpt)
        cur = best["firing"] if best else 0.0
        if use_bar:
            it.set_postfix(val=f"{v_smooth:.3f}", f1=f"{t_f1:.3f}", fire=f"{cur:.2f}", pat=pc)
        elif (ep % cfg.HEARTBEAT_EVERY == 0) or improved:
            print(f"      ep{ep:>3}/{cfg.MAX_EPOCHS} val={v_smooth:.4f} f1={t_f1:.4f} "
                  f"fire={cur:.3f} pat={pc}{'  *' if improved else ''}", flush=True)
        if pc >= cfg.PATIENCE: break
    if use_bar: it.close()
    best["params"] = n_params; best["best_epoch"] = best_epoch
    return best
RES_PATH = RUN_DIR / "results.json"
res = json.loads(RES_PATH.read_text()) if RES_PATH.exists() else {}
print('\n' + '═' * 78 + '\n DIAT per-neuron bake-off (homogeneous, matched to ablation)\n' + '═' * 78, flush=True)
for neuron in cfg.NEURONS:
    for seed in cfg.SEEDS:
        key = f'{neuron}:{seed}'
        if key in res:
            continue
        print(f'\n[{neuron} seed {seed}]', flush=True)
        r = train_one(neuron, seed)
        res[key] = r
        RES_PATH.write_text(json.dumps(res))
        print(f'   -> {key}: F1={r['f1']:.4f} acc={r['acc']:.4f} fire={r['firing']:.3f} @ep{r['best_epoch']} params={r['params']}', flush=True)

def agg(neuron):
    ks = [f"{neuron}:{s}" for s in cfg.SEEDS if f"{neuron}:{s}" in res]
    return ks, [res[k]["f1"] for k in ks], [res[k]["acc"] for k in ks], [res[k]["firing"] for k in ks]
print('\n' + '═' * 78 + '\n DIAT RANKING  (macro-F1, mean±std over seeds)\n' + '═' * 78, flush=True)
diat_now = {}
for neuron in cfg.NEURONS:
    ks, f1, ac, fr = agg(neuron)
    if not ks:
        continue
    diat_now[neuron] = float(np.mean(f1))
    print(f'   {neuron:16s} F1 {np.mean(f1):.4f}±{np.std(f1):.4f}  acc {np.mean(ac):.4f}  fire {np.mean(fr):.3f}', flush=True)
dop_repro = diat_now.get("Doppler-LIF")
if dop_repro is not None:
    print(f'\n[CHECK] Doppler-LIF DIAT reproduction: {dop_repro:.4f}  (locked ablation 0.8719, Δ={dop_repro - 0.8719:+.4f})', flush=True)
print('\n' + '═' * 78 + '\n DOUBLE DISSOCIATION 2×2  (winner of each domain)\n' + '═' * 78, flush=True)
print(f'   {'neuron':16s}   DIAT-F1    SHD-F1', flush=True)
for n in ['Doppler-LIF', 'CrossInhib-LIF']:
    d = diat_now.get(n, DIAT_HOMOG.get(n))
    s = SHD_HOMOG.get(n)
    print(f'   {n:16s}   {d:.4f}    {s:.4f}', flush=True)
ciP = diat_now.get("Phase-LIF")
if ciP is not None:
    print(f'   {'Phase-LIF':16s}   {ciP:.4f}    {SHD_HOMOG['Phase-LIF']:.4f}', flush=True)
dopD, ciD = diat_now.get("Doppler-LIF"), diat_now.get("CrossInhib-LIF")
ciS, dopS = SHD_HOMOG["CrossInhib-LIF"], SHD_HOMOG["Doppler-LIF"]
square = (dopD is not None and ciD is not None and dopD > ciD and ciS > dopS)
print(f'\n   DIAT: Doppler {dopD:.4f} vs CrossInhib {ciD:.4f}  -> {('Doppler wins' if dopD > ciD else 'CrossInhib wins')}', flush=True)
print(f'   SHD : CrossInhib {ciS:.4f} vs Doppler {dopS:.4f}  -> {('CrossInhib wins' if ciS > dopS else 'Doppler wins')}', flush=True)
print(f'   2×2 DOUBLE DISSOCIATION {('CONFIRMED' if square else 'NOT confirmed')} (each domain won by its physics-matched neuron)', flush=True)
print('\n[BOOTSTRAP] paired on DIAT test, per seed (consistent = same sign all seeds)', flush=True)
boot = {}
for a, b in [('Doppler-LIF', 'CrossInhib-LIF'), ('Doppler-LIF', 'Phase-LIF')]:
    if not (any((f'{a}:{s}' in res for s in cfg.SEEDS)) and any((f'{b}:{s}' in res for s in cfg.SEEDS))):
        continue
    for label, mfn in [('F1', METRICS['macro_f1']), ('acc', METRICS['accuracy'])]:
        per = []
        for s in cfg.SEEDS:
            ka, kb = (f'{a}:{s}', f'{b}:{s}')
            if ka in res and kb in res:
                per.append(paired_bootstrap(np.array(res[ka]['preds']), np.array(res[kb]['preds']), np.array(res[ka]['trues']), mfn, C))
        if not per:
            continue
        mean = float(np.mean([x[0] for x in per]))
        cis = [(round(l, 3), round(h, 3)) for _, l, h in per]
        consistent = all((l > 0 for _, l, _ in per)) or all((h < 0 for _, _, h in per))
        boot[f'{a}-{b}[{label}]'] = {'mean': mean, 'cis': cis, 'consistent': consistent}
        print(f'   {a} - {b} [{label}]: mean {mean:+.4f}  CIs {cis}  consistent={consistent}', flush=True)
summary = {"run_dir": str(RUN_DIR), "diat_now": diat_now, "doppler_reproduction": dop_repro,
           "diat_homog_locked": DIAT_HOMOG, "shd_homog": SHD_HOMOG,
           "square_confirmed": bool(square), "bootstrap": boot, "timestamp": datetime.now().isoformat()}
(RUN_DIR / 'summary.json').write_text(json.dumps(summary, indent=2))
print(f'\n[DONE] summary -> {RUN_DIR / 'summary.json'}', flush=True)
