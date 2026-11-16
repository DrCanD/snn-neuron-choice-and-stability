# Adapted from neuron-task-matching 96c2c484: local real-data execution; plots/setup omitted.
if __name__ != '__main__':
    raise RuntimeError('Use run.py train; importing an experiment must not start training.')
import mstask
from mstask.core import FeatureStack, VerticalNet, _calibrate_stack, _make_norm, _strat_split, accuracy, calibrate_thresholds, count_params, macro_f1
import os, json, math, copy, time, warnings
from pathlib import Path
from datetime import datetime
warnings.filterwarnings('ignore')
IN_COLAB = True
ON_COLAB = True

class HPO:
    W_CHOICES = [96, 128, 192, 256, 384]
    L_CHOICES = [3, 4, 5, 6]
    BETA_CHOICES = [0.85, 0.9, 0.95, 0.97, 0.99]
    LR_LOW, LR_HIGH = (0.0003, 0.004)
    SCHED_CHOICES = ['none', 'cosine']
    SURR_CHOICES = [10.0, 25.0, 50.0]
    FIRE_CHOICES = [0.1, 0.15, 0.2]
    INPUT_CHOICES = ['plain', 'log1p']
    WEIGHT_DECAY = 0.0
    BATCH = 128
    MAX_EPOCHS = 200
    PATIENCE = 20
    VAL_SMOOTH = 5
    VAL_FRAC = 0.15
    SELECT_METRIC = 'macro_f1'
    N_TRIALS = 55
    HPO_SEED = 42
    REEVAL_SEEDS = [42, 123, 999]
    ARCHS = ['doppler', 'd2', 'lif']
    USE_TQDM = True
    HEARTBEAT_EVERY = 5
    PRUNE_EPOCH = 30
    PRUNE_FLOOR = 0.3
    PROJECT = os.environ.get('RESEARCH_ROOT', './data') + '/NISAC_DeepHetero'
    DATA_DIR = os.environ.get('DIAT_DATA', './data/DIAT_uSAT/processed')
cfg = HPO()
cfg.ARCHS    = ["doppler", "lif", "plif"]
cfg.N_TRIALS = 30
cfg.REEVAL_SEEDS = [42, 123, 999, 7, 2024]   # 5-seed headline; doppler/lif resume (3 cached + 2 new)
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import optuna
try:
    from tqdm.auto import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False
import dikmen_neurons as D
from dikmen_neurons import BaseNeuron, spike_hard, NeuronRegistry

def _locate_surrogate():
    c = getattr(D, "FastSigmoidSurrogate", None)
    if c is not None:
        return c
    sh = getattr(D, "spike_hard", None)
    if sh is not None and hasattr(sh, "__globals__"):
        c = sh.__globals__.get("FastSigmoidSurrogate")
        if c is not None:
            return c
    try:
        import pkgutil, importlib
        for _, name, _ in pkgutil.walk_packages(D.__path__, D.__name__ + "."):
            m = importlib.import_module(name)
            if hasattr(m, "FastSigmoidSurrogate"):
                return getattr(m, "FastSigmoidSurrogate")
    except Exception:
        pass
    return None
SURROGATE = _locate_surrogate()
if SURROGATE is None:
    print('[WARN] FastSigmoidSurrogate not found; surrogate-slope tuning disabled (package default).')
device = "cuda" if torch.cuda.is_available() else "cpu"
Path(cfg.PROJECT).mkdir(parents=True, exist_ok=True)
RUN_ID  = os.environ.get("RUN_ID", "hpo_main")   # STABLE across re-runs -> resume; change for a new experiment
RUN_DIR = Path(cfg.PROJECT) / "hpo" / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = RUN_DIR / "ckpt"; CKPT_DIR.mkdir(exist_ok=True)
CKPT_DIR.mkdir(exist_ok=True)
STORAGE  = f"sqlite:///{RUN_DIR / 'studies.db'}"
print(f'[HW] device={device}' + (f'  gpu={torch.cuda.get_device_name(0)}' if device == 'cuda' else ''))
print(f'[RUN] {RUN_DIR}', flush=True)

class VanillaLIF(BaseNeuron):
    _family = "lif"
    _description = "Standard leaky integrate-and-fire baseline (no resonance/freq-selectivity)."
    def single_step(self, x_t, state):
        mem = self.beta * state["mem"] + x_t
        spk = spike_hard(mem, self.threshold)
        mem = mem * (1.0 - spk)
        return spk, {"mem": mem}
NeuronRegistry._all["Vanilla-LIF"] = VanillaLIF

class ParametricLIF(BaseNeuron):
    _family = "plif"
    _description = "Parametric LIF: learnable per-neuron membrane decay (no resonance)."
    def __init__(self, size, beta=0.95, threshold=1.0):
        super().__init__(size, beta=beta, threshold=threshold)
        b = min(max(float(beta), 1e-3), 1 - 1e-3)
        self.beta_logit = nn.Parameter(torch.full((size,), math.log(b / (1 - b))))
    def single_step(self, x_t, state):
        beta = torch.sigmoid(self.beta_logit)
        mem = beta * state["mem"] + x_t
        spk = spike_hard(mem, self.threshold)
        mem = mem * (1.0 - spk)
        return spk, {"mem": mem}
NeuronRegistry._all["Parametric-LIF"] = ParametricLIF
D2_PATHS = ["Doppler-LIF", "Chirp-LIF", "STFT-IF", "Dual-tau-LIF", "CrossInhib-LIF"]

class PathBankNet(nn.Module):
    def __init__(self, in_features, n_classes, path_types, width, n_layers,
                 fusion="concat", neuron_kwargs=None, norm="none", shared_stem=True):
        super().__init__()
        if shared_stem:
            self.stem = nn.Linear(in_features, width); path_in = width
        else:
            self.stem = None; path_in = in_features
        self.paths = nn.ModuleList(
            [FeatureStack(path_in, [t]*n_layers, width, neuron_kwargs, norm) for t in path_types])
        self.fusion = fusion; self.width = width
        feat_dim = width * len(path_types)
        self.readout = nn.Linear(feat_dim, n_classes)
    def forward(self, x):
        if self.stem is not None:
            B, T, d = x.shape; x = self.stem(x.reshape(B*T, d)).reshape(B, T, -1)
        feats = [p(x)[0] for p in self.paths]
        return self.readout(torch.cat(feats, dim=-1))
    def path_firing(self, x):
        if self.stem is not None:
            B, T, d = x.shape; x = self.stem(x.reshape(B*T, d)).reshape(B, T, -1)
        return [p(x)[1].float().mean().item() for p in self.paths]

def load_data():
    base = Path(cfg.DATA_DIR)
    X = np.load(base / "X.npy").astype(np.float32)     # [N,64,64]=[N,time,Doppler]
    y = np.load(base / "y.npy").astype(np.int64)
    assert X.ndim == 3, f"expected [N,64,64], got {X.shape}"
    Xtv, ytv, Xte, yte = _strat_split(X, y, 0.20, 0)
    Xtr, ytr, Xva, yva = _strat_split(Xtv, ytv, cfg.VAL_FRAC/0.80, 0)
    out = {}
    for tname in cfg.INPUT_CHOICES:
        if tname == "log1p":
            f = lambda a: np.log1p(np.clip(a, 0.0, None))
            tr, va, te = f(Xtr), f(Xva), f(Xte)
        else:
            tr, va, te = Xtr, Xva, Xte
        mu, sd = float(tr.mean()), float(tr.std() + 1e-6)
        nz = lambda a: torch.as_tensor((a - mu)/sd).float()
        out[tname] = (nz(tr), torch.as_tensor(ytr).long(),
                      nz(va), torch.as_tensor(yva).long(),
                      nz(te), torch.as_tensor(yte).long())
    C = int(y.max() + 1); Fin = X.shape[-1]
    return out, C, Fin
DATA, C, Fin = load_data()
n_tr = len(DATA[cfg.INPUT_CHOICES[0]][0]); n_va = len(DATA[cfg.INPUT_CHOICES[0]][2]); n_te = len(DATA[cfg.INPUT_CHOICES[0]][4])
n_tr = len(DATA[cfg.INPUT_CHOICES[0]][0]); n_va = len(DATA[cfg.INPUT_CHOICES[0]][2]); n_te = len(DATA[cfg.INPUT_CHOICES[0]][4])
n_tr = len(DATA[cfg.INPUT_CHOICES[0]][0]); n_va = len(DATA[cfg.INPUT_CHOICES[0]][2]); n_te = len(DATA[cfg.INPUT_CHOICES[0]][4])
print(f'[DATA] train={n_tr} val={n_va} test={n_te}  C={C}  F={Fin}  chance={1 / C:.3f}', flush=True)
METRICS = {"accuracy": lambda p, t, C: accuracy(p, t),
           "macro_f1": lambda p, t, C: macro_f1(p, t, C)[0]}

@torch.no_grad()
def evaluate(model, X, Y, batch):
    model.eval(); preds = []
    for i in range(0, len(X), batch):
        preds.append(model(X[i:i+batch].to(device)).argmax(1).cpu())
    return torch.cat(preds).numpy(), Y.numpy()

def firing_mean(model, xb):
    fr = model.firing_rates(xb) if isinstance(model, VerticalNet) else model.path_firing(xb)
    return float(np.mean(fr))

def paired_bootstrap(pa, pb, trues, mfn, C, nboot=2000, seed=0):
    rng = np.random.default_rng(seed); n = len(trues)
    base = mfn(pa, trues, C) - mfn(pb, trues, C); boots = np.empty(nboot)
    for k in range(nboot):
        i = rng.integers(0, n, n)
        boots[k] = mfn(pa[i], trues[i], C) - mfn(pb[i], trues[i], C)   # paired resample
    return float(base), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

def build_model(arch, W, L, beta):
    if arch == "doppler":
        nk = {"Doppler-LIF": {"beta": beta}}
        return VerticalNet(Fin, C, ["Doppler-LIF"]*L, W, nk, "none")
    if arch == "lif":
        nk = {"Vanilla-LIF": {"beta": beta}}
        return VerticalNet(Fin, C, ["Vanilla-LIF"]*L, W, nk, "none")
    if arch == "plif":
        nk = {"Parametric-LIF": {"beta": beta}}
        return VerticalNet(Fin, C, ["Parametric-LIF"]*L, W, nk, "none")
    if arch == "d2":
        nk = {t: {"beta": beta} for t in D2_PATHS}
        return PathBankNet(Fin, C, D2_PATHS, W, L, "concat", nk, "none", True)
    raise KeyError(arch)

def make_sched(opt, schedule, max_epochs, warmup=5):
    if schedule != "cosine": return None
    def fn(ep):
        if ep < warmup: return (ep + 1) / warmup
        prog = (ep - warmup) / max(1, max_epochs - warmup)
        return 0.5 * (1 + math.cos(math.pi * prog))
    return torch.optim.lr_scheduler.LambdaLR(opt, fn)

def train_eval(arch, p, seed, ckpt_tag, trial=None, verbose=False):
    """Returns dict: f1, acc, firing, params, best_epoch, preds, trues (test @ best smoothed-val)."""
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if SURROGATE is not None:
        SURROGATE.scale = p["surr"]                               # global surrogate slope

    Xtr, ytr, Xva, yva, Xte, yte = DATA[p["input"]]
    fire_batch = Xtr[:256].to(device)
    model = build_model(arch, p["W"], p["L"], p["beta"]).to(device)
    calibrate_thresholds(model, fire_batch, p["fire"])            # config A: init only
    n_params = count_params(model)
    opt = torch.optim.Adam(model.parameters(), lr=p["lr"], weight_decay=cfg.WEIGHT_DECAY)
    sched = make_sched(opt, p["sched"], cfg.MAX_EPOCHS)

    ckpt = CKPT_DIR / f"{ckpt_tag}.pt"
    start, m = 0, {}
    if ckpt.exists():
        ck = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        if sched is not None and ck.get("sched"): sched.load_state_dict(ck["sched"])
        start, m = ck["epoch"] + 1, ck["m"]

    sel = METRICS[cfg.SELECT_METRIC]
    val_hist = m.get("val_hist", []); best_val = m.get("best_val", -1.0)
    best_epoch = m.get("best_epoch", -1); pc = m.get("pc", 0); best = m.get("best", None)
    batch = cfg.BATCH

    use_bar = cfg.USE_TQDM and _HAS_TQDM
    epochs = range(start, cfg.MAX_EPOCHS)
    bar = tqdm(epochs, desc=ckpt_tag, leave=False, dynamic_ncols=True,
               initial=start, total=cfg.MAX_EPOCHS) if use_bar else epochs
    for ep in bar:
        model.train(); perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), batch):
            b = perm[i:i+batch]
            xb, yb = Xtr[b].to(device), ytr[b].to(device)
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(model(xb), yb).backward(); opt.step()
        if sched is not None: sched.step()

        vp, vt = evaluate(model, Xva, yva, batch); v_raw = sel(vp, vt, C)
        val_hist.append(v_raw)
        v_smooth = float(np.mean(val_hist[-cfg.VAL_SMOOTH:]))
        tp, tt = evaluate(model, Xte, yte, batch)
        t_acc = accuracy(tp, tt); t_f1, t_f1c = macro_f1(tp, tt, C)
        improved = v_smooth > best_val + 1e-4
        if improved:
            best_val, best_epoch, pc = v_smooth, ep, 0
            best = {"acc": t_acc, "f1": t_f1, "f1_per_class": t_f1c,
                    "firing": firing_mean(model, fire_batch),
                    "preds": tp.tolist(), "trues": tt.tolist()}
        else:
            pc += 1
        torch.save({"epoch": ep, "model": copy.deepcopy(model.state_dict()), "opt": opt.state_dict(),
                    "sched": sched.state_dict() if sched is not None else None,
                    "m": {"val_hist": val_hist, "best_val": best_val, "best_epoch": best_epoch,
                          "pc": pc, "best": best}}, ckpt)
        cur_fire = best["firing"] if best else 0.0
        if use_bar:
            bar.set_postfix(val=f"{v_smooth:.3f}", f1=f"{t_f1:.3f}", fire=f"{cur_fire:.2f}", pat=pc)
        elif (ep % cfg.HEARTBEAT_EVERY == 0) or improved:
            print(f"      ep{ep:>3}/{cfg.MAX_EPOCHS}  val={v_smooth:.4f}  f1={t_f1:.4f}  "
                  f"fire={cur_fire:.3f}  pat={pc}{'  *best' if improved else ''}", flush=True)
        # conservative absolute-floor pruning (HPO only; safe vs late convergence)
        if trial is not None and ep == cfg.PRUNE_EPOCH and v_smooth < cfg.PRUNE_FLOOR:
            if use_bar: bar.close()
            raise optuna.TrialPruned()
        if pc >= cfg.PATIENCE:
            break
    if use_bar:
        bar.close()
    best["params"] = n_params; best["best_epoch"] = best_epoch
    return best

def make_objective(arch):
    def objective(trial):
        p = dict(
            W     = trial.suggest_categorical("W", cfg.W_CHOICES),
            L     = trial.suggest_categorical("L", cfg.L_CHOICES),
            beta  = trial.suggest_categorical("beta", cfg.BETA_CHOICES),
            lr    = trial.suggest_float("lr", cfg.LR_LOW, cfg.LR_HIGH, log=True),
            sched = trial.suggest_categorical("sched", cfg.SCHED_CHOICES),
            surr  = trial.suggest_categorical("surr", cfg.SURR_CHOICES),
            fire  = trial.suggest_categorical("fire", cfg.FIRE_CHOICES),
            input = trial.suggest_categorical("input", cfg.INPUT_CHOICES),
        )
        tag = f"{arch}_t{trial.number}_s{cfg.HPO_SEED}"
        print(f"\n[{arch} trial {trial.number}/{cfg.N_TRIALS}] W={p['W']} L={p['L']} beta={p['beta']} "
              f"lr={p['lr']:.1e} {p['sched']} surr={p['surr']} fire={p['fire']} {p['input']}", flush=True)
        r = train_eval(arch, p, cfg.HPO_SEED, tag, trial=trial)
        print(f"   -> trial {trial.number} done: F1={r['f1']:.4f}  acc={r['acc']:.4f}  "
              f"fire={r['firing']:.3f}  @ep{r['best_epoch']}  params={r['params']}", flush=True)
        trial.set_user_attr("params", r["params"])
        trial.set_user_attr("best_epoch", r["best_epoch"])
        trial.set_user_attr("acc", r["acc"])
        return r["f1"], r["firing"]            # (maximize, minimize)
    return objective

def run_study(arch):
    study = optuna.create_study(
        study_name=f"hpo_{arch}", storage=STORAGE, load_if_exists=True,
        directions=["maximize", "minimize"],
        sampler=optuna.samplers.TPESampler(multivariate=True, seed=cfg.HPO_SEED))
    done = len([t for t in study.trials if t.state.name in ("COMPLETE", "PRUNED")])
    todo = max(0, cfg.N_TRIALS - done)
    print(f"\n[STUDY {arch}] {done}/{cfg.N_TRIALS} done, running {todo} more", flush=True)
    if todo:
        study.optimize(make_objective(arch), n_trials=todo, show_progress_bar=False)
    return study

def select_configs(study):
    """From the Pareto front pick (best_f1) and a firing-efficient (high-F1, min-firing) config."""
    pf = [t for t in study.best_trials]
    if not pf:
        comp = [t for t in study.trials if t.state.name == "COMPLETE"]
        pf = sorted(comp, key=lambda t: t.values[0], reverse=True)[:1]
    best_f1 = max(pf, key=lambda t: t.values[0])
    f1max = best_f1.values[0]
    near = [t for t in pf if t.values[0] >= f1max - 0.01]
    efficient = min(near, key=lambda t: t.values[1])
    out = {"best_f1": best_f1.params}
    if efficient.number != best_f1.number:
        out["efficient"] = efficient.params
    return out, pf
REEVAL_PATH = RUN_DIR / "reeval.json"
reeval = json.loads(REEVAL_PATH.read_text()) if REEVAL_PATH.exists() else {}
studies, pareto = {}, {}
for arch in cfg.ARCHS:
    studies[arch] = run_study(arch)
print('\n' + '═' * 78 + '\n PARETO FRONTS  (F1 ↑, firing ↓, params)\n' + '═' * 78)
selected = {}
for arch in cfg.ARCHS:
    cfgs, pf = select_configs(studies[arch])
    selected[arch] = cfgs
    pareto[arch] = []
    print(f'\n[{arch}]  ({len(pf)} Pareto points)')
    for t in sorted(pf, key=lambda x: x.values[0], reverse=True):
        pr = t.user_attrs.get('params', -1)
        pareto[arch].append({'f1': t.values[0], 'firing': t.values[1], 'params': pr, 'params_cfg': t.params})
        print(f'   F1={t.values[0]:.4f}  fire={t.values[1]:.3f}  params={pr:>7}  W={t.params['W']} L={t.params['L']} beta={t.params['beta']} lr={t.params['lr']:.1e} sched={t.params['sched']} surr={t.params['surr']} fire={t.params['fire']} in={t.params['input']}')
print('\n' + '═' * 78 + '\n RE-EVAL selected configs on seeds ' + str(cfg.REEVAL_SEEDS) + '\n' + '═' * 78)
for arch in cfg.ARCHS:
    for cname, params in selected[arch].items():
        for seed in cfg.REEVAL_SEEDS:
            key = f'{arch}:{cname}:{seed}'
            if key in reeval:
                continue
            p = dict(W=params['W'], L=params['L'], beta=params['beta'], lr=params['lr'], sched=params['sched'], surr=params['surr'], fire=params['fire'], input=params['input'])
            tag = f'reeval_{arch}_{cname}_s{seed}'
            print(f'\n[re-eval {arch}:{cname} seed {seed}] W={params['W']} L={params['L']} beta={params['beta']} {params['sched']} fire={params['fire']}', flush=True)
            r = train_eval(arch, p, seed, tag)
            reeval[key] = {'f1': r['f1'], 'acc': r['acc'], 'firing': r['firing'], 'params': r['params'], 'best_epoch': r['best_epoch'], 'preds': r['preds'], 'trues': r['trues']}
            REEVAL_PATH.write_text(json.dumps(reeval))
            print(f'   {key:<34} F1={r['f1']:.4f} acc={r['acc']:.4f} fire={r['firing']:.3f} params={r['params']}', flush=True)

def agg(arch, cname):
    ks = [f"{arch}:{cname}:{s}" for s in cfg.REEVAL_SEEDS if f"{arch}:{cname}:{s}" in reeval]
    f1s = [reeval[k]["f1"] for k in ks]; firs = [reeval[k]["firing"] for k in ks]
    accs = [reeval[k]["acc"] for k in ks]; pr = reeval[ks[0]]["params"]
    return ks, f1s, firs, accs, pr
print('\n' + '═' * 78 + '\n FINAL  (best_f1 config, mean±std over re-eval seeds)\n' + '═' * 78)
final = {}
for arch in cfg.ARCHS:
    ks, f1s, firs, accs, pr = agg(arch, 'best_f1')
    final[arch] = {'f1_mean': float(np.mean(f1s)), 'f1_std': float(np.std(f1s)), 'acc_mean': float(np.mean(accs)), 'firing_mean': float(np.mean(firs)), 'params': pr, 'keys': ks}
    print(f'   {arch:<9} F1 {np.mean(f1s):.4f}±{np.std(f1s):.4f}  acc {np.mean(accs):.4f}  fire {np.mean(firs):.3f}  params {pr}')
print('\n[BOOTSTRAP] paired on test, per re-eval seed (consistent = same sign all seeds)')
boot = {}
pairs = [("doppler", "lif"), ("doppler", "plif"), ("plif", "lif"), ("d2", "lif"), ("doppler", "d2")]
for a, b in pairs:
    if a not in cfg.ARCHS or b not in cfg.ARCHS:
        continue
    for label, mfn in [('F1', METRICS['macro_f1']), ('acc', METRICS['accuracy'])]:
        per_seed = []
        for s in cfg.REEVAL_SEEDS:
            ka, kb = (f'{a}:best_f1:{s}', f'{b}:best_f1:{s}')
            if ka in reeval and kb in reeval:
                pa = np.array(reeval[ka]['preds'])
                pb = np.array(reeval[kb]['preds'])
                tr = np.array(reeval[ka]['trues'])
                per_seed.append(paired_bootstrap(pa, pb, tr, mfn, C))
        if not per_seed:
            continue
        mean = float(np.mean([x[0] for x in per_seed]))
        cis = [(round(lo, 3), round(hi, 3)) for _, lo, hi in per_seed]
        consistent = all((lo > 0 for _, lo, _ in per_seed)) or all((hi < 0 for _, _, hi in per_seed))
        boot[f'{a}-{b}[{label}]'] = {'mean': mean, 'cis': cis, 'consistent': consistent}
        print(f'   {a} - {b} [{label}]: mean {mean:+.4f}  CIs {cis}  consistent={consistent}')

class SmallCNN(nn.Module):
    def __init__(self, n_classes, ch=(16, 32, 64)):
        super().__init__()
        layers, c0 = [], 1
        for c in ch:
            layers += [nn.Conv2d(c0, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(), nn.MaxPool2d(2)]
            c0 = c
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(ch[-1], n_classes))
    def forward(self, x):                       # x: [N, 64, 64] -> 1-channel image
        return self.head(self.features(x.unsqueeze(1)))

def count_macs_cnn(model):
    macs = [0]
    def ch(m, i, o):
        oh, ow = o.shape[-2], o.shape[-1]; kh, kw = m.kernel_size
        macs[0] += oh*ow*m.out_channels*m.in_channels*kh*kw
    def lh(m, i, o):
        macs[0] += m.in_features*m.out_features
    hs = []
    for mod in model.modules():
        if isinstance(mod, nn.Conv2d): hs.append(mod.register_forward_hook(ch))
        elif isinstance(mod, nn.Linear): hs.append(mod.register_forward_hook(lh))
    model.eval()
    with torch.no_grad(): model(torch.zeros(1, Fin, Fin, device=device))
    for h in hs: h.remove()
    return macs[0]

def cnn_train_eval(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    Xtr, ytr, Xva, yva, Xte, yte = DATA["plain"]
    model = SmallCNN(C).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.MAX_EPOCHS)
    best_val, best, pc, val_hist = -1.0, None, 0, []
    use_bar = cfg.USE_TQDM and _HAS_TQDM
    it = tqdm(range(cfg.MAX_EPOCHS), desc=f"cnn_s{seed}", leave=False) if use_bar else range(cfg.MAX_EPOCHS)
    for ep in it:
        model.train(); perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), cfg.BATCH):
            b = perm[i:i+cfg.BATCH]; opt.zero_grad(set_to_none=True)
            F.cross_entropy(model(Xtr[b].to(device)), ytr[b].to(device)).backward(); opt.step()
        sched.step()
        vp, vt = evaluate(model, Xva, yva, cfg.BATCH); v = macro_f1(vp, vt, C)[0]; val_hist.append(v)
        vs = float(np.mean(val_hist[-cfg.VAL_SMOOTH:]))
        tp, tt = evaluate(model, Xte, yte, cfg.BATCH); tf1, _ = macro_f1(tp, tt, C); ta = accuracy(tp, tt)
        if use_bar: it.set_postfix(val=f"{vs:.3f}", f1=f"{tf1:.3f}", pat=pc)
        if vs > best_val + 1e-4:
            best_val, pc = vs, 0
            best = {"f1": tf1, "acc": ta, "preds": tp.tolist(), "trues": tt.tolist()}
        else:
            pc += 1
        if pc >= cfg.PATIENCE: break
    if use_bar: it.close()
    best["params"] = count_params(model); best["macs"] = count_macs_cnn(model)
    return best
print('\n' + '═' * 78 + '\n CNN BASELINE on the same split (seeds ' + str(cfg.REEVAL_SEEDS) + ')\n' + '═' * 78)
CNN_PATH = RUN_DIR / "cnn.json"
cnn = json.loads(CNN_PATH.read_text()) if CNN_PATH.exists() else {}
for s in cfg.REEVAL_SEEDS:
    k = f'cnn:{s}'
    if k in cnn:
        continue
    print(f'\n[cnn seed {s}]', flush=True)
    r = cnn_train_eval(s)
    cnn[k] = r
    CNN_PATH.write_text(json.dumps(cnn))
    print(f'   cnn:{s}  F1={r['f1']:.4f} acc={r['acc']:.4f} params={r['params']} macs={r['macs']}', flush=True)
cnn_f1 = [cnn[f"cnn:{s}"]["f1"] for s in cfg.REEVAL_SEEDS if f"cnn:{s}" in cnn]
cnn_acc = [cnn[f"cnn:{s}"]["acc"] for s in cfg.REEVAL_SEEDS if f"cnn:{s}" in cnn]
cnn_params = cnn[f"cnn:{cfg.REEVAL_SEEDS[0]}"]["params"]; cnn_macs = cnn[f"cnn:{cfg.REEVAL_SEEDS[0]}"]["macs"]
cnn_params = cnn[f"cnn:{cfg.REEVAL_SEEDS[0]}"]["params"]; cnn_macs = cnn[f"cnn:{cfg.REEVAL_SEEDS[0]}"]["macs"]
print(f'\n   CNN  F1 {np.mean(cnn_f1):.4f}±{np.std(cnn_f1):.4f}  acc {np.mean(cnn_acc):.4f}  params {cnn_params}  macs {cnn_macs / 1000000.0:.2f}M', flush=True)
print('\n[BOOTSTRAP vs CNN] (informational; CNN is a different, dense model class)')
for a in [x for x in cfg.ARCHS if x in ('doppler', 'lif', 'plif')]:
    for label, mfn in [('F1', METRICS['macro_f1']), ('acc', METRICS['accuracy'])]:
        per_seed = []
        for s in cfg.REEVAL_SEEDS:
            ka, kc = (f'{a}:best_f1:{s}', f'cnn:{s}')
            if ka in reeval and kc in cnn:
                pa = np.array(reeval[ka]['preds'])
                pc_ = np.array(cnn[kc]['preds'])
                tr = np.array(reeval[ka]['trues'])
                per_seed.append(paired_bootstrap(pa, pc_, tr, mfn, C))
        if not per_seed:
            continue
        mean = float(np.mean([x[0] for x in per_seed]))
        cis = [(round(lo, 3), round(hi, 3)) for _, lo, hi in per_seed]
        consistent = all((lo > 0 for _, lo, _ in per_seed)) or all((hi < 0 for _, _, hi in per_seed))
        boot[f'{a}-cnn[{label}]'] = {'mean': mean, 'cis': cis, 'consistent': consistent}
        print(f'   {a} - cnn [{label}]: mean {mean:+.4f}  CIs {cis}  consistent={consistent}')
E_MAC, E_AC, T = 4.6e-12, 0.9e-12, Fin

def snn_energy(W, L, firing):
    macs = Fin*W*T + W*C                        # layer-1 dense + readout
    syn  = (L-1)*(firing*W*W*T)                 # spike-driven hidden layers
    return macs, syn, macs*E_MAC + syn*E_AC
print('\n' + '═' * 78 + '\n ENERGY PROXY  (per inference; 45nm estimate, relative only)\n' + '═' * 78)
energy = {}
for a in [x for x in cfg.ARCHS if x in ('doppler', 'lif', 'plif')]:
    ca = selected[a]['best_f1']
    W, L = (ca['W'], ca['L'])
    fr = final[a]['firing_mean']
    macs, syn, E = snn_energy(W, L, fr)
    energy[a] = {'macs': macs, 'synops': syn, 'nJ': E * 1000000000.0, 'W': W, 'L': L, 'firing': fr}
    print(f'   {a:<9} W={W} L={L} fire={fr:.3f}  MAC={macs / 1000000.0:.2f}M  SynOps={syn / 1000000.0:.2f}M  ~{E * 1000000000.0:.1f} nJ')
cnn_E = cnn_macs*E_MAC
energy["cnn"] = {"macs": cnn_macs, "synops": 0, "nJ": cnn_E*1e9}
print(f'   {'cnn':<9} (dense)        MAC={cnn_macs / 1000000.0:.2f}M  SynOps=0.00M       ~{cnn_E * 1000000000.0:.1f} nJ')
if 'doppler' in energy:
    print(f'\n   energy ratio  CNN / Doppler = {cnn_E / (energy['doppler']['nJ'] * 1e-09):.1f}x  (Doppler {energy['doppler']['nJ']:.1f} nJ vs CNN {cnn_E * 1000000000.0:.1f} nJ)')
summary = {"run_dir": str(RUN_DIR), "archs": cfg.ARCHS, "n_trials": cfg.N_TRIALS,
           "reeval_seeds": cfg.REEVAL_SEEDS, "search_space": {
               "W": cfg.W_CHOICES, "L": cfg.L_CHOICES, "beta": cfg.BETA_CHOICES,
               "lr": [cfg.LR_LOW, cfg.LR_HIGH], "sched": cfg.SCHED_CHOICES, "surr": cfg.SURR_CHOICES,
               "fire": cfg.FIRE_CHOICES, "input": cfg.INPUT_CHOICES},
           "selected": selected, "pareto": pareto, "final": final, "bootstrap": boot,
           "cnn": {"f1_mean": float(np.mean(cnn_f1)), "f1_std": float(np.std(cnn_f1)),
                   "acc_mean": float(np.mean(cnn_acc)), "params": cnn_params, "macs": cnn_macs},
           "energy": energy,
           "timestamp": datetime.now().isoformat()}
(RUN_DIR / 'summary.json').write_text(json.dumps(summary, indent=2))
print(f'\n[DONE] summary -> {RUN_DIR / 'summary.json'}')
