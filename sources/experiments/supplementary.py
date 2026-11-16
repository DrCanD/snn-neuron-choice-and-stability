# Legacy S1-S9 numerical definitions from the archived 2026-09 revision.
# Local paths replace Colab setup; analysis/presentation code is omitted.
import os, sys, json, math, time, copy, gzip, shutil, hashlib, subprocess, warnings, ast
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
import numpy as np
ON_COLAB = IN_COLAB = True
STAGES = ONLY_METHODS = None
RESEARCH = Path(os.environ['RESEARCH_ROOT'])
PROJECT = RESEARCH / 'NISAC_DeepHetero'
REV = PROJECT / 'revision_R1'
EXP_ROOT = REV / 'experiments'
CACHE = Path(os.environ['STUDY_SHD_DATA'])
DIAT_DATA_DEFAULT = os.environ['DIAT_DATA']
DIAT_CLASS_NAMES = ['3_long_blade_rotor', '3_short_blade_rotor', 'Bird', 'Bird+mini-helicopter', 'Quadcopter', 'RC_plane']
NTM_DIR = Path(__file__).resolve().parents[1]
NTM_SHA = '96c2c484c45114afed21c0a7e768d3cba9b3c723'
DSN_SHA = 'ebab00ac1a177ae08472db87ad5051f0d6ca7ebb'
DSN_EXPECTED = {'ms_if/__init__.py': 'b60adf5df5426197', 'base.py': '140dae1217293de3'}
def _sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]
DSN_EXPECTED = {'ms_if/__init__.py': 'b60adf5df5426197', 'base.py': '140dae1217293de3'}
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import mstask
from mstask.core import FeatureStack, PathBankNet, VerticalNet, _solve_width, _strat_split, accuracy, calibrate_thresholds, count_params, firing_snapshot, get_dataset_path, load_registry, macro_f1, register_dataset
import dikmen_neurons
from dikmen_neurons import BaseNeuron, NeuronRegistry, spike_hard
from dikmen_neurons.ms_if import CrossInhibLIF, DualTauLIF, STFTIF
assert mstask.core.NeuronRegistry is NeuronRegistry, 'registry object mismatch (import order)'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def extract_defs(py_path, names):
    src = Path(py_path).read_text(encoding='utf-8')
    lines = src.splitlines(keepends=True)

    def seg(node):
        start = min([d.lineno for d in getattr(node, 'decorator_list', [])] + [node.lineno])
        return ''.join(lines[start - 1:node.end_lineno])
    tree = ast.parse(src)
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            out[node.name] = seg(node)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in names:
                    out[t.id] = seg(node)
    missing = set(names) - set(out)
    assert not missing, f'[FIDELITY] {py_path.name}: missing {sorted(missing)}'
    return out
EXP = NTM_DIR / 'experiments'
REF02 = extract_defs(EXP / '02_diat_ablation.py', ['Config', 'load_diat', 'save_ckpt', 'load_ckpt', 'evaluate', 'METRICS', 'train_one', 'paired_bootstrap', 'H1_ORDER', 'D2_PATHS'])
REF07 = extract_defs(EXP / '07_shd_disjoint.py', ['Config', 'VanillaLIF', 'fast_download', 'SHD_URLS', '_download_shd', '_bin_h5', 'load_shd_disjoint', '_make_synth_h5'])
REF04 = extract_defs(EXP / '04_best_model_search_full.py', ['ParametricLIF'])
_ns02, _ns07 = ({'dataclass': dataclass, 'field': field}, {'dataclass': dataclass, 'field': field})
exec(REF02['Config'], _ns02)
exec(REF07['Config'], _ns07)
RefCfg02, RefCfg07 = (_ns02['Config'](), _ns07['Config']())
for name in ['load_diat', 'save_ckpt', 'load_ckpt', 'evaluate', 'METRICS', 'train_one', 'paired_bootstrap', 'H1_ORDER', 'D2_PATHS']:
    exec(REF02[name], globals())
for name in ['VanillaLIF', 'fast_download', 'SHD_URLS', '_download_shd', '_bin_h5', 'load_shd_disjoint', '_make_synth_h5']:
    exec(REF07[name], globals())
exec(REF04['ParametricLIF'], globals())
NeuronRegistry._all['Vanilla-LIF'] = VanillaLIF
NeuronRegistry._all['Parametric-LIF'] = ParametricLIF
REF_SOURCES = {**{f'02:{k}': v for k, v in REF02.items()}, **{f'07:{k}': v for k, v in REF07.items()}, '04:ParametricLIF': REF04['ParametricLIF'], 'mstask.core': (NTM_DIR / 'src' / 'mstask' / 'core.py').read_text(encoding='utf-8'), 'dsn.ms_if': (Path(dikmen_neurons.__file__).parent / 'ms_if' / '__init__.py').read_text(encoding='utf-8'), 'dsn.base': (Path(dikmen_neurons.__file__).parent / 'base.py').read_text(encoding='utf-8')}
REF_HASH = hashlib.sha256('\n'.join((REF_SOURCES[k] for k in sorted(REF_SOURCES))).encode()).hexdigest()[:12]
PROVENANCE = 'imported functions AST-identical to Drive notebook "Shd deep probe .ipynb" (2026-06-14..18); no_grad calibration wrapper retained; historical runtime trajectory not fully reconstructed; pinned ' + DSN_SHA[:8] + ')'
calibrate_thresholds = torch.no_grad()(mstask.core.calibrate_thresholds)
_ref_save_ckpt, _ref_load_ckpt = (save_ckpt, load_ckpt)

def save_ckpt(model, opt, epoch, metrics, path):
    path = Path(path)
    tmp = path.with_suffix('.pt.tmp')
    _ref_save_ckpt(model, opt, epoch, metrics, tmp)
    os.replace(tmp, path)

def load_ckpt(model, opt, path):
    path = Path(path)
    try:
        return _ref_load_ckpt(model, opt, path)
    except Exception as e:
        if path.exists():
            path.replace(path.with_suffix('.pt.corrupt'))
        print(f'    [RESUME] checkpoint unreadable ({type(e).__name__}); quarantined, starting this run fresh')
        return (0, {})

def write_json_atomic(obj, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def read_json_safe(path):
    try:
        return json.load(open(path))
    except Exception:
        Path(path).replace(Path(path).with_suffix('.json.corrupt'))
        print(f'[PERSIST] {Path(path).name} truncated; quarantined')
        return None

@dataclass
class Config:
    L: int = 4
    W: int = 128
    K: int = 5
    target_firing: float = 0.15
    norm: str = 'none'
    max_epochs: int = 200
    patience: int = 20
    val_frac: float = 0.15
    val_smooth: int = 5
    select_metric: str = 'macro_f1'
    lr: float = 0.001
    batch_size: int = 128
    seeds: list = field(default_factory=lambda: [42, 123, 999])
    nboot: int = 2000
    stft_window_diat: int = 16
    stft_window_shd: int = 8
    T: int = 12
    F: int = 700
    duration_s: float = 1.0
    sweep_widths: list = field(default_factory=lambda: [32, 64, 128, 256, 512])
    regime_widths: list = field(default_factory=lambda: [64, 256])
    rescue_w0: float = 0.2
    rescue_lr: float = 0.0003
    extra_seeds: list = field(default_factory=lambda: [7, 2024])
    s8_pool_T: list = field(default_factory=lambda: [16, 8, 32])
    s8_rho_w0: list = field(default_factory=lambda: [0.25, 0.35, 0.4, 0.45, 0.55, 0.7])
    s8_rho_seeds: list = field(default_factory=lambda: [42])
    s8_shd_T: int = 64
    s9_dualtau_beta: list = field(default_factory=lambda: [0.8, 0.98])
    run_stages: list = field(default_factory=lambda: ['S3_rescue', 'S1_cap_sweep', 'S4_shd', 'S2_regime', 'S6_seeds', 'S7_signal', 'S8_diat', 'S8_shd', 'S9_lifdiff', 'S9_initfix', 'S9_shd_rank', 'S5_analysis'])
    smoke: bool = False
    exp_total: int = 12
    exp_manifest: list = field(default_factory=lambda: [(1, 'S3_rescue: CrossInhib collapse on radar, diagnosis + rescue arms; STFT/Phase diagnostics', 'pending'), (2, 'S1_cap_sweep: Vanilla/Parametric/Doppler vs width (LIF plateau)', 'pending'), (3, 'S2_regime: matched-capacity ablation at W=64 and W=256; unit-matched bank', 'pending'), (4, 'S4_shd: SHD speaker-disjoint re-runs with parameter dumps', 'pending'), (5, 'S6_seeds: two extra seeds for the 2x2 cells (radar + audio)', 'pending'), (6, 'S7_signal: DIAT first-difference / SHD cumulative-sum re-parameterisation', 'pending'), (7, 'S8_diat: sequence length on radar (T=16/8/32) + rho sweep at T=64', 'pending'), (8, 'S8_shd: sequence length on audio (SHD at T=64)', 'pending'), (9, 'S9_lifdiff: Vanilla/Parametric LIF on the radar first-difference view', 'pending'), (10, 'S9_initfix: Dual-tau and STFT with corrected initialisation (radar + audio)', 'pending'), (11, 'S9_shd_rank: the paper SHD ranking (Table 3) re-run in this harness + Parametric-LIF', 'pending'), (12, 'S5_analysis: tables, bootstraps, capacity table, revision summary', 'pending')])
cfg = Config()
if STAGES:
    unknown = [x for x in STAGES if x not in cfg.run_stages]
    assert not unknown, f'unknown stage(s) {unknown}; pick from {cfg.run_stages}'
    cfg.run_stages = list(STAGES)
if ONLY_METHODS:
    assert isinstance(ONLY_METHODS, dict), "ONLY_METHODS must be {'<stage>': [<method names>]}"
    stray = [k for k in ONLY_METHODS if k not in cfg.run_stages]
    assert not stray, f'ONLY_METHODS names stage(s) {stray} that this session does not run ({cfg.run_stages})'
EXP_ROOT.mkdir(parents=True, exist_ok=True)
RUN_VERSION = 1
ANALYSIS_VERSION = 2
_KEY_EXCLUDE = ('run_stages', 'exp_total', 'exp_manifest', 'smoke')
DEVIATIONS = {'S1_cap_sweep': 'width W varied per run (paper: W=128); Vanilla/Parametric-LIF defined exactly as in 07/04', 'S2_regime': 'reference width W_ref in {64,256} instead of 128; D2_units_W128 matches spiking-unit count (K*L*w = L*W) instead of parameters; D2_concat_stable_W128 = paper bank with the CrossInhib path at the stable init', 'S3_rescue': 'CI_stable: CrossInhib-LIF with initial cross weights softplus(w)=%.2f instead of the library 0.854 (dynamics unchanged); CI_lowlr: lr=%.0e instead of 1e-3' % (cfg.rescue_w0, cfg.rescue_lr), 'S4_shd': 'as 07_shd_disjoint; CI_stable arm added; learned neuron parameters dumped from the last trained epoch (<= patience epochs after the selected one)', 'S6_seeds_diat': "seeds %s added to the paper's [42,123,999] for Doppler / CrossInhib / CrossInhib-stable at W=128 (radar)" % cfg.extra_seeds, 'S6_seeds_shd': "seeds %s added to the paper's [42,123,999] for Doppler / CrossInhib / CrossInhib-stable at W=128 (audio)" % cfg.extra_seeds, 'S7_signal_diat': "input re-parameterised by the invertible first difference x'[t]=x[t]-x[t-1], x'[0]=x[0] (after the paper's train-stat standardisation); protocol otherwise unchanged", 'S7_signal_shd': "input re-parameterised by the invertible cumulative sum x'[t]=sum_{s<=t} x[s] over the 12 frames; protocol otherwise unchanged", 'diagnostics': 'analysis-only forward passes on a fixed test batch after training; no effect on training or selection', 'S8_diat_T16': "DIAT mean-pooled over time from 64 to 16 frames (after the paper's standardisation); protocol otherwise unchanged", 'S8_diat_T8': 'DIAT mean-pooled over time from 64 to 8 frames; protocol otherwise unchanged', 'S8_diat_T32': 'DIAT mean-pooled over time from 64 to 32 frames; protocol otherwise unchanged', 'S8_diat_rho': 'CrossInhib-LIF with initial cross weight softplus(w) = w0 swept (library: 0.854); T=64; ONE seed per w0 (boundary sweep: collapse is a 0.05-vs-0.88 macro-F1 outcome)', 'S8_shd_T64': "SHD re-binned to T=64 frames (paper: 12); the registry entry keeps the paper's T=12; protocol otherwise unchanged", 'S9_lifdiff': "Vanilla-LIF and Parametric-LIF on the S7 radar view x'[t]=x[t]-x[t-1], x'[0]=x[0] (W=%d); protocol otherwise unchanged" % cfg.W, 'S9_initfix_diat': "Dual-tau-LIF with leaks initialised at sigmoid(stored) = %s (library: 0.444/0.495 because log() literals pass through sigmoid); STFT-IF with band centre sigmoid(stored) = u, u = the library's own draw rand*0.5 clamped to [1e-3, 0.499], i.e. inside the rFFT range [0, 0.5] (library: sigmoid(u) in [0.5, 0.622]); dynamics unchanged; radar W=%d; a library-init Dual-tau arm is run alongside" % (cfg.s9_dualtau_beta, cfg.W), 'S9_initfix_shd': 'as S9_initfix_diat, on SHD (T=%d, W=%d); the library-init arms are r_dualtau / r_stft of S9_shd_rank' % (cfg.T, cfg.W), 'S9_shd_rank': "none: the paper's SHD neurons not re-run in S4 (Dual-tau, Beam, Chirp, Vanilla, Phase, STFT) plus Parametric-LIF, as 07_shd_disjoint"}
for _T in cfg.s8_pool_T:
    DEVIATIONS.setdefault(f'S8_diat_T{_T}', f'DIAT mean-pooled over time from 64 to {_T} frames; protocol otherwise unchanged')
DEVIATIONS.setdefault(f'S8_shd_T{cfg.s8_shd_T}', f'SHD re-binned to T={cfg.s8_shd_T} frames (paper: 12); registry entry untouched')

def fidelity_gate():
    checks = [('02', RefCfg02, 'L', cfg.L), ('02', RefCfg02, 'K', cfg.K), ('02', RefCfg02, 'target_firing', cfg.target_firing), ('02', RefCfg02, 'norm', cfg.norm), ('02', RefCfg02, 'stft_window', cfg.stft_window_diat), ('02', RefCfg02, 'val_frac', cfg.val_frac), ('02', RefCfg02, 'select_metric', cfg.select_metric), ('02', RefCfg02, 'lr', cfg.lr), ('02', RefCfg02, 'batch_size', cfg.batch_size), ('07', RefCfg07, 'T', cfg.T), ('07', RefCfg07, 'F', cfg.F), ('07', RefCfg07, 'duration_s', cfg.duration_s), ('07', RefCfg07, 'stft_window', cfg.stft_window_shd), ('07', RefCfg07, 'lr', cfg.lr), ('07', RefCfg07, 'target_firing', cfg.target_firing)]
    locked = [('02', RefCfg02, 'W', cfg.W), ('02', RefCfg02, 'max_epochs', cfg.max_epochs), ('02', RefCfg02, 'patience', cfg.patience), ('02', RefCfg02, 'val_smooth', cfg.val_smooth), ('02', RefCfg02, 'seeds', cfg.seeds), ('07', RefCfg07, 'max_epochs', cfg.max_epochs), ('07', RefCfg07, 'patience', cfg.patience), ('07', RefCfg07, 'val_smooth', cfg.val_smooth), ('07', RefCfg07, 'seeds', cfg.seeds), ('07', RefCfg07, 'nboot', cfg.nboot)]
    bad = []
    for tag, ref, fld, ours in checks + ([] if cfg.smoke else locked):
        rv = getattr(ref, fld)
        if rv != ours:
            bad.append(f'  {tag}.Config.{fld}: reference={rv!r}  ours={ours!r}')
    for fld in ['max_epochs', 'patience', 'val_smooth', 'val_frac', 'lr', 'batch_size', 'target_firing', 'select_metric', 'norm', 'L', 'W']:
        if getattr(RefCfg02, fld) != getattr(RefCfg07, fld):
            bad.append(f'  02 vs 07 disagree on {fld}: {getattr(RefCfg02, fld)!r} vs {getattr(RefCfg07, fld)!r}')
    if bad:
        raise SystemExit("[FIDELITY] HALT, config disagrees with the paper's scripts:\n" + '\n'.join(bad))
    print(f'[FIDELITY] config gate passed ({len(checks) + (0 if cfg.smoke else len(locked))} fields)  ref_hash={REF_HASH}' + ('   [SMOKE: epochs/patience/seeds relaxed]' if cfg.smoke else ''))

class CrossInhibLIFStable(CrossInhibLIF):
    """DEVIATION (S3/S4 rescue arm). Library init: w_inh=w_exc=0.3 -> softplus 0.854, and with
    beta_e=sigmoid(0.95)=0.721, beta_i=sigmoid(0.90)=0.711 the differential mode of the E-I pair
    has eigenvalue ~ beta + w = 1.57 > 1 (unstable). Here softplus(w) = cfg.rescue_w0 at init,
    so beta + w < 1. The weights stay learnable; single_step is the library's, untouched."""
    _description = 'CrossInhib-LIF with a stable initial cross weight (revision rescue arm).'

    def __init__(self, size, beta=0.95, threshold=1.0, w0=None):
        super().__init__(size, beta, threshold)
        w0 = cfg.rescue_w0 if w0 is None else w0
        raw = math.log(math.expm1(w0))
        with torch.no_grad():
            self.w_inh.fill_(raw)
            self.w_exc.fill_(raw)
NeuronRegistry._all['CrossInhib-LIF-stable'] = CrossInhibLIFStable

def _logit(p):
    return math.log(p / (1.0 - p))

class DualTauLIFFixInit(DualTauLIF):
    """DEVIATION (S9_initfix). The library stores log(0.8), log(0.98) but reads the leaks through sigmoid, so they start at
    0.444 / 0.495 ('fast' and 'slow' almost equal). Here sigmoid(stored) = cfg.s9_dualtau_beta at init, the values the
    literals express. No random draw is added or removed, so every later weight init is the same as the library arm's.
    single_step is the library's, untouched; the leaks stay learnable."""
    _description = 'Dual-tau-LIF with leaks initialised at the intended 0.80 / 0.98 (revision sensitivity arm).'

    def __init__(self, size, beta=0.95, threshold=1.0):
        super().__init__(size, beta, threshold)
        bf, bs = cfg.s9_dualtau_beta
        with torch.no_grad():
            self.log_beta_fast.fill_(_logit(bf))
            self.log_beta_slow.fill_(_logit(bs))
NeuronRegistry._all['Dual-tau-LIF-fixinit'] = DualTauLIFFixInit

class STFTIFFixInit(STFTIF):
    """DEVIATION (S9_initfix). The library draws u = rand*0.5 and stores it as band_center, then reads fc = sigmoid(u), so
    every band starts in [0.5, 0.622]: at or above the highest rFFT bin k/N = 0.5. Here the SAME draw u (no extra random
    number, so later inits are unchanged) is stored as logit(u), i.e. fc = u in [0, 0.5] at init. single_step untouched."""
    _description = 'STFT-IF with the band centre initialised inside the rFFT range (revision sensitivity arm).'

    def __init__(self, size, beta=0.95, threshold=1.0, window_len=16):
        super().__init__(size, beta, threshold, window_len)
        with torch.no_grad():
            u = self.band_center.clamp(0.001, 0.5 - 0.001)
            self.band_center.copy_(torch.log(u) - torch.log1p(-u))
NeuronRegistry._all['STFT-IF-fixinit'] = STFTIFFixInit

def nk_for(window_len):
    """neuron kwargs by registry name; the corrected STFT arm needs the same window as the library STFT"""
    return {'STFT-IF': {'window_len': window_len}, 'STFT-IF-fixinit': {'window_len': window_len}}

def ei_spectral_radius(beta_e, beta_i, w_inh, w_exc):
    """max |eig| of [[be, -wi], [-we, bi]] per channel (arrays, already through sigmoid/softplus)."""
    tr = beta_e + beta_i
    det = beta_e * beta_i - w_inh * w_exc
    disc = tr * tr / 4 - det
    lam1 = tr / 2 + np.sqrt(np.abs(disc)) * (disc >= 0)
    lam2 = tr / 2 - np.sqrt(np.abs(disc)) * (disc >= 0)
    mag = np.where(disc >= 0, np.maximum(np.abs(lam1), np.abs(lam2)), np.sqrt(np.abs(det)))
    return mag

def build_model(method, Fin, C, NK, matched):
    s = matched[method]
    n = cfg.norm
    if s['arch'] == 'homog':
        return VerticalNet(Fin, C, [s['neuron']] * cfg.L, s['W'], NK, n)
    if s['arch'] == 'vertical':
        return VerticalNet(Fin, C, list(s['order']), s['W'], NK, n)
    if s['arch'] == 'bank':
        return PathBankNet(Fin, C, list(s.get('paths', D2_PATHS)), s['w'], cfg.L, 'concat', NK, n, True)
    raise KeyError(method)

def count_units(model):
    return int(sum((m.size for m in model.modules() if isinstance(m, BaseNeuron))))

def solve_bank_width(Fin, C, NK, target_params):
    w, p, _ = _solve_width(lambda w: PathBankNet(Fin, C, list(D2_PATHS), w, cfg.L, 'concat', NK, cfg.norm, True), target_params, wmax=600)
    return (w, p)

@torch.no_grad()
def diagnose_model(model, xb):
    model.eval()
    recs = []
    if isinstance(model, VerticalNet):
        stacks, h0 = ([('stack', model.stack)], xb)
    else:
        h0 = xb
        if model.stem is not None:
            B, T, d = xb.shape
            h0 = model.stem(xb.reshape(B * T, d)).reshape(B, T, -1)
        stacks = [(f'path{i}', p) for i, p in enumerate(model.paths)]
    for sname, stack in stacks:
        h = h0
        for li, (proj, norm, neuron) in enumerate(zip(stack.projs, stack.norms, stack.neurons)):
            B, T, d = h.shape
            z = norm(proj(h.reshape(B * T, d))).reshape(B, T, -1)
            spk, _ = neuron(z)
            s = spk.float()
            tidx = torch.arange(T, device=s.device).view(1, T, 1).expand_as(s)
            first = torch.where(s > 0, tidx, torch.full_like(tidx, T)).min(dim=1).values.float()
            fired = first < T
            dz = (z[:, 1:] - z[:, :-1]).abs().flatten()
            q = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], device=z.device)
            sub = dz[torch.randperm(dz.numel(), device=dz.device)[:200000]] if dz.numel() > 200000 else dz
            rec = {'stack': sname, 'layer': li, 'neuron': type(neuron).__name__, 'threshold': float(neuron.threshold), 'rate': float(s.mean()), 'rate_t': [round(v, 5) for v in s.mean(dim=(0, 2)).tolist()], 'dead_unit_frac': float((s.sum(dim=(0, 1)) == 0).float().mean()), 'silent_after_first_frac': float(((s.sum(dim=1) == 1) & fired).float().sum() / max(1.0, fired.float().sum())), 'first_spike_t_mean': float(first[fired].mean()) if fired.any() else None, 'dz_abs_quantiles': [round(v, 5) for v in torch.quantile(sub, q).tolist()], 'z_abs_quantiles': [round(v, 5) for v in torch.quantile(z.abs().flatten()[:200000], q).tolist()], 'params': {pn: [round(v, 5) for v in p.detach().cpu().flatten().tolist()] for pn, p in neuron.named_parameters()}}
            if hasattr(neuron, 'w_inh'):
                be = torch.sigmoid(neuron.beta_e).cpu().numpy()
                bi = torch.sigmoid(neuron.beta_i).cpu().numpy()
                wi = F.softplus(neuron.w_inh).cpu().numpy()
                we = F.softplus(neuron.w_exc).cpu().numpy()
                rho = ei_spectral_radius(be, bi, wi, we)
                rec['ei'] = {'rho_mean': float(rho.mean()), 'rho_median': float(np.median(rho)), 'frac_unstable': float((rho > 1).mean()), 'beta_e_mean': float(be.mean()), 'beta_i_mean': float(bi.mean()), 'w_inh_mean': float(wi.mean()), 'w_exc_mean': float(we.mean())}
            if hasattr(neuron, 'f_center'):
                fc = neuron.f_center.detach().cpu().numpy()
                bw = neuron.log_bw.exp().detach().cpu().numpy()
                rec['doppler'] = {'f_center_quantiles': [round(float(v), 5) for v in np.quantile(fc, [0.05, 0.25, 0.5, 0.75, 0.95])], 'frac_f_center_negative': float((fc < 0).mean()), 'bw_median': float(np.median(bw))}
            recs.append(rec)
            h = spk
    return recs

def diagnose_from_ckpt(k, sd, Fin, C, NK, specs, fire_batch, diag_batch, ckpt_path):
    """Rebuild the run's model the way train_one does (seed -> build -> calibrate thresholds on fire_batch), THEN load the
    trained weights. The calibrated thresholds are plain float attributes, not state_dict entries, so loading a checkpoint
    into a fresh model leaves thr=1.0 and every firing diagnostic would be wrong (ANALYSIS_VERSION 1 had this bug)."""
    torch.manual_seed(sd)
    np.random.seed(sd)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(sd)
    model = build_model(k, Fin, C, NK, specs).to(device)
    calibrate_thresholds(model, fire_batch.to(device), cfg.target_firing)
    thresholds = [float(n.threshold) for n in model.modules() if isinstance(n, BaseNeuron)]
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck['model_state_dict'])
    diag = diagnose_model(model, diag_batch)
    meta = {'calibrated': True, 'analysis_version': ANALYSIS_VERSION, 'thresholds': thresholds, 'note': 'parameters/firing from the LAST trained epoch (<= patience epochs after the selected one); test scores from the selected epoch'}
    return (diag, meta, ck)

def const_predictor_macro_f1(trues, C):
    """macro-F1 of a classifier that always outputs class c, for every c (derived, not asserted)."""
    trues = np.asarray(trues)
    N = len(trues)
    out = {}
    for c in range(C):
        n_c = int((trues == c).sum())
        out[str(c)] = round(2 * n_c / (N + n_c) / C, 4) if n_c else 0.0
    return out

def fingerprint(*arrays):
    h = hashlib.md5()
    for a in arrays:
        h.update(np.ascontiguousarray(a.cpu().numpy() if torch.is_tensor(a) else a).tobytes())
    return h.hexdigest()[:8]

def _make_synth_diat():
    raise RuntimeError('Synthetic data is not a study input.')

def ensure_diat_registered():
    reg = load_registry()
    if 'diat' not in reg:
        path = DIAT_DATA_DEFAULT if ON_COLAB else str(_make_synth_diat())
        register_dataset('diat', path=path, source='DIAT micro-Doppler uSAT', format='folder-npy', classes=6, T=64, features=64, notes='X.npy [N,64,64] + y.npy; 80/20 stratified split seed=0')
        print(f"[REGISTRY] 'diat' registered -> {path}")
    p = Path(load_registry()['diat']['path'])
    missing = [f for f in ('X.npy', 'y.npy') if not (p / f).exists()]
    if missing:
        raise FileNotFoundError(f"DIAT files {missing} not found in {p}\n  -> edit DIAT_DATA_DEFAULT at the top of the cell, or fix 'diat' in {RESEARCH / 'datasets.json'}")
    print(f'[DATA] DIAT folder: {p}  (X.npy {os.path.getsize(p / 'X.npy') / 1000000.0:.1f} MB)')

def get_diat():
    ensure_diat_registered()
    Xtr, ytr, Xva, yva, Xte, yte = load_diat()
    fp = fingerprint(Xtr, ytr, Xva, yva, Xte, yte)
    print(f'[DATA] DIAT train={tuple(Xtr.shape)} val={tuple(Xva.shape)} test={tuple(Xte.shape)}  fp={fp}')
    return ((Xtr, ytr, Xva, yva, Xte, yte), fp)

def get_shd():
    Xtr, ytr, Xva, yva, Xte, yte = load_shd_disjoint()
    fp = fingerprint(Xtr, ytr, Xva, yva, Xte, yte)
    print(f'[DATA] SHD  train={tuple(Xtr.shape)} val={tuple(Xva.shape)} test={tuple(Xte.shape)}  fp={fp}')
    return ((Xtr, ytr, Xva, yva, Xte, yte), fp)

def make_loaders(data):
    Xtr, ytr, Xva, yva, Xte, yte = data
    nw = 2 if ON_COLAB else 0
    tr = DataLoader(TensorDataset(Xtr, ytr), batch_size=cfg.batch_size, shuffle=True, num_workers=nw, pin_memory=device.type == 'cuda', persistent_workers=nw > 0)
    va = DataLoader(TensorDataset(Xva, yva), batch_size=256, shuffle=False)
    te = DataLoader(TensorDataset(Xte, yte), batch_size=256, shuffle=False)
    return (tr, va, te)
SESSION_ID = hashlib.md5(f'{os.getpid()}-{datetime.now().isoformat()}'.encode()).hexdigest()[:6]

def stage_guard(EXP_DIR, stage, mine=None, touch_only=False):
    """Warn (never block) when another session looks active on the same stage. Parallel sessions are fine
    on DIFFERENT stages: each writes its own experiment dir. Two sessions on ONE stage fight over checkpoints."""
    lock = EXP_DIR / '_running.json'
    if not touch_only and lock.exists():
        info = read_json_safe(lock) or {}
        age_min = (time.time() - lock.stat().st_mtime) / 60
        theirs = info.get('methods')
        overlap = (mine is None or theirs is None) or set(mine) & set(theirs)
        if info.get('session') != SESSION_ID and age_min < 60 and overlap:
            both = sorted(set(mine) & set(theirs)) if mine and theirs else 'ALL'
            print('!' * 78)
            print(f'!! WARNING: stage {stage} was written {age_min:.0f} min ago by another session ({info.get('session')}),')
            print(f'!! and the method sets OVERLAP on: {both}')
            print('!! Two sessions on one method+seed fight over the same checkpoint. Stop one, or split ONLY_METHODS.')
            print('!' * 78)
        elif info.get('session') != SESSION_ID and age_min < 60:
            print(f'[PARALLEL] stage {stage} also active in session {info.get('session')} on a disjoint method set: OK')
    write_json_atomic({'session': SESSION_ID, 'stage': stage, 'host_pid': os.getpid(), 'methods': list(mine) if mine is not None else None, 'updated': datetime.now().isoformat()}, lock)

def run_key(stage, data_fp):
    payload = {k: v for k, v in asdict(cfg).items() if k not in _KEY_EXCLUDE}
    payload.update(stage=stage, run_version=RUN_VERSION, data_fp=data_fp, ref_hash=REF_HASH)
    return hashlib.md5(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:8]

def _entry_key(name):
    return name.split(':')[0]

def print_manifest(current):
    print(f'[PROGRESS] Manifest:')
    for idx, name, status in cfg.exp_manifest:
        st = 'THIS' if current and current.startswith(_entry_key(name)) else status
        mark = '->' if st == 'THIS' else 'x' if st == 'done' else '  '
        print(f'   {mark} {idx}/{cfg.exp_total}: {name} [{st}]')

def probe_cost(spec_model, tr_loader, Fin, C, NK):
    """time one real forward+backward on the ACTUAL batch shape, extrapolate one epoch."""
    xb, yb = next(iter(tr_loader))
    xb, yb = (xb.to(device), yb.to(device))
    m = spec_model.to(device)
    opt = torch.optim.Adam(m.parameters(), lr=cfg.lr)
    for _ in range(2):
        t0 = time.time()
        opt.zero_grad()
        F.cross_entropy(m(xb), yb).backward()
        opt.step()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        dt = time.time() - t0
    n_b = len(tr_loader)
    print(f'[SHAPE] batch={tuple(xb.shape)} fwd+bwd={dt:.2f}s -> ~{dt * n_b * 1.3:.0f}s/epoch (x{n_b} batches, +30% for the three evals); epochs per run are decided by early stopping')
    return dt * n_b * 1.3

def run_stage(stage, specs, data, data_fp, NK, mark_done):
    """specs: ordered dict key -> spec. Each (key, seed) is one run; finished runs are skipped."""
    key = run_key(stage, data_fp)
    EXP_DIR = EXP_ROOT / f'{stage}_v{RUN_VERSION}_{data_fp}'
    CKPT_DIR = EXP_DIR / 'checkpoints'
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    sel = (ONLY_METHODS or {}).get(stage)
    all_specs = specs
    if sel is not None:
        bad = [m for m in sel if m not in specs]
        assert not bad, f"ONLY_METHODS['{stage}'] names unknown method(s) {bad}; this stage has {list(specs)}"
        specs = {k: v for k, v in specs.items() if k in sel}
        if not specs:
            print(f'\n[STAGE] {stage}: method filter selected nothing; skipped in this session')
            return (EXP_DIR, {}, {})
    print(f'\n{'=' * 78}\n[STAGE] {stage}   dir={EXP_DIR.name}   key={key}   session={SESSION_ID}')
    if sel is not None:
        print(f'[FILTER] this session runs {len(specs)}/{len(all_specs)} methods: {list(specs)}')
        print(f'[FILTER] left to another session: {[k for k in all_specs if k not in specs]}')
    stage_guard(EXP_DIR, stage, mine=list(specs))
    print(f'[DEVIATION] {DEVIATIONS[stage]}')
    print_manifest(stage)
    Xtr, ytr, Xva, yva, Xte, yte = data
    Fin, C = (Xtr.shape[-1], int(max(ytr.max(), yva.max(), yte.max()) + 1))
    tr_loader, va_loader, te_loader = make_loaders(data)
    fire_batch = Xtr[:cfg.batch_size]
    diag_batch = Xte[:256].to(device)
    for k, s in specs.items():
        m = build_model(k, Fin, C, NK, specs)
        s['params'] = count_params(m)
        s['units'] = count_units(m)
    n_runs = len(specs) * len(cfg.seeds)
    done = {p.name for p in EXP_DIR.glob('results_*.json')}
    todo = [(k, sd) for k in specs for sd in cfg.seeds if f'results_{k}_seed{sd}.json' not in done]
    print(f'[WORKLOAD] {len(specs)} methods x {len(cfg.seeds)} seeds = {n_runs} runs; {n_runs - len(todo)} already done, {len(todo)} to run; <= {cfg.max_epochs} epochs each, patience {cfg.patience}')
    for k, s in specs.items():
        print(f'   {k:28s} params={s['params']:>8,d}  units={s['units']:>5d}  {s.get('note', '')}')
    if todo:
        k0 = todo[0][0]
        est_epoch = probe_cost(build_model(k0, Fin, C, NK, specs), tr_loader, Fin, C, NK)
        print(f'[ETA] very rough: {len(todo)} runs x ~150 epochs x {est_epoch:.0f}s = {len(todo) * 150 * est_epoch / 3600:.1f} h (~150 = typical stopping epoch of the revision runs; a run that collapses stops after ~25)')
    results, first_t, run_counter = ({}, None, 0)
    for k, s in specs.items():
        for sd in cfg.seeds:
            run_counter += 1
            out_path = EXP_DIR / f'results_{k}_seed{sd}.json'
            if out_path.exists():
                r = read_json_safe(out_path)
                if r and r.get('run_version') == RUN_VERSION and (r.get('ref_hash') == REF_HASH) and (bool(r.get('smoke')) == cfg.smoke):
                    if not r.get('diagnostics_meta', {}).get('calibrated'):
                        ckp = CKPT_DIR / f'{k}_seed{sd}.pt'
                        if ckp.exists():
                            diag, meta, _ = diagnose_from_ckpt(k, sd, Fin, C, NK, specs, fire_batch, diag_batch, ckp)
                            r['diagnostics'], r['diagnostics_meta'] = (diag, meta)
                            r.pop('diag_note', None)
                            write_json_atomic(r, out_path)
                            print(f'  Run {run_counter}/{n_runs}: {k} | seed={sd}  [SKIP] done; diagnostics re-computed with calibrated thresholds')
                        else:
                            print(f'  Run {run_counter}/{n_runs}: {k} | seed={sd}  [SKIP] done; WARNING no checkpoint, old (thr=1.0) diagnostics kept')
                    else:
                        print(f'  Run {run_counter}/{n_runs}: {k} | seed={sd}  [SKIP] done')
                    results.setdefault(k, []).append(r)
                    continue
                print(f'  Run {run_counter}/{n_runs}: {k} | seed={sd}  stale result ignored (version/ref/smoke mismatch)')
            print(f'\n  Run {run_counter}/{n_runs}: {k} | seed={sd} | lr={s.get('lr', cfg.lr):.0e}')
            t0 = time.time()
            lr_saved = cfg.lr
            cfg.lr = s.get('lr', cfg.lr)
            try:
                best, flog = train_one(k, sd, Fin, C, NK, specs, tr_loader, va_loader, te_loader, fire_batch, CKPT_DIR)
            finally:
                cfg.lr = lr_saved
            dt = time.time() - t0
            diag, meta, ck = diagnose_from_ckpt(k, sd, Fin, C, NK, specs, fire_batch, diag_batch, CKPT_DIR / f'{k}_seed{sd}.pt')
            r = {'stage': stage, 'method': k, 'spec': {kk: vv for kk, vv in s.items()}, 'seed': sd, 'acc': best['acc'], 'f1': best['f1'], 'f1_per_class': best['f1_per_class'], 'preds': best['preds'], 'trues': best['trues'], 'best_epoch': int(ck['metrics']['best_epoch']), 'epochs_trained': int(ck['epoch']) + 1, 'fire_log': flog, 'diagnostics': diag, 'diagnostics_meta': meta, 'const_predictor_macro_f1': const_predictor_macro_f1(best['trues'], C), 'time_s': round(dt, 1), 'run_version': RUN_VERSION, 'ref_hash': REF_HASH, 'data_fp': data_fp, 'smoke': cfg.smoke, 'deviation': DEVIATIONS[stage], 'timestamp': datetime.now().isoformat()}
            write_json_atomic(r, out_path)
            stage_guard(EXP_DIR, stage, mine=list(specs), touch_only=True)
            results.setdefault(k, []).append(r)
            if first_t is None:
                first_t = dt
                print(f'    [ETA] first run {dt / 60:.1f} min -> ~{dt * (len(todo) - 1) / 60:.0f} min for the remaining {len(todo) - 1} runs')
            print(f'    => {k} seed {sd}: acc {best['acc']:.4f} f1 {best['f1']:.4f}  (best ep {r['best_epoch']}, {dt / 60:.1f} min)')
    agg = {}
    for k, rs in results.items():
        f1s = [r['f1'] for r in rs]
        accs = [r['acc'] for r in rs]
        agg[k] = {'f1_mean': float(np.mean(f1s)), 'f1_std': float(np.std(f1s)), 'f1s': f1s, 'acc_mean': float(np.mean(accs)), 'acc_std': float(np.std(accs)), 'accs': accs, 'params': specs[k]['params'], 'units': specs[k]['units'], 'spec': specs[k], 'n_seeds': len(rs)}
        print(f'  => {k:28s} F1 {agg[k]['f1_mean']:.4f} +/- {agg[k]['f1_std']:.4f}   acc {agg[k]['acc_mean']:.4f}   params {specs[k]['params']:,}  units {specs[k]['units']}')
    summary = {'stage': stage, 'run_key': key, 'run_version': RUN_VERSION, 'analysis_version': ANALYSIS_VERSION, 'ref_hash': REF_HASH, 'ntm_sha': NTM_SHA, 'dsn_sha': DSN_SHA, 'provenance': PROVENANCE, 'data_fp': data_fp, 'smoke': cfg.smoke, 'deviation': DEVIATIONS[stage], 'config': asdict(cfg), 'results': agg, 'method_filter': sel, 'methods_in_stage': list(all_specs), 'complete': sel is None and all((len(results.get(k, [])) == len(cfg.seeds) for k in specs)), 'this_session_complete': all((len(results.get(k, [])) == len(cfg.seeds) for k in specs)), 'timestamp': datetime.now().isoformat()}
    write_json_atomic(summary, EXP_DIR / ('summary.json' if sel is None else f'summary_{SESSION_ID}.json'))
    write_json_atomic(summary, REV / f'latest_summary_{stage}{('' if sel is None else '_' + SESSION_ID)}.json')
    if summary['complete']:
        mark_done(stage)
    return (EXP_DIR, results, agg)

def specs_S3():
    return {'CI_default': {'arch': 'homog', 'neuron': 'CrossInhib-LIF', 'W': cfg.W, 'note': 'paper arm (Fig. 5 radar side)'}, 'CI_stable': {'arch': 'homog', 'neuron': 'CrossInhib-LIF-stable', 'W': cfg.W, 'note': f'rescue: softplus(w0)={cfg.rescue_w0}'}, 'CI_lowlr': {'arch': 'homog', 'neuron': 'CrossInhib-LIF', 'W': cfg.W, 'lr': cfg.rescue_lr, 'note': 'rescue: low lr'}, 'STFT_default': {'arch': 'homog', 'neuron': 'STFT-IF', 'W': cfg.W, 'note': 'diagnostics (collapse in Fig. 3)'}, 'Phase_default': {'arch': 'homog', 'neuron': 'Phase-LIF', 'W': cfg.W, 'note': 'diagnostics (collapse in Fig. 5)'}}

def specs_S1():
    out = {}
    for W in cfg.sweep_widths:
        for tag, neuron in [('vanilla', 'Vanilla-LIF'), ('plif', 'Parametric-LIF'), ('doppler', 'Doppler-LIF')]:
            out[f'cap_{tag}_W{W}'] = {'arch': 'homog', 'neuron': neuron, 'W': W}
    return out

def specs_S2(Fin, C, NK):
    """Ordered by how much the revision needs each one, because a parallel session may be stopped early:
    1. the paper's bank with a LIVE CrossInhib path (the paper's bank carried a dead path)
    2. the second width regime at W=64
    3. the unit-matched bank
    4. the second width regime at W=256"""
    out = {}
    d1_ref = count_params(VerticalNet(Fin, C, list(H1_ORDER), cfg.W, NK, cfg.norm))
    w_ref, _ = solve_bank_width(Fin, C, NK, d1_ref)
    paths_stable = ['CrossInhib-LIF-stable' if t == 'CrossInhib-LIF' else t for t in D2_PATHS]
    out[f'D2_concat_stable_W{cfg.W}'] = {'arch': 'bank', 'w': w_ref, 'W': cfg.W, 'paths': paths_stable, 'note': f'PRIORITY 1: paper bank, CrossInhib path at the stable init (matched to D1 {d1_ref:,} params)'}

    def regime(W, prio):
        d1 = count_params(VerticalNet(Fin, C, list(H1_ORDER), W, NK, cfg.norm))
        w, _ = solve_bank_width(Fin, C, NK, d1)
        out[f'D0_doppler_W{W}'] = {'arch': 'homog', 'neuron': 'Doppler-LIF', 'W': W, 'note': f'PRIORITY {prio}'}
        out[f'D1_H1_W{W}'] = {'arch': 'vertical', 'order': list(H1_ORDER), 'W': W, 'note': f'PRIORITY {prio}'}
        out[f'D2_concat_W{W}'] = {'arch': 'bank', 'w': w, 'W': W, 'note': f'PRIORITY {prio}: bank matched to D1 ({d1:,} params)'}
    widths = list(cfg.regime_widths)
    if widths:
        regime(widths[0], 2)
    w_u = max(2, int(round(cfg.W / cfg.K)))
    out[f'D2_units_W{cfg.W}'] = {'arch': 'bank', 'w': w_u, 'W': cfg.W, 'note': 'PRIORITY 3: UNIT-matched bank (fewer params)'}
    for W in widths[1:]:
        regime(W, 4)
    return out

def specs_S4():
    return {'shd_doppler': {'arch': 'homog', 'neuron': 'Doppler-LIF', 'W': cfg.W, 'note': 'Table 3 re-run + f_center dump'}, 'shd_CI_default': {'arch': 'homog', 'neuron': 'CrossInhib-LIF', 'W': cfg.W, 'note': 'Table 3 re-run + w dump'}, 'shd_CI_stable': {'arch': 'homog', 'neuron': 'CrossInhib-LIF-stable', 'W': cfg.W, 'note': 'rescue arm on audio'}}

def specs_2x2(prefix):
    return {f'{prefix}doppler': {'arch': 'homog', 'neuron': 'Doppler-LIF', 'W': cfg.W}, f'{prefix}CI_default': {'arch': 'homog', 'neuron': 'CrossInhib-LIF', 'W': cfg.W}, f'{prefix}CI_stable': {'arch': 'homog', 'neuron': 'CrossInhib-LIF-stable', 'W': cfg.W}}

def ci_with_w0(w0):
    """register (once) a CrossInhib-LIF whose cross weights start at softplus(w) = w0; dynamics = library's"""
    name = f'CrossInhib-LIF-w{w0:.2f}'
    if name not in NeuronRegistry._all:

        def _init(self, size, beta=0.95, threshold=1.0, _w0=w0):
            CrossInhibLIFStable.__init__(self, size, beta, threshold, w0=_w0)
        NeuronRegistry._all[name] = type(f'CrossInhibLIF_w{int(round(w0 * 100))}', (CrossInhibLIFStable,), {'__init__': _init})
    return name

def rho_init_of(neuron_name):
    """spectral radius of the linear E-I pair at initialisation (None for other neurons); constant fill -> one number"""
    n = NeuronRegistry.create(neuron_name, size=4)
    if not hasattr(n, 'w_inh'):
        return None
    be, bh = (torch.sigmoid(n.beta_e).detach().numpy(), torch.sigmoid(n.beta_i).detach().numpy())
    wi, we = (F.softplus(n.w_inh).detach().numpy(), F.softplus(n.w_exc).detach().numpy())
    return float(ei_spectral_radius(be, bh, wi, we).max())

def pool_time(data, T):
    """mean-pool the time axis to T frames (T must divide the original length)"""

    def f(X):
        B, T0, D = X.shape
        assert T0 % T == 0, f'cannot pool {T0} frames to {T}'
        return X.reshape(B, T, T0 // T, D).mean(2)
    Xtr, ytr, Xva, yva, Xte, yte = data
    return (f(Xtr), ytr, f(Xva), yva, f(Xte), yte)

def _s8_spec(neuron, T, domain):
    return {'arch': 'homog', 'neuron': neuron, 'W': cfg.W, 'T': T, 'domain': domain, 'rho_init': rho_init_of(neuron)}

def specs_S8_pool(T):
    if T == cfg.s8_pool_T[0]:
        return {f'q{T}_CI_default': _s8_spec('CrossInhib-LIF', T, 'radar'), f'q{T}_CI_stable': _s8_spec('CrossInhib-LIF-stable', T, 'radar'), f'q{T}_doppler': _s8_spec('Doppler-LIF', T, 'radar')}
    return {f'q{T}_CI_default': _s8_spec('CrossInhib-LIF', T, 'radar')}

def specs_S8_rho():
    return {f'rho_w{w0:.2f}': _s8_spec(ci_with_w0(w0), 64, 'radar') for w0 in cfg.s8_rho_w0}

def specs_S8_shd():
    T = cfg.s8_shd_T
    return {f't{T}_CI_default': _s8_spec('CrossInhib-LIF', T, 'audio'), f't{T}_CI_stable': _s8_spec('CrossInhib-LIF-stable', T, 'audio'), f't{T}_doppler': _s8_spec('Doppler-LIF', T, 'audio')}

def get_shd_T(T):
    """the reference SHD loader at a different binning, without touching the registry entry (paper: T=12)"""
    saved_T, saved_reg = (cfg.T, globals()['register_dataset'])
    cfg.T = T
    globals()['register_dataset'] = lambda *a, **k: None
    try:
        data = load_shd_disjoint()
    finally:
        cfg.T = saved_T
        globals()['register_dataset'] = saved_reg
    fp = fingerprint(*data)
    print(f'[DATA] SHD re-binned T={T}: train={tuple(data[0].shape)} test={tuple(data[4].shape)}  fp={fp}')
    return (data, fp)

def diat_first_difference(data):
    """S7: invertible re-parameterisation of time (x'[0]=x[0], x'[t]=x[t]-x[t-1]); same information, onset-rich statistics."""
    f = lambda X: torch.cat([X[:, :1], X[:, 1:] - X[:, :-1]], dim=1)
    Xtr, ytr, Xva, yva, Xte, yte = data
    return (f(Xtr), ytr, f(Xva), yva, f(Xte), yte)

def shd_cumulative(data):
    """S7: invertible re-parameterisation of time (x'[t]=cumsum_t x); same information, sustained statistics."""
    f = lambda X: torch.cumsum(X, dim=1)
    Xtr, ytr, Xva, yva, Xte, yte = data
    return (f(Xtr), ytr, f(Xva), yva, f(Xte), yte)

def specs_S9_lifdiff():
    """S9-a: the two LIF controls on the radar first-difference view (Doppler on that view is m_doppler of S7_signal_diat)"""
    return {'d_vanilla': {'arch': 'homog', 'neuron': 'Vanilla-LIF', 'W': cfg.W, 'note': 'vs cap_vanilla_W128 (S1, original input)'}, 'd_plif': {'arch': 'homog', 'neuron': 'Parametric-LIF', 'W': cfg.W, 'note': 'vs cap_plif_W128 (S1, original input)'}}

def specs_S9_initfix(domain):
    """S9-b, ordered by priority in case a session is stopped early. Radar carries its own library-init Dual-tau arm (the
    revision harness has none on radar); STFT's library arm is STFT_default of S3. On audio both library arms are in S9_shd_rank."""
    if domain == 'radar':
        return {'fx_stft_fixed': {'arch': 'homog', 'neuron': 'STFT-IF-fixinit', 'W': cfg.W, 'note': 'vs STFT_default (S3)'}, 'fx_dualtau_fixed': {'arch': 'homog', 'neuron': 'Dual-tau-LIF-fixinit', 'W': cfg.W, 'note': 'vs fx_dualtau_default'}, 'fx_dualtau_default': {'arch': 'homog', 'neuron': 'Dual-tau-LIF', 'W': cfg.W, 'note': 'library init, same harness'}}
    return {'fxs_stft_fixed': {'arch': 'homog', 'neuron': 'STFT-IF-fixinit', 'W': cfg.W, 'note': 'vs r_stft (S9_shd_rank)'}, 'fxs_dualtau_fixed': {'arch': 'homog', 'neuron': 'Dual-tau-LIF-fixinit', 'W': cfg.W, 'note': 'vs r_dualtau (S9_shd_rank)'}}
PAPER_TABLE3 = {'Dual-tau-LIF': 0.671, 'CrossInhib-LIF': 0.635, 'Beam-IF': 0.625, 'Chirp-LIF': 0.581, 'Vanilla-LIF': 0.581, 'Doppler-LIF': 0.451, 'Phase-LIF': 0.286, 'STFT-IF': 0.097}
S9_RANK_TAGS = [('r_dualtau', 'Dual-tau-LIF'), ('r_beam', 'Beam-IF'), ('r_chirp', 'Chirp-LIF'), ('r_vanilla', 'Vanilla-LIF'), ('r_phase', 'Phase-LIF'), ('r_stft', 'STFT-IF'), ('r_plif', 'Parametric-LIF')]

def specs_S9_shd_rank():
    """S9-c: Doppler and CrossInhib are already in S4_shd (same harness, same data fingerprint)"""
    return {k: {'arch': 'homog', 'neuron': n, 'W': cfg.W} for k, n in S9_RANK_TAGS}
