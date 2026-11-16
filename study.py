"""Original study programs and supplementary numerical stages."""
import json
import os
import platform
import runpy
from runtime import ROOT, read, sources


def run(args, jobs):
    os.environ['RESEARCH_ROOT'] = str(args.output)
    os.environ['RUN_ID'] = args.experiment
    os.environ['DIAT_DATA'] = str(args.data)
    os.environ['SHD_CACHE'] = str(args.data)
    os.environ['STUDY_SHD_DATA'] = str(args.shd_data or args.data)
    for key in ('REDUCED', 'SMOKE'):
        os.environ.pop(key, None)
    repo, lib, core = sources()
    import torch
    import numpy as np
    if not torch.cuda.is_available():
        raise RuntimeError('Study training requires CUDA; no training started.')
    env = dict(python=platform.python_version(), torch=str(torch.__version__),
               numpy=np.__version__, cuda=torch.version.cuda,
               cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(0))
    for module in ('scipy', 'sklearn', 'h5py', 'optuna'):
        try:
            env[module] = __import__(module).__version__
        except ImportError:
            if module == 'optuna' and 'search' not in args.experiment:
                continue
            raise
    (args.output / 'environment.json').write_text(json.dumps(env, indent=2), encoding='utf-8')
    entry = read(ROOT / 'configs/study.json')[args.experiment]
    if 'script' in entry:
        runpy.run_path(str(repo / 'experiments' / entry['script']), run_name='__main__')
    else:
        supplementary(repo, entry['stage'])


def supplementary(repo, stage):
    ns = runpy.run_path(str(repo / 'experiments/supplementary.py'))
    cfg = ns['cfg']
    cfg.run_stages = [stage]
    ns['fidelity_gate']()
    nk_diat = {'STFT-IF': {'window_len': cfg.stft_window_diat}}
    nk_shd = {'STFT-IF': {'window_len': cfg.stft_window_shd}}
    def run(name, specs, data, fingerprint, nk):
        return ns['run_stage'](name, specs, data, fingerprint, nk, lambda name: None)
    if stage in ('S1_cap_sweep', 'S2_regime', 'S3_rescue', 'S6_seeds',
                 'S7_signal', 'S8_diat', 'S9_lifdiff', 'S9_initfix'):
        diat, fp = ns['get_diat']()
    if stage == 'S1_cap_sweep':
        run(stage, ns['specs_S1'](), diat, fp, nk_diat)
    elif stage == 'S2_regime':
        classes = int(max(diat[1].max(), diat[3].max(), diat[5].max()) + 1)
        run(stage, ns['specs_S2'](diat[0].shape[-1], classes, nk_diat), diat, fp, nk_diat)
    elif stage == 'S3_rescue':
        run(stage, ns['specs_S3'](), diat, fp, nk_diat)
    elif stage == 'S4_shd':
        data, fp = ns['get_shd']()
        run(stage, ns['specs_S4'](), data, fp, nk_shd)
    elif stage == 'S6_seeds':
        cfg.seeds = list(cfg.extra_seeds)
        run('S6_seeds_diat', ns['specs_2x2']('x_'), diat, fp, nk_diat)
        data, fp = ns['get_shd']()
        run('S6_seeds_shd', ns['specs_2x2']('x_shd_'), data, fp, nk_shd)
    elif stage == 'S7_signal':
        data = ns['diat_first_difference'](diat)
        run('S7_signal_diat', ns['specs_2x2']('m_'), data, ns['fingerprint'](*data), nk_diat)
        data, fp = ns['get_shd']()
        data = ns['shd_cumulative'](data)
        run('S7_signal_shd', ns['specs_2x2']('m_'), data, ns['fingerprint'](*data), nk_shd)
    elif stage == 'S8_diat':
        def pooled(t):
            data = ns['pool_time'](diat, t)
            run(f'S8_diat_T{t}', ns['specs_S8_pool'](t), data, ns['fingerprint'](*data), nk_diat)
        pooled(cfg.s8_pool_T[0])
        saved = cfg.seeds
        cfg.seeds = list(cfg.s8_rho_seeds)
        try:
            run('S8_diat_rho', ns['specs_S8_rho'](), diat, fp, nk_diat)
        finally:
            cfg.seeds = saved
        for t in cfg.s8_pool_T[1:]:
            pooled(t)
    elif stage == 'S8_shd':
        data, fp = ns['get_shd_T'](cfg.s8_shd_T)
        run(f'S8_shd_T{cfg.s8_shd_T}', ns['specs_S8_shd'](), data, fp, nk_shd)
    elif stage == 'S9_lifdiff':
        data = ns['diat_first_difference'](diat)
        run(stage, ns['specs_S9_lifdiff'](), data, ns['fingerprint'](*data), nk_diat)
    elif stage == 'S9_initfix':
        run('S9_initfix_diat', ns['specs_S9_initfix']('radar'), diat, fp, ns['nk_for'](cfg.stft_window_diat))
        data, fp = ns['get_shd']()
        run('S9_initfix_shd', ns['specs_S9_initfix']('audio'), data, fp, ns['nk_for'](cfg.stft_window_shd))
    elif stage == 'S9_shd_rank':
        data, fp = ns['get_shd']()
        run(stage, ns['specs_S9_shd_rank'](), data, fp, ns['nk_for'](cfg.stft_window_shd))
    else:
        raise ValueError(stage)
