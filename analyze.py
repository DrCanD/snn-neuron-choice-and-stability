"""Numerical analyses only. Never train or modify input result files."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import runpy
import shutil
import tempfile
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def paired(a_files, b_files, classes):
    import numpy as np
    from scipy import stats
    from sklearn.metrics import f1_score
    def group(files):
        out = {}
        for path in files:
            r = read(path)
            seed = r.get('seed', r.get('spec', {}).get('seed'))
            if seed is None or seed in out:
                raise ValueError('Each group must contain exactly one result per seed.')
            score = r.get('best', r)
            if 'preds' not in score or 'trues' not in score:
                raise ValueError(f'Predictions and true labels are required: {path}')
            value = float(f1_score(score['trues'], score['preds'], labels=np.arange(classes), average='macro', zero_division=0))
            out[seed] = (value, score['trues'])
        return out
    a, b = group(a_files), group(b_files)
    if set(a) != set(b) or len(a) < 3 or len(a) > 20:
        raise ValueError('Provide matching sets of 3–20 seeds; incomplete pairs are rejected.')
    seeds = sorted(a)
    if any(a[s][1] != b[s][1] for s in seeds) or any(a[s][1] != a[seeds[0]][1] for s in seeds):
        raise ValueError('Test labels/order differ; the supplied records are not paired.')
    d = np.array([a[s][0] - b[s][0] for s in seeds])
    mean, sd = float(d.mean()), float(d.std(ddof=1))
    se, df = sd / np.sqrt(len(d)), len(d)-1
    if not np.isfinite(d).all() or se == 0:
        raise ValueError('The t-based analysis requires finite, nonzero seed variance.')
    ci = {str(level): list(map(float, stats.t.interval(level, df, loc=mean, scale=se))) for level in (.90, .95)}
    tests = []
    for margin in (.01, .02, .03, .04):
        lp = float(stats.t.sf((mean + margin)/se, df))
        up = float(stats.t.cdf((mean - margin)/se, df))
        tests.append(dict(margin=margin, p_lower=lp, p_upper=up, p_tost=max(lp, up), nominal_pass=max(lp, up)<.05, confirmatory_equivalence=False))
    signs = np.asarray(list(itertools.product((-1, 1), repeat=len(d))))
    return dict(seeds=seeds, a_values=[a[s][0] for s in seeds], b_values=[b[s][0] for s in seeds],
                difference_direction='A minus B', differences=d.tolist(), mean=mean, sample_sd=sd,
                standard_error=float(se), ci=ci, tost=tests,
                paired_difference_p=float(stats.ttest_1samp(d, 0).pvalue),
                exact_signflip_p=float(np.mean(np.abs((signs*d).mean(1)) >= abs(mean)-1e-14)),
                inference='Exploratory bounds; fixed test set; pairing by seed does not establish identical RNG/minibatch trajectories. Supply one protocol cohort at a time. Do not choose a passing margin after seeing results.')


def supplementary(results, data):
    # A private temporary workspace keeps even legacy bookkeeping away from inputs.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    from runtime import sources
    from run import data_checks
    data_checks(argparse.Namespace(experiment='S1_cap_sweep', data=data, shd_data=None, data_sha256=None))
    with tempfile.TemporaryDirectory(prefix='snn-analysis-') as temporary:
        os.environ.update(RESEARCH_ROOT=temporary, DIAT_DATA=str(data), STUDY_SHD_DATA=temporary)
        os.environ.pop('SMOKE', None)
        repo, _, _ = sources()
        ns = runpy.run_path(str(repo/'experiments/supplementary.py'))
        scope = ns['get_diat'].__globals__
        stats_file = repo/'experiments/supplementary_statistics.py'
        exec(compile(stats_file.read_text(encoding='utf-8'), str(stats_file), 'exec'), scope)
        keys = set()
        stages = set()
        for root in results:
            for path in root.rglob('results_*.json'):
                r = read(path)
                if r.get('ref_hash') != scope['REF_HASH'] or r.get('smoke') or r.get('run_version') != scope['RUN_VERSION']:
                    raise ValueError(f'Result protocol differs from the packaged supplementary source: {path}')
                key = (r['stage'], r['method'], r['seed'])
                if key in keys:
                    raise ValueError(f'Duplicate result: {key}')
                keys.add(key); stages.add(r['stage'])
                target = scope['EXP_ROOT']/f"{r['stage']}_v{r['run_version']}_{r['data_fp']}"
                target.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target/path.name)
        required = {'S1_cap_sweep','S2_regime','S3_rescue','S4_shd','S6_seeds_diat','S6_seeds_shd',
                    'S7_signal_diat','S7_signal_shd','S8_diat_T16','S8_diat_T8','S8_diat_T32','S8_diat_rho',
                    'S8_shd_T64','S9_lifdiff','S9_initfix_diat','S9_initfix_shd','S9_shd_rank'}
        if stages != required or len(keys) != 195:
            raise ValueError(f'All 195 supplementary runs are required; found {len(keys)}. Missing stages: {sorted(required-stages)}')
        diat, fp = scope['get_diat']()
        candidates = list(scope['EXP_ROOT'].glob('S4_shd_v1_*'))
        if len(candidates) != 1:
            raise ValueError('SHD supplementary results have inconsistent data fingerprints.')
        report = scope['stage_S5'](diat, fp, candidates[0].name.rsplit('_',1)[1], {'STFT-IF': {'window_len': 16}})
        if not report['complete']:
            raise ValueError('Supplementary stage audit failed; no complete result written.')
        return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('analysis', choices=['paired', 'shd', 'supplementary'])
    p.add_argument('--results', nargs='+', type=Path)
    p.add_argument('--data', type=Path, help='Processed DIAT directory for supplementary statistics')
    p.add_argument('--a', nargs='+', type=Path)
    p.add_argument('--b', nargs='+', type=Path)
    p.add_argument('--classes', type=int, choices=[6, 20])
    p.add_argument('--output', type=Path, required=True, help='New JSON file; existing files are rejected')
    args = p.parse_args()
    from runtime import verify_package
    verify_package()
    if args.output.exists():
        p.error('Output exists; choose a new JSON file.')
    paths = (args.a or []) + (args.b or []) + (args.results or []) + ([args.data] if args.data else [])
    if any(args.output.resolve() == x.resolve() or (x.is_dir() and x.resolve() in args.output.resolve().parents) for x in paths):
        p.error('Analysis output must be outside the input folders.')
    if args.analysis == 'paired':
        if not args.a or not args.b or not args.classes:
            p.error('paired requires --a, --b and --classes.')
        result = paired(args.a, args.b, args.classes)
    elif args.analysis == 'shd':
        if not args.results or len(args.results) != 1:
            p.error('shd requires one --results parent containing all three seed outputs.')
        ns = runpy.run_path(str(ROOT/'sources/shd_statistics.py'))
        result = ns['analyze'](argparse.Namespace(results=args.results[0]), ROOT)
    else:
        if not args.results or not args.data:
            p.error('supplementary requires --results and --data.')
        result = supplementary(args.results, args.data.resolve())
    result['training_started'] = False
    result['input_sha256'] = {str(x): hashlib.sha256(x.read_bytes()).hexdigest() for x in (args.a or [])+(args.b or [])}
    # Exclusive creation also protects against another process creating the file.
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(str(args.output))


if __name__ == '__main__':
    main()
