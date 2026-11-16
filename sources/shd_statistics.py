"""Audited SHD numerical contrasts; no training or plotting."""
import json, hashlib
import numpy as np
from scipy import stats
from sklearn.metrics import f1_score, accuracy_score
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def contrast(name, values):
    values = np.asarray(values, float)
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(len(values)))
    if se:
        ci = stats.t.interval(0.95, len(values) - 1, loc=mean, scale=se)
        p = float(stats.ttest_1samp(values, 0).pvalue)
    else:
        ci = (mean, mean)
        p = 1.0 if mean == 0 else 0.0
    return {'name': name, 'seed_differences': values.tolist(), 'mean': mean, 'ci95': list(map(float, ci)), 'p_raw': p, 'n_seeds': len(values), 'ci95_scope': 'Unadjusted descriptive t interval; normal seed differences assumed.'}

def analyze(args, ROOT):
    contract = json.loads((ROOT / 'configs/shd.json').read_text())
    chash = sha(ROOT / 'configs/shd.json')
    files = list(args.results.rglob('result.json'))
    expected = {f'{m}_{c['name']}_s{s}' for s in contract['seeds'] for m in contract['models'] for c in contract['conditions']}
    rows = {}
    checks = []
    truth = None
    dh = None
    for path in files:
        r = json.loads(path.read_text())
        job = r['job']
        assert job in expected and job not in rows
        assert r['contract_sha256'] == chash and r['status'] == 'COMPLETED' and r['selected_replay_exact']
        assert r['environment'] == contract['environment']
        assert len(r['trues']) == len(r['preds']) == 2264
        assert all((type(x) == int and 0 <= x < 20 for x in r['preds']))
        if truth is None:
            truth = r['trues']
            dh = r['data_hashes']
        assert truth == r['trues'] and dh == r['data_hashes']
        f = float(f1_score(r['trues'], r['preds'], labels=np.arange(20), average='macro', zero_division=0))
        assert abs(f - r['test_macro_f1']) < 1e-12 and abs(accuracy_score(r['trues'], r['preds']) - r['test_accuracy']) < 1e-12
        assert sha(path.parent / 'selected.pt') == r['selected_sha256']
        hist = r['validation_history']
        assert len(hist) == r['epochs'] == len(r['minibatch_permutation_sha256'])
        best = -1.0
        selected = -1
        patience = 0
        for ep in range(len(hist)):
            v = float(np.mean(hist[max(0, ep - 4):ep + 1]))
            if v > best + 0.0001:
                best, selected, patience = (v, ep, 0)
            else:
                patience += 1
            assert not (patience >= 20 and ep < len(hist) - 1)
        assert selected == r['selected_epoch_zero_based'] and (len(hist) == 200 or patience == 20)
        rows[job] = r
        checks.append({'job': job, 'result_sha256': sha(path), 'selected_sha256': r['selected_sha256'], 'selection_valid': True, 'metrics_valid': True})
    assert set(rows) == expected, f'Missing jobs: {sorted(expected - set(rows))}'
    for s in contract['seeds']:
        complete = json.loads(next((p for p in args.results.rglob('complete.json') if json.loads(p.read_text()).get('seed') == s)).read_text())
        assert complete['status'] == 'COMPLETED' and complete['contract_sha256'] == chash
        rr = [r for r in rows.values() if r['seed'] == s]
        assert set(complete['jobs']) == {r['job'] for r in rr}
        for m in contract['models']:
            mm = [r for r in rr if r['model'] == m]
            assert len({r['initial_state_sha256'] for r in mm}) == 1
            assert all((r['initial_thresholds'] == mm[0]['initial_thresholds'] for r in mm))
        for ep in range(max((r['epochs'] for r in rr))):
            assert len({r['minibatch_permutation_sha256'][ep] for r in rr if r['epochs'] > ep}) == 1
    seeds = contract['seeds']
    conditions = [c['name'] for c in contract['conditions']]

    def vals(m, c):
        return np.array([rows[f'{m}_{c}_s{s}']['test_macro_f1'] for s in seeds])
    gap = {c: vals('ei_stable', c) - vals('doppler', c) for c in conditions}
    primary = [contrast('EI_minus_Doppler_' + c, gap[c]) for c in conditions[:4]]

    def effects(v, prefix):
        f, c, ff, cc = [v[k] for k in conditions[:4]]
        return [contrast(prefix + 'fine_minus_coarse_N48', f - c), contrast(prefix + 'fine_minus_coarse_N96', ff - cc), contrast(prefix + 'N96_minus_N48_fine', ff - f), contrast(prefix + 'N96_minus_N48_coarse', cc - c), contrast(prefix + 'interaction', ff - cc - (f - c)), contrast(prefix + 'shuffled_minus_fine_N48', v[conditions[4]] - f)]
    primary += effects(gap, 'gap_')
    assert len(primary) == 10
    order = sorted(range(10), key=lambda i: primary[i]['p_raw'])
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (10 - rank) * primary[i]['p_raw'])
        primary[i]['p_holm'] = min(1.0, running)
    secondary = []
    for m in contract['models']:
        secondary += effects({c: vals(m, c) for c in conditions}, m + '_')
    groups = [{'model': m, 'condition': c, 'seeds': seeds, 'values': vals(m, c).tolist(), 'mean': float(vals(m, c).mean()), 'sample_sd': float(vals(m, c).std(ddof=1))} for m in contract['models'] for c in conditions]
    final = {'status': 'ALL_30_VERIFIED', 'contract_sha256': chash, 'groups': groups, 'primary_contrasts': primary, 'secondary_descriptive_contrasts': secondary, 'checks': checks, 'limitations': ['Discrete update-count intervention with fixed coefficients; not a continuous-time-invariant model.', 'Coarsening changes temporal placement and count amplitudes; the shuffle arm separately holds count multisets fixed.', 'Common reference thresholds are deliberately held fixed; not a best-recalibrated attainable-performance comparison.', 'Three training seeds, one fixed test set; t assumptions and limited power.', 'No automatic primary-driver, universal ranking or equivalence claim. Old cohorts are not pooled.']}
    return final
