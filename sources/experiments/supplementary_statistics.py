# Numerical S5 analysis from the same archive. Executed with the supplementary namespace.
def load_stage_results(stage, data_fp):
    d = EXP_ROOT / f'{stage}_v{RUN_VERSION}_{data_fp}'
    out = {}
    for p in sorted(d.glob('results_*.json')):
        r = read_json_safe(p)
        if r and r.get('run_version') == RUN_VERSION and (r.get('ref_hash') == REF_HASH) and (bool(r.get('smoke')) == cfg.smoke):
            out.setdefault(r['method'], []).append(r)
    for k in out:
        out[k].sort(key=lambda r: cfg.seeds.index(r['seed']) if r['seed'] in cfg.seeds else 99)
    return out

def completeness_audit(diat_fp, shd_fp, Fin, C, NK_diat):
    """Expected (method, seed) pairs per stage, from the SAME spec functions that launched the runs, against the
    result files actually on disk. S5 used to skip missing files silently; a stage split across two Colab sessions
    once landed in two same-named Drive folders, and only one of them is visible through the mount. This audit is
    what makes such a gap visible instead of turning into a half-empty table."""
    plan = [('S3_rescue', diat_fp, list(specs_S3()), cfg.seeds), ('S1_cap_sweep', diat_fp, list(specs_S1()), cfg.seeds), ('S2_regime', diat_fp, list(specs_S2(Fin, C, NK_diat)), cfg.seeds), ('S6_seeds_diat', diat_fp, list(specs_2x2('x_')), cfg.extra_seeds)]
    if shd_fp:
        plan += [('S4_shd', shd_fp, list(specs_S4()), cfg.seeds), ('S6_seeds_shd', shd_fp, list(specs_2x2('x_shd_')), cfg.extra_seeds)]
    for st in ('S7_signal_diat', 'S7_signal_shd'):
        cands = sorted(EXP_ROOT.glob(f'{st}_v{RUN_VERSION}_*'))
        plan.append((st, cands[-1].name.split('_')[-1] if cands else None, list(specs_2x2('m_')), cfg.seeds))
    for d in sorted(EXP_ROOT.glob(f'S8_*_v{RUN_VERSION}_*')):
        st, fp = d.name.rsplit(f'_v{RUN_VERSION}_', 1)
        if st == 'S8_diat_rho':
            plan.append((st, fp, list(specs_S8_rho()), cfg.s8_rho_seeds))
        elif st.startswith('S8_diat_T'):
            plan.append((st, fp, list(specs_S8_pool(int(st[len('S8_diat_T'):]))), cfg.seeds))
        elif st.startswith('S8_shd_T'):
            plan.append((st, fp, list(specs_S8_shd()), cfg.seeds))
    s9_specs = {'S9_lifdiff': specs_S9_lifdiff, 'S9_initfix_diat': lambda: specs_S9_initfix('radar'), 'S9_initfix_shd': lambda: specs_S9_initfix('audio'), 'S9_shd_rank': specs_S9_shd_rank}
    for d in sorted(EXP_ROOT.glob(f'S9_*_v{RUN_VERSION}_*')):
        st, fp = d.name.rsplit(f'_v{RUN_VERSION}_', 1)
        if st in s9_specs:
            plan.append((st, fp, list(s9_specs[st]()), cfg.seeds))
    rows, missing_all = ([], {})
    pass
    for st, fp, methods, seeds in plan:
        found = load_stage_results(st, fp) if fp else {}
        have = {(k, r['seed']) for k, rs in found.items() for r in rs}
        want = [(m, sd) for m in methods for sd in seeds]
        miss = [f'{m}/seed{sd}' for m, sd in want if (m, sd) not in have]
        extra = sorted((f'{k}/seed{sd}' for k, sd in have if k not in methods or sd not in seeds))
        mark = 'ok' if not miss else 'MISSING'
        pass
        rows.append({'stage': st, 'data_fp': fp, 'expected': len(want), 'found': len(want) - len(miss), 'missing': miss})
        if miss:
            missing_all[st] = miss
    if missing_all:
        n = sum((len(v) for v in missing_all.values()))
        pass
        pass
        pass
        pass
        pass
    return {'rows': rows, 'missing': missing_all, 'complete': not missing_all}

def boot_pair(ra, rb, C, label):
    """paired bootstrap per seed, exactly the paper's criterion (sign consistent across all seeds)."""
    if not ra or not rb or len(ra) != len(rb):
        return None
    out = {}
    for mname, mfn in [('F1', METRICS['macro_f1']), ('acc', METRICS['accuracy'])]:
        per_seed = [paired_bootstrap(np.array(a['preds']), np.array(b['preds']), np.array(a['trues']), mfn, C, nboot=cfg.nboot, seed=a['seed']) for a, b in zip(ra, rb)]
        diffs = [d for d, _, _ in per_seed]
        sig = all((lo > 0 for _, lo, _ in per_seed)) or all((hi < 0 for _, _, hi in per_seed))
        pass
        out[mname] = {'per_seed': per_seed, 'mean_diff': float(np.mean(diffs)), 'consistent_sign': bool(sig)}
    return out

def agg_f1(rs):
    f = [r['f1'] for r in rs]
    return (float(np.mean(f)), float(np.std(f)), f)

def merge_by_seed(*lists):
    """union of result lists keyed by seed, in the order [paper seeds..., extra seeds...]"""
    by = {}
    for rs in lists:
        for r in rs or []:
            by.setdefault(r['seed'], r)
    order = list(cfg.seeds) + [sd for sd in cfg.extra_seeds if sd not in cfg.seeds]
    return [by[sd] for sd in order if sd in by]

def align_pairs(ra, rb):
    """keep only seeds present on both sides, same order"""
    sa = {r['seed']: r for r in ra or []}
    sb = {r['seed']: r for r in rb or []}
    common = [sd for sd in list(cfg.seeds) + list(cfg.extra_seeds) if sd in sa and sd in sb]
    return ([sa[sd] for sd in common], [sb[sd] for sd in common])

def load_stage_results_any(stage_prefix):
    """results of a stage whose data fingerprint is not known here (S7 transformed inputs): newest matching dir"""
    cands = sorted(EXP_ROOT.glob(f'{stage_prefix}_v{RUN_VERSION}_*'))
    if not cands:
        return {}
    return load_stage_results(stage_prefix, cands[-1].name.split('_')[-1])

def fmt_p(p, floor):
    """a permutation p has a floor (1/n! for an exact Spearman test); never print below it"""
    if p <= floor * 1.01:
        return f'p <= {floor:.2e} (floor at this n)'
    return f'p = {p:.2e}' if p < 0.0001 else f'p = {p:.4f}'

def spearman_exact(a, b):
    """Spearman rho (average ranks for ties) with an EXACT permutation p over all n! orderings (n <= 9)"""
    from scipy.stats import rankdata
    from itertools import permutations
    ra, rb = (rankdata(a), rankdata(b))
    n = len(ra)

    def rho(x, Y):
        x = x - x.mean()
        Y = Y - Y.mean(axis=-1, keepdims=True)
        return Y @ x / (np.sqrt((Y * Y).sum(-1)) * np.sqrt(x @ x))
    obs = float(rho(ra, rb[None, :])[0])
    if n > 9:
        return {'rho': obs, 'n': n, 'p_one_sided': None, 'p_two_sided': None, 'floor': None}
    P = np.array(list(permutations(range(n))))
    perm = rho(ra, rb[P])
    return {'rho': obs, 'n': n, 'p_one_sided': float(np.mean(perm >= obs - 1e-12)), 'p_two_sided': float(np.mean(np.abs(perm) >= abs(obs) - 1e-12)), 'floor': 1.0 / len(P)}

def learned_init_summary(rs):
    """median effective values per layer from the seed-0 diagnostics (last trained epoch): Dual-tau leaks and mix, STFT band centre"""
    if not rs:
        return None
    sig = lambda v: 1.0 / (1.0 + np.exp(-np.asarray(v, float)))
    out = []
    for l in rs[0]['diagnostics']:
        pr, e = (l['params'], {})
        if 'log_beta_fast' in pr:
            e = {'beta_fast_median': round(float(np.median(sig(pr['log_beta_fast']))), 4), 'beta_slow_median': round(float(np.median(sig(pr['log_beta_slow']))), 4), 'mix_median': round(float(np.median(sig(pr['mix']))), 4)}
        elif 'band_center' in pr:
            fc = sig(pr['band_center'])
            e = {'band_centre_median': round(float(np.median(fc)), 4), 'frac_band_centre_below_nyquist': round(float((fc < 0.5).mean()), 4)}
        out.append(e)
    return out

def stage_S5(diat, diat_fp, shd_fp, NK_diat):
    pass
    Xtr, ytr, Xva, yva, Xte, yte = diat
    Fin, C = (Xtr.shape[-1], int(max(ytr.max(), yva.max(), yte.max()) + 1))
    S1, S2, S3 = (load_stage_results('S1_cap_sweep', diat_fp), load_stage_results('S2_regime', diat_fp), load_stage_results('S3_rescue', diat_fp))
    S4 = load_stage_results('S4_shd', shd_fp) if shd_fp else {}
    audit = completeness_audit(diat_fp, shd_fp, Fin, C, NK_diat)
    rep = {'analysis_version': ANALYSIS_VERSION, 'run_version': RUN_VERSION, 'ref_hash': REF_HASH, 'smoke': cfg.smoke, 'ntm_sha': NTM_SHA, 'dsn_sha': DSN_SHA, 'provenance': PROVENANCE, 'timestamp': datetime.now().isoformat(), 'complete': audit['complete'], 'completeness': audit}
    Wr = 128 if not cfg.smoke else cfg.W
    d1 = count_params(VerticalNet(Fin, C, list(H1_ORDER), Wr, NK_diat, cfg.norm))
    rows = []
    for name in ['Doppler-LIF', 'STFT-IF', 'Chirp-LIF', 'Dual-tau-LIF', 'CrossInhib-LIF', 'Phase-LIF', 'Beam-IF', 'Vanilla-LIF', 'Parametric-LIF']:
        m = VerticalNet(Fin, C, [name] * cfg.L, Wr, NK_diat, cfg.norm)
        rows.append({'variant': f'homogeneous {name}', 'width': Wr, 'params': count_params(m), 'units': count_units(m)})
    for tag, order in [('vertical H1', H1_ORDER)]:
        m = VerticalNet(Fin, C, list(order), Wr, NK_diat, cfg.norm)
        rows.append({'variant': tag, 'width': Wr, 'params': count_params(m), 'units': count_units(m)})
    w, p = solve_bank_width(Fin, C, NK_diat, d1)
    m = PathBankNet(Fin, C, list(D2_PATHS), w, cfg.L, 'concat', NK_diat, cfg.norm, True)
    rows.append({'variant': f'horizontal bank K={len(D2_PATHS)} (param-matched)', 'width': w, 'params': count_params(m), 'units': count_units(m)})
    for r in rows:
        r['params_rel_to_D1_pct'] = round(100 * (r['params'] - d1) / d1, 2)
    per_channel = {}
    for name in ['Doppler-LIF', 'Chirp-LIF', 'Dual-tau-LIF', 'STFT-IF', 'Phase-LIF', 'Beam-IF', 'CrossInhib-LIF', 'Vanilla-LIF', 'Parametric-LIF']:
        n = NeuronRegistry.create(name, size=8, **NK_diat.get(name, {}))
        per_channel[name] = {pn: int(p.numel() // 8) for pn, p in n.named_parameters()}
    rep['capacity_table'] = {'reference_D1_params': d1, 'rows': rows, 'per_channel_params': per_channel, 'formula': 'P = (F*W + W) + (L-1)*(W^2 + W) + W*sum_l k_l + (W*C + C); bank: stem (F*w+w) + K*[L*(w^2+w) + L*k*w] + (K*w*C + C)'}
    pass
    for r in rows:
        pass
    if S1:
        sweep = {}
        for k, rs in S1.items():
            m, s, f = agg_f1(rs)
            sweep[k] = {'f1_mean': m, 'f1_std': s, 'f1s': f, 'params': rs[0]['spec']['params'], 'W': rs[0]['spec']['W'], 'neuron': rs[0]['spec']['neuron'], 'n': len(rs)}
        rep['S1_cap_sweep'] = sweep
        pass
        for W in cfg.sweep_widths:
            line = f'   W={W:>4d}: '
            for tag in ['vanilla', 'plif', 'doppler']:
                k = f'cap_{tag}_W{W}'
                if k in sweep:
                    line += f'{tag:8s} {sweep[k]['f1_mean']:.3f}+/-{sweep[k]['f1_std']:.3f} ({sweep[k]['params']:,})   '
            pass
        Wc = 128 if 128 in cfg.sweep_widths else cfg.sweep_widths[0]
        conf = {}
        for tag in ['vanilla', 'plif', 'doppler']:
            k = f'cap_{tag}_W{Wc}'
            if k in S1:
                cm = np.zeros((C, C), int)
                for r in S1[k]:
                    for t, p in zip(r['trues'], r['preds']):
                        cm[t, p] += 1
                conf[k] = cm.tolist()
                names = DIAT_CLASS_NAMES if len(DIAT_CLASS_NAMES) == C and (not cfg.smoke) else [str(i) for i in range(C)]
                pass
                for ci, row in enumerate(cm):
                    pass
        rep['S1_confusion_W%d' % Wc] = {'class_names': DIAT_CLASS_NAMES if len(DIAT_CLASS_NAMES) == C else None, 'matrices': conf}
        rep['S1_bootstrap'] = {}
        for W in cfg.sweep_widths:
            a, b = (S1.get(f'cap_doppler_W{W}'), S1.get(f'cap_vanilla_W{W}'))
            bp = boot_pair(a, b, C, f'Doppler@{W} - Vanilla@{W}')
            if bp:
                rep['S1_bootstrap'][f'doppler_vs_vanilla_W{W}'] = bp
        Wmax = max(cfg.sweep_widths)
        bp = boot_pair(S1.get(f'cap_doppler_W{Wc}'), S1.get(f'cap_vanilla_W{Wmax}'), C, f'Doppler@{Wc} - Vanilla@{Wmax}')
        if bp:
            rep['S1_bootstrap'][f'doppler_W{Wc}_vs_vanilla_W{Wmax}'] = bp
    if S2:
        reg = {}
        for k, rs in S2.items():
            m, s, f = agg_f1(rs)
            reg[k] = {'f1_mean': m, 'f1_std': s, 'f1s': f, 'params': rs[0]['spec']['params'], 'units': rs[0]['spec']['units'], 'n': len(rs)}
        rep['S2_regime'] = reg
        rep['S2_bootstrap'] = {}
        pass
        for k, v in reg.items():
            pass
        for W in cfg.regime_widths:
            for other in [f'D1_H1_W{W}', f'D2_concat_W{W}']:
                bp = boot_pair(S2.get(other), S2.get(f'D0_doppler_W{W}'), C, f'{other} - D0_doppler_W{W}')
                if bp:
                    rep['S2_bootstrap'][f'{other}_vs_D0_doppler_W{W}'] = bp
        ku = f'D2_units_W{cfg.W}'
        kd = f'cap_doppler_W{cfg.W}'
        bp = boot_pair(S2.get(ku), S1.get(kd), C, f'{ku} - {kd} (S1)')
        if bp:
            rep['S2_bootstrap'][f'{ku}_vs_{kd}'] = bp
        ks = f'D2_concat_stable_W{cfg.W}'
        for ref_name, ref_rs in [(f'{kd} (S1)', S1.get(kd)), ('CI_stable (S3)', S3.get('CI_stable'))]:
            bp = boot_pair(S2.get(ks), ref_rs, C, f'{ks} - {ref_name}')
            if bp:
                rep['S2_bootstrap'][f'{ks}_vs_{ref_name.split()[0]}'] = bp
    if S3:
        res = {}
        for k, rs in S3.items():
            m, s, f = agg_f1(rs)
            dg = [r['diagnostics'] for r in rs]
            res[k] = {'f1_mean': m, 'f1_std': s, 'f1s': f, 'best_epochs': [r['best_epoch'] for r in rs], 'const_predictor_macro_f1_range': [min(rs[0]['const_predictor_macro_f1'].values()), max(rs[0]['const_predictor_macro_f1'].values())], 'layer_rate_last_epoch': [[round(l['rate'], 4) for l in d] for d in dg], 'dead_unit_frac': [[round(l['dead_unit_frac'], 3) for l in d] for d in dg], 'silent_after_first_frac': [[round(l['silent_after_first_frac'], 3) for l in d] for d in dg], 'first_spike_t_mean': [[l['first_spike_t_mean'] for l in d] for d in dg], 'ei': [[l.get('ei') for l in d] for d in dg], 'diagnostics_calibrated': [bool(r.get('diagnostics_meta', {}).get('calibrated')) for r in rs]}
        rep['S3_rescue'] = res
        pass
        for k, v in res.items():
            pass
            if v['ei'][0][0]:
                pass
        rep['S3_bootstrap'] = {}
        kd = f'cap_doppler_W{cfg.W}'
        for k in ['CI_default', 'CI_stable', 'CI_lowlr']:
            bp = boot_pair(S1.get(kd), S3.get(k), C, f'{kd} (S1) - {k}')
            if bp:
                rep['S3_bootstrap'][f'{kd}_vs_{k}'] = bp
        bp = boot_pair(S3.get('CI_stable'), S3.get('CI_default'), C, 'CI_stable - CI_default')
        if bp:
            rep['S3_bootstrap']['CI_stable_vs_CI_default'] = bp
    if S4:
        res = {}
        for k, rs in S4.items():
            m, s, f = agg_f1(rs)
            dg = [r['diagnostics'] for r in rs]
            res[k] = {'f1_mean': m, 'f1_std': s, 'f1s': f, 'layer_rate_last_epoch': [[round(l['rate'], 4) for l in d] for d in dg], 'silent_after_first_frac': [[round(l['silent_after_first_frac'], 3) for l in d] for d in dg], 'ei': [[l.get('ei') for l in d] for d in dg], 'doppler': [[l.get('doppler') for l in d] for d in dg]}
        rep['S4_shd'] = res
        pass
        for k, v in res.items():
            pass
        C_shd = None
        a, b = (S4.get('shd_CI_default'), S4.get('shd_doppler'))
        if a and b:
            C_shd = int(max((max(r['trues']) for r in a)) + 1)
            rep['S4_bootstrap'] = {'CI_default_vs_doppler': boot_pair(a, b, C_shd, 'shd CI_default - Doppler'), 'CI_stable_vs_doppler': boot_pair(S4.get('shd_CI_stable'), b, C_shd, 'shd CI_stable - Doppler')}
        sig = {}
        if S1.get(f'cap_doppler_W{cfg.W}') and S4.get('shd_doppler'):
            sig['doppler_f_center_q50_radar'] = [[l['doppler']['f_center_quantiles'][2] for l in r['diagnostics']] for r in S1[f'cap_doppler_W{cfg.W}']]
            sig['doppler_f_center_q50_audio'] = [[l['doppler']['f_center_quantiles'][2] for l in r['diagnostics']] for r in S4['shd_doppler']]
            sig['doppler_dz_q50_radar'] = [[l['dz_abs_quantiles'][2] for l in r['diagnostics']] for r in S1[f'cap_doppler_W{cfg.W}']]
            sig['doppler_dz_q50_audio'] = [[l['dz_abs_quantiles'][2] for l in r['diagnostics']] for r in S4['shd_doppler']]
        if S3.get('CI_default') and S4.get('shd_CI_default'):
            sig['ei_rho_radar'] = [[l['ei']['rho_mean'] for l in r['diagnostics']] for r in S3['CI_default']]
            sig['ei_rho_audio'] = [[l['ei']['rho_mean'] for l in r['diagnostics']] for r in S4['shd_CI_default']]
        rep['cross_domain_signature'] = sig
    S6d, S6s = (load_stage_results('S6_seeds_diat', diat_fp), load_stage_results('S6_seeds_shd', shd_fp) if shd_fp else {})
    if S6d or S6s:
        cells = {'radar_doppler': merge_by_seed(S1.get(f'cap_doppler_W{cfg.W}'), S6d.get('x_doppler')), 'radar_CI_default': merge_by_seed(S3.get('CI_default'), S6d.get('x_CI_default')), 'radar_CI_stable': merge_by_seed(S3.get('CI_stable'), S6d.get('x_CI_stable')), 'audio_doppler': merge_by_seed(S4.get('shd_doppler'), S6s.get('x_shd_doppler')), 'audio_CI_default': merge_by_seed(S4.get('shd_CI_default'), S6s.get('x_shd_CI_default')), 'audio_CI_stable': merge_by_seed(S4.get('shd_CI_stable'), S6s.get('x_shd_CI_stable'))}
        five = {k: {'n_seeds': len(v), 'seeds': [r['seed'] for r in v], 'f1_mean': float(np.mean([r['f1'] for r in v])) if v else None, 'f1_std': float(np.std([r['f1'] for r in v])) if v else None, 'f1s': [r['f1'] for r in v]} for k, v in cells.items()}
        pass
        for k, v in five.items():
            if v['n_seeds']:
                pass
        boots = {}
        for dom, C_dom in [('radar', C), ('audio', int(max((max(r['trues']) for r in cells['audio_doppler'])) + 1) if cells['audio_doppler'] else None)]:
            if C_dom is None:
                continue
            for arm in ['CI_default', 'CI_stable']:
                a, b = align_pairs(cells[f'{dom}_doppler'], cells[f'{dom}_{arm}'])
                bp = boot_pair(a, b, C_dom, f'{dom}: Doppler - {arm} ({len(a)} seeds)')
                if bp:
                    boots[f'{dom}_doppler_minus_{arm}'] = {**bp, 'n_seeds': len(a)}
        rep['S6_five_seed_2x2'] = {'cells': five, 'bootstrap': boots}
    S7d, S7s = (load_stage_results_any('S7_signal_diat'), load_stage_results_any('S7_signal_shd'))
    if S7d or S7s:
        sig7 = {}
        pass
        for dom, Sm, Sorig, C_dom in [('radar_first_difference', S7d, {'doppler': S1.get(f'cap_doppler_W{cfg.W}'), 'CI_default': S3.get('CI_default'), 'CI_stable': S3.get('CI_stable')}, C), ('audio_cumulative', S7s, {'doppler': S4.get('shd_doppler'), 'CI_default': S4.get('shd_CI_default'), 'CI_stable': S4.get('shd_CI_stable')}, None)]:
            if not Sm:
                continue
            if C_dom is None:
                C_dom = int(max((max(r['trues']) for rs in Sm.values() for r in rs)) + 1)
            entry = {}
            for arm in ['doppler', 'CI_default', 'CI_stable']:
                rs = Sm.get(f'm_{arm}')
                if rs:
                    m, sd_, f = agg_f1(rs)
                    entry[arm] = {'f1_mean': m, 'f1_std': sd_, 'f1s': f, 'silent_after_first_frac_seed0': [round(l['silent_after_first_frac'], 3) for l in rs[0]['diagnostics']], 'layer_rate_seed0': [round(l['rate'], 4) for l in rs[0]['diagnostics']]}
                    o = Sorig.get(arm)
                    if o:
                        entry[arm]['original_f1_mean'] = float(np.mean([r['f1'] for r in o]))
                    pass
            for arm in ['CI_default', 'CI_stable']:
                a, b = align_pairs(Sm.get('m_doppler'), Sm.get(f'm_{arm}'))
                bp = boot_pair(a, b, C_dom, f'{dom}: Doppler - {arm}')
                if bp:
                    entry[f'bootstrap_doppler_minus_{arm}'] = bp
            sig7[dom] = entry
        rep['S7_signal'] = sig7

    def _stage_any(prefix):
        out = {}
        for d in sorted(EXP_ROOT.glob(f'{prefix}_v{RUN_VERSION}_*')):
            out.update(load_stage_results(prefix, d.name.rsplit(f'_v{RUN_VERSION}_', 1)[1]))
        return out
    S8 = {}
    for d in sorted(EXP_ROOT.glob(f'S8_*_v{RUN_VERSION}_*')):
        st = d.name.rsplit(f'_v{RUN_VERSION}_', 1)[0]
        S8[st] = _stage_any(st)
    if any(S8.values()):
        S6d_ = load_stage_results('S6_seeds_diat', diat_fp)
        S6s_ = load_stage_results('S6_seeds_shd', shd_fp) if shd_fp else {}
        conds = [('radar', 64, 'CrossInhib-LIF', 'paper runs (S3+S6)', merge_by_seed(S3.get('CI_default'), S6d_.get('x_CI_default'))), ('radar', 64, 'CrossInhib-LIF-stable', 'paper runs (S3+S6)', merge_by_seed(S3.get('CI_stable'), S6d_.get('x_CI_stable'))), ('audio', 12, 'CrossInhib-LIF', 'paper runs (S4+S6)', merge_by_seed(S4.get('shd_CI_default'), S6s_.get('x_shd_CI_default'))), ('audio', 12, 'CrossInhib-LIF-stable', 'paper runs (S4+S6)', merge_by_seed(S4.get('shd_CI_stable'), S6s_.get('x_shd_CI_stable')))]
        for st, res in S8.items():
            for k, rs in res.items():
                sp_ = rs[0]['spec']
                conds.append((sp_['domain'], sp_['T'], sp_['neuron'], f'{st}/{k}', rs))
        rows8 = []
        pass
        pass
        for dom, T, neuron, src, rs in conds:
            if not rs:
                continue
            f1 = [r['f1'] for r in rs]
            rho = rho_init_of(neuron)
            const = max((max(r['const_predictor_macro_f1'].values()) for r in rs))
            collapsed = float(np.mean(f1)) < 3 * const
            logG = None if rho is None else T * math.log10(rho)
            rows8.append({'domain': dom, 'T': T, 'neuron': neuron, 'source': src, 'rho_init': rho, 'log10_rho_T': logG, 'f1_mean': float(np.mean(f1)), 'f1_std': float(np.std(f1)), 'f1s': f1, 'n': len(f1), 'const_predictor_f1_max': const, 'collapsed': collapsed})
            pass
        ei_rows = [r for r in rows8 if r['log10_rho_T'] is not None]
        tr = [r['log10_rho_T'] for r in ei_rows if not r['collapsed']]
        co = [r['log10_rho_T'] for r in ei_rows if r['collapsed']]
        sep = bool(tr and co and (max(tr) < min(co)))
        verdict = {'rule': 'collapsed := mean macro-F1 < 3 x best constant-predictor macro-F1 of that test set', 'max_log10_rho_T_trained': max(tr) if tr else None, 'min_log10_rho_T_collapsed': min(co) if co else None, 'separable_by_one_threshold': sep}
        pass
        if tr and co:
            pass
        rep['S8_rho_T'] = {'rows': rows8, 'verdict': verdict}
    S9L = load_stage_results_any('S9_lifdiff')
    S9Fd = load_stage_results('S9_initfix_diat', diat_fp)
    S9Fs = load_stage_results('S9_initfix_shd', shd_fp) if shd_fp else {}
    S9R = load_stage_results('S9_shd_rank', shd_fp) if shd_fp else {}
    C_shd9 = None
    for rs in list(S9R.values()) + list(S9Fs.values()) + list(S4.values()):
        if rs:
            C_shd9 = int(max((max(r['trues']) for r in rs)) + 1)
            break

    def _cell(rs):
        if not rs:
            return None
        m, sd_, f = agg_f1(rs)
        return {'f1_mean': m, 'f1_std': sd_, 'f1s': f, 'acc_mean': float(np.mean([r['acc'] for r in rs])), 'n': len(rs), 'params': rs[0]['spec'].get('params'), 'seeds': [r['seed'] for r in rs]}

    def _boot(ra, rb, C_, label):
        a, b = align_pairs(ra, rb)
        return boot_pair(a, b, C_, f'{label} ({len(a)} seeds)') if a else None
    if S9L:
        pass
        kd, kv, kp = (f'cap_doppler_W{cfg.W}', f'cap_vanilla_W{cfg.W}', f'cap_plif_W{cfg.W}')
        rows = {'Vanilla-LIF': (S1.get(kv), S9L.get('d_vanilla')), 'Parametric-LIF': (S1.get(kp), S9L.get('d_plif')), 'Doppler-LIF': (S1.get(kd), S7d.get('m_doppler') if S7d else None)}
        tab = {}
        for nm, (o, d) in rows.items():
            tab[nm] = {'original': _cell(o), 'first_difference': _cell(d)}
            fo = f'{tab[nm]['original']['f1_mean']:.3f}+/-{tab[nm]['original']['f1_std']:.3f}' if o else '   -   '
            fd = f'{tab[nm]['first_difference']['f1_mean']:.3f}+/-{tab[nm]['first_difference']['f1_std']:.3f}' if d else '   -   '
            pass
        boots = {}
        for key_, ra, rb, lab in [('vanilla_diff_minus_orig', S9L.get('d_vanilla'), S1.get(kv), 'Vanilla first-diff - Vanilla original'), ('plif_diff_minus_orig', S9L.get('d_plif'), S1.get(kp), 'Parametric first-diff - Parametric original'), ('doppler_orig_minus_vanilla_diff', S1.get(kd), S9L.get('d_vanilla'), 'Doppler original - Vanilla first-diff'), ('doppler_diff_minus_vanilla_diff', S7d.get('m_doppler') if S7d else None, S9L.get('d_vanilla'), 'Doppler first-diff - Vanilla first-diff')]:
            bp = _boot(ra, rb, C, lab)
            if bp:
                boots[key_] = bp
        rep['S9_lifdiff'] = {'table': tab, 'bootstrap': boots, 'note': 'paired over the same test samples; the first-difference view is invertible (S7)'}
    if S9Fd or S9Fs:
        pass
        fx = {}
        pairs = [('radar', 'Dual-tau-LIF', S9Fd.get('fx_dualtau_fixed'), S9Fd.get('fx_dualtau_default'), C), ('radar', 'STFT-IF', S9Fd.get('fx_stft_fixed'), S3.get('STFT_default'), C), ('audio', 'Dual-tau-LIF', S9Fs.get('fxs_dualtau_fixed'), S9R.get('r_dualtau'), C_shd9), ('audio', 'STFT-IF', S9Fs.get('fxs_stft_fixed'), S9R.get('r_stft'), C_shd9)]
        for dom, nm, fixed, lib, C_ in pairs:
            if not fixed and (not lib):
                continue
            e = {'fixed': _cell(fixed), 'library': _cell(lib), 'learned_fixed_seed0': learned_init_summary(fixed), 'learned_library_seed0': learned_init_summary(lib)}
            ff = f'{e['fixed']['f1_mean']:.3f}+/-{e['fixed']['f1_std']:.3f}' if fixed else '   -   '
            fl = f'{e['library']['f1_mean']:.3f}+/-{e['library']['f1_std']:.3f}' if lib else '   -   '
            pass
            for lab, v in [('library', e['learned_library_seed0']), ('fixed', e['learned_fixed_seed0'])]:
                if v:
                    pass
            if fixed and lib and C_:
                bp = _boot(fixed, lib, C_, f'{dom} {nm}: fixed - library')
                if bp:
                    e['bootstrap_fixed_minus_library'] = bp
            fx[f'{dom}_{nm}'] = e
        if S9Fd.get('fx_dualtau_fixed') and S1.get(f'cap_doppler_W{cfg.W}'):
            bp = _boot(S1.get(f'cap_doppler_W{cfg.W}'), S9Fd.get('fx_dualtau_fixed'), C, 'radar: Doppler - Dual-tau fixed')
            if bp:
                fx['radar_doppler_minus_dualtau_fixed'] = bp
        rep['S9_initfix'] = fx
    if S9R or S4:
        full = {'Doppler-LIF': S4.get('shd_doppler'), 'CrossInhib-LIF': S4.get('shd_CI_default'), 'CrossInhib-LIF (stable init)': S4.get('shd_CI_stable')}
        for k, nm in S9_RANK_TAGS:
            full[nm] = S9R.get(k)
        cells = {nm: _cell(rs) for nm, rs in full.items() if rs}
        order = sorted(cells, key=lambda nm: -cells[nm]['f1_mean'])
        ns_ = sorted({c['n'] for c in cells.values()})
        pass
        pass
        for i, nm in enumerate(order, 1):
            c = cells[nm]
            ref = PAPER_TABLE3.get(nm)
            pass
        common = [nm for nm in PAPER_TABLE3 if nm in cells]
        agree = None
        if len(common) >= 4:
            agree = spearman_exact([cells[nm]['f1_mean'] for nm in common], [PAPER_TABLE3[nm] for nm in common])
            pass
            moved = [nm for nm in common if abs(cells[nm]['f1_mean'] - PAPER_TABLE3[nm]) > 2 * max(cells[nm]['f1_std'], 0.001)]
            pass
        rep['S9_shd_rank'] = {'cells': cells, 'order': order, 'submitted_table3': PAPER_TABLE3, 'agreement_with_submitted': agree, 'complete': len(common) == len(PAPER_TABLE3)}
    out_name = 'revision_R1_summary.json' if audit['complete'] else 'revision_R1_summary_PARTIAL.json'
    pass
    pass
    pass
    return rep
