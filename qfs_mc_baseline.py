"""
qfs_mc_baseline.py
=====================================================================
Baseline without RL: pure uniform Monte Carlo search, done separately for every
test map, plus test-set statistics in the same table format as the RL script.

For one map k:
    u^(i) = (u_1, ..., u_T),   u_t ~ U[mult_min, mult_max]  i.i.d.,   i = 1, 2, ...
    J(u)  = sum_t ( prod_t - lambda_P * pviol_t )        (same J as the RL environment)
    result: max_i J(u^(i)) and the u that achieved it.
Every rollout starts from the same post-warm-up state of map k (snapshot), so the
search optimizes an open-loop schedule for a KNOWN map ("oracle" baseline).
The best sample is a lower bound on the true optimum of J.

Stopping rule per map:
    converged   : n >= N_MIN and the best J improved by <= TOL over the last PATIENCE samples
    max_samples : n >= N_MAX
    incomplete  : anything else (run interrupted)

"Window hit": window t is a hit  <=>  p_max_t >= P_LIMIT  (p_max_t = max pressure in window t)

Test-set statistics over M maps, for every column c:
    mean = (1/M) sum_k c_k,   std = sqrt( sum_k (c_k - mean)^2 / (M - STD_DDOF) )
    #maps with hit = #{ k : number of hit windows of map k > 0 }
    J_rand_mean    = mean J of all uniform samples (expected J of a random policy;
                     failed rollouts excluded)

Outputs (results_<tag>/):
  report_<tag>.xlsx
      test_table          : one row per map
      test_stats          : mean / std / median / min / max / #maps with hit
      settings
      best_episodes_long  : best-u episode of every map (window, u, p_max, prod, inj, penalty, cumulative)
      k<map>_best_episode, k<map>_samples : overlay maps only
      figures
  table_test.tex          : LaTeX (booktabs) table: rows + mean +/- std + #maps with hit
  test_table.csv / test_stats.csv / best_episodes_long.csv / settings.csv
  overlay_maps.png        : 4 panels (multiplier / max pressure / per window / cumulative) for the overlay maps
  envelope_all.png        : same 4 panels for all maps: thin line per map + mean +/- std band
  grid_maps_p<page>.png   : one small plot per map (u on the left axis, p_max on the right, hits as red x)
  per_map/episode_k<map>.png, per_map/samples_k<map>.png, per_map/k<map>_samples.csv
  samples_k<map>.npz      : raw samples (for --resume and --report-only)
  env_info.json

Examples:
  python qfs_mc_baseline.py --map-range 951 1000             # all 50 test maps
  python qfs_mc_baseline.py --maps 980 990
  python qfs_mc_baseline.py --report-only results_<tag>      # rebuild the report from the npz files, no simulation
"""
import os
import re
import glob
import json
import time
import argparse
import importlib
import multiprocessing as mp

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =====================================================================
# 0. Settings
# =====================================================================
RL_MODULE       = "qfs_ppo"     # module that defines QFSSinglePhaseEnv (changeable with --env-module)
P_LIMIT         = 230.0       # [bar] penalty threshold (overrides the env value)
N_WINDOWS       = 10          # number of control windows T
DAYS_PER_WINDOW = 30.0        # [day]
MAPS_DEFAULT         = [980, 990]
OVERLAY_MAPS_DEFAULT = [980, 990]

# Stopping rule (in samples); can be overridden on the command line
N_MIN    = 5000
N_MAX    = 50000
PATIENCE = 5000
TOL      = 0.1
BATCH_PER_WORKER = 2

STD_DDOF   = 0                # 0: population std (numpy default), 1: sample std
VALID_RTOL = 1e-6
GRID_PER_PAGE, GRID_NCOL = 25, 5

# =====================================================================
# 1. Worker processes (each one owns a MATLAB engine)
# =====================================================================
_env = None
_cur_map = None
_validated = False


def _init_worker(module_name=RL_MODULE):
    # The module name is passed explicitly: spawned workers re-import this file,
    # so a value changed in main() would not reach them as a global.
    global _env
    mod = importlib.import_module(module_name)
    _env = mod.QFSSinglePhaseEnv()
    _env.days_per_window = float(DAYS_PER_WINDOW)
    _env.n_windows       = int(N_WINDOWS)
    _env.T               = int(N_WINDOWS)
    _env.total_days      = _env.days_per_window * _env.n_windows
    _env.Pmax_limit      = float(P_LIMIT)


def _env_info():
    return dict(lambdaP=float(_env.lambdaP), n_warmup=int(_env.n_warmup),
                warmup_mults=[float(m) for m in (getattr(_env, 'warmup_mults', None)
                                                  or [_env.warmup_mult] * _env.n_warmup)],
                mult_min=float(_env.mult_min), mult_max=float(_env.mult_max),
                dt_days=float(_env.dt_days), Pmax_limit_env=float(_env.Pmax_limit))


def _rollout(u_seq, restore=True):
    """Run T control windows with the schedule u_seq and return J and per-window values.
    restore=True first restores the post-warm-up snapshot of the current map."""
    if restore:
        _env.eng.eval("qfs_state = qfs_state_warm;", nargout=0)
    lam = _env.lambdaP
    T = len(u_seq)
    p_max_w = np.full(T, np.nan); prod_w = np.full(T, np.nan); pen_w = np.full(T, np.nan)
    inj_w = np.full(T, np.nan)
    for t, u in enumerate(u_seq):
        res = _env._solver(float(u), _env.dt_days, False, 0.0, clear_carry=(t == 0))
        prod, pviol, pmax = float(res['prod']), float(res['pviol']), float(res['p_max'])
        if bool(res['failed']) or not np.all(np.isfinite([prod, pviol, pmax])):
            return -np.inf, p_max_w, prod_w, pen_w, inj_w
        p_max_w[t], prod_w[t], pen_w[t] = pmax, prod, -lam * pviol
        inj_w[t] = float(res['inj'])
    return float(prod_w.sum() + pen_w.sum()), p_max_w, prod_w, pen_w, inj_w


def _load_map(k):
    """Reset to map k (incl. warm-up) and keep a snapshot of the post-warm-up state."""
    global _cur_map, _validated
    if _cur_map == k:
        return
    _, info = _env.reset(options={'map_idx': int(k)})
    if info['warmup_failed']:
        raise RuntimeError(f"warm-up failed on map {k}")
    _env.eng.eval("qfs_state_warm = qfs_state;", nargout=0)
    if not _validated:                      # check once per worker: snapshot restore gives the same J as a full reset
        u_chk = np.full(N_WINDOWS, 1.0)
        J_snap = _rollout(u_chk, restore=True)[0]
        _env.reset(options={'map_idx': int(k)})
        J_full = _rollout(u_chk, restore=False)[0]
        if not (np.isfinite(J_snap) and abs(J_snap - J_full) <= VALID_RTOL * max(1.0, abs(J_full))):
            raise RuntimeError(f"[k={k}] snapshot check failed: J_snap={J_snap} vs J_full={J_full}")
        _env.reset(options={'map_idx': int(k)})
        _env.eng.eval("qfs_state_warm = qfs_state;", nargout=0)
        _validated = True
    _cur_map = int(k)


def task_rollout(job):
    k, u = job
    _load_map(k)
    return _rollout(np.asarray(u, dtype=float))


# =====================================================================
# 2. Samples -> per-map record
# =====================================================================
def conv_status(J):
    best = np.maximum.accumulate(J)            # well defined even with -inf (failed rollouts)
    n = len(J)
    gain = best[-1] - best[-PATIENCE - 1] if n > PATIENCE else np.inf
    if n >= N_MIN and gain <= TOL:
        return "converged"
    return "max_samples" if n >= N_MAX else "incomplete"


def make_rec(k, U, J, PM, PR, PE, IN, sec, npz):
    i = int(np.argmax(J))
    return dict(k=int(k), status=conv_status(J), n=len(J), J_best=float(J[i]), i_best=i,
                u_best=U[i], p_max_w=PM[i], prod_w=PR[i], pen_w=PE[i], inj_w=IN[i],
                prod=float(PR[i].sum()), pen=float(PE[i].sum()),
                J_all=J, U_all=U, PM_all=PM, PR_all=PR, PE_all=PE, sec=sec, npz=npz)


def load_npz(path):
    d = np.load(path)
    PR = d['prod']
    IN = d['inj'] if 'inj' in d.files else np.full_like(PR, np.nan)   # older npz files have no inj
    return d['U'], d['J'], d['p_max'], PR, d['pen'], IN


def run_map(pool, k, seed, info, batch, out_dir, resume):
    lo, hi, T = info['mult_min'], info['mult_max'], N_WINDOWS
    npz = os.path.join(out_dir, f"samples_k{k}.npz")
    U = np.zeros((0, T)); J = np.zeros(0)
    PM = np.zeros((0, T)); PR = np.zeros((0, T)); PE = np.zeros((0, T)); IN = np.zeros((0, T))
    if resume and os.path.isfile(resume.format(k=k)):
        U, J, PM, PR, PE, IN = load_npz(resume.format(k=k))
        print(f"  [k={k}] resumed {len(J)} samples from {resume.format(k=k)}", flush=True)
    rng = np.random.default_rng([seed, int(k), len(J)])     # different random stream when resuming

    t0 = time.time()
    while len(J) < N_MAX:
        Ub = rng.uniform(lo, hi, size=(batch, T))
        rs = pool.map(task_rollout, [(k, u.tolist()) for u in Ub], chunksize=1)
        U  = np.vstack([U, Ub])
        J  = np.concatenate([J, [r[0] for r in rs]])
        PM = np.vstack([PM, [r[1] for r in rs]])
        PR = np.vstack([PR, [r[2] for r in rs]])
        PE = np.vstack([PE, [r[3] for r in rs]])
        IN = np.vstack([IN, [r[4] for r in rs]])
        np.savez(npz, U=U, J=J, p_max=PM, prod=PR, pen=PE, inj=IN)

        best = np.maximum.accumulate(J)
        n = len(J)
        gain = best[-1] - best[-PATIENCE - 1] if n > PATIENCE else np.inf
        print(f"  [k={k}] n={n:6d} ({time.time() - t0:6.0f}s) | best J={best[-1]:9.3f} | "
              f"gain over last {PATIENCE}={gain:8.3f} | failed={np.sum(~np.isfinite(J))}", flush=True)
        if n >= N_MIN and gain <= TOL:
            break
    return make_rec(k, U, J, PM, PR, PE, IN, time.time() - t0, npz)


# =====================================================================
# 3. Statistics (same definitions as in qfs_ppo.py)
# =====================================================================
def count_cols():
    return ['n_ge_Plim']


VALUE_COLS = ['p_peak', 'J', 'prod', 'penalty', 'inj', 'u_mean']


def episode_stats(k, u, p_max_w, prod_w, pen_w, inj_w):
    """One table row for one episode (= one map, T windows)."""
    pm = np.asarray(p_max_w, float)
    d = dict(map_idx=int(k), n_ge_Plim=int(np.sum(pm >= P_LIMIT)))
    prod, pen = float(np.sum(prod_w)), float(np.sum(pen_w))
    d.update(p_peak=float(np.max(pm)), J=prod + pen, prod=prod, penalty=pen,
             inj=float(np.sum(inj_w)), u_mean=float(np.mean(u)))
    return d


def aggregate_stats(df, extra_cols=()):
    """mean / std / median / min / max over M maps, plus the number of maps with at least one hit."""
    import pandas as pd
    cc = count_cols()
    cols = cc + VALUE_COLS + [c for c in extra_cols if c in df.columns]
    X = {c: df[c].to_numpy(float) for c in cols}
    funcs = [('mean', np.mean), (f'std (ddof={STD_DDOF})', lambda x: np.std(x, ddof=STD_DDOF)),
             ('median', np.median), ('min', np.min), ('max', np.max)]
    rows = [{'stat': name, **{c: float(f(X[c])) for c in cols}} for name, f in funcs]
    rows.append({'stat': f'#maps with hit (of {len(df)})',
                 **{c: (int(np.sum(X[c] > 0)) if c in cc else np.nan) for c in cols}})
    return pd.DataFrame(rows)


def baseline_test_table(recs):
    import pandas as pd
    rows = []
    for r in recs:
        d = episode_stats(r['k'], r['u_best'], r['p_max_w'], r['prod_w'], r['pen_w'], r['inj_w'])
        assert np.isclose(d['J'], r['J_best'], rtol=1e-12, atol=1e-9), (d['J'], r['J_best'])
        Jf = r['J_all'][np.isfinite(r['J_all'])]
        d.update(J_rand_mean=float(np.mean(Jf)), J_rand_std=float(np.std(Jf)),
                 n_samples=r['n'], best_sample=r['i_best'] + 1, status=r['status'],
                 failed=int(len(r['J_all']) - len(Jf)), sec=r['sec'])
        rows.append(d)
    return pd.DataFrame(rows).sort_values('map_idx').reset_index(drop=True)


def write_latex(df, agg, path, caption):
    cc = count_cols()
    heads = {'n_ge_Plim': r'$\ge P_{\lim}$', 'p_peak': r'max $p_{\max}$ [bar]',
             'J': r'$J$ [m$^3$]', 'prod': r'Prod. [m$^3$]', 'penalty': r'Penalty [m$^3$]',
             'J_rand_mean': r'$\bar J_{\mathrm{rand}}$ [m$^3$]'}
    cols = cc + ['p_peak', 'J', 'prod', 'penalty'] + (['J_rand_mean'] if 'J_rand_mean' in df.columns else [])
    fmt = lambda c, v: f"{int(v)}" if c in cc else f"{v:.2f}"
    m, s, h = agg.iloc[0], agg.iloc[1], agg.iloc[-1]
    L = [r'\begin{table}[ht]', r'\centering', rf'\caption{{{caption}}}',
         r'\begin{tabular}{r' + 'r' * len(cols) + '}', r'\toprule',
         'Map & ' + ' & '.join(heads[c] for c in cols) + r' \\', r'\midrule']
    for _, row in df.iterrows():
        L.append(f"{int(row['map_idx'])} & " + ' & '.join(fmt(c, row[c]) for c in cols) + r' \\')
    L.append(r'\midrule')
    L.append(r'mean$\pm$std & ' + ' & '.join(f"{m[c]:.2f}$\\pm${s[c]:.2f}" for c in cols) + r' \\')
    L.append(r'\#maps with hit & ' + ' & '.join((f"{int(h[c])}/{len(df)}" if c in cc else '')
                                                 for c in cols) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    with open(path, 'w') as f:
        f.write('\n'.join(L) + '\n')


# =====================================================================
# 4. Figures
# =====================================================================
def _step_xy(u):
    return np.arange(len(u) + 1) + 1.0, np.append(u, u[-1])


def _limit_lines(a):
    a.axhline(P_LIMIT, color='purple', ls='--', lw=2, label=f'$P_{{lim}}$ = {P_LIMIT:.0f} bar (penalty)')


def plot_episode(rec, tag, info, path):
    """2x2 figure for one map, driven with its best u."""
    T = N_WINDOWS; w = np.arange(1, T + 1); u = rec['u_best']
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"{tag}: uniform-MC best u on kmap #{rec['k']}  (J={rec['J_best']:.1f}, "
                 f"sample {rec['i_best'] + 1}/{rec['n']})")
    a = ax[0, 0]
    a.step(*_step_xy(u), where='post', color='darkgreen', lw=2, label='best u')
    a.axhline(info['mult_min'], color='k', ls=':', lw=1); a.axhline(info['mult_max'], color='k', ls=':', lw=1)
    a.set_ylim(info['mult_min'] - 0.1, info['mult_max'] + 0.1)
    a.set_title('Injection multiplier'); a.set_xlabel('Window'); a.legend(); a.grid(alpha=.3)
    a = ax[0, 1]
    a.plot(w, rec['p_max_w'], 'o-', color='red', lw=2); _limit_lines(a)
    a.set_title('Max pressure (bar)'); a.set_xlabel('Window'); a.legend(); a.grid(alpha=.3)
    a = ax[1, 0]
    a.plot(w, rec['prod_w'], 'o-', color='green', lw=2, label='Produced')
    a.plot(w, rec['inj_w'], 's--', color='blue', lw=2, label='Injected')
    a.set_title('Per window (m^3)'); a.set_xlabel('Window'); a.legend(); a.grid(alpha=.3)
    a = ax[1, 1]
    a.plot(w, np.cumsum(rec['prod_w']), 'o-', color='green', lw=2, label='cum. production')
    a.plot(w, np.cumsum(rec['prod_w'] + rec['pen_w']), 's--', color='k', lw=1.5, label='cum. J (prod + penalty)')
    a.set_title('Cumulative (m^3)'); a.set_xlabel('Window'); a.legend(); a.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()


def plot_samples(rec, tag, path):
    J = rec['J_all']; ok = np.isfinite(J)
    i = np.arange(1, len(J) + 1)
    best = np.maximum.accumulate(np.where(ok, J, -np.inf))
    run_mean = np.cumsum(np.where(ok, J, 0)) / np.maximum(np.cumsum(ok), 1)
    fig, ax = plt.subplots(1, 3, figsize=(20, 5), gridspec_kw=dict(width_ratios=[2.2, 1.3, 1]))
    fig.suptitle(f"{tag}: uniform-MC samples on kmap #{rec['k']}  (n={rec['n']}, {rec['status']})")
    for a, xs in ((ax[0], 'linear'), (ax[1], 'log')):
        a.scatter(i[ok], J[ok], s=2, color='gray', alpha=.35, label='J of sample i')
        a.plot(i, best, color='darkgreen', lw=2, label='best-so-far  max_{j<=i} J_j')
        a.plot(i, run_mean, color='orange', lw=2, label='running mean  (1/i) sum J_j')
        a.set_xscale(xs); a.set_xlabel(f'sample i ({xs})'); a.set_ylabel('J (m^3)'); a.grid(alpha=.3, which='both')
    ax[0].legend(loc='lower right'); ax[0].set_title('J per sample')
    ax[1].set_ylim(np.percentile(J[ok], 90), rec['J_best'] + 0.02 * abs(rec['J_best']) + 1)
    ax[1].set_title('zoom: top 10% band, log x')
    ax[2].hist(J[ok], bins=60, color='gray'); ax[2].axvline(rec['J_best'], color='darkgreen', lw=2, label='max')
    ax[2].axvline(run_mean[-1], color='orange', lw=2, label='mean'); ax[2].legend()
    ax[2].set_title('distribution of J'); ax[2].set_xlabel('J (m^3)'); ax[2].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()


def plot_overlay(recs, tag, info, path):
    """Overlay of the 4 panels for the selected maps (one color per map)."""
    T = N_WINDOWS; w = np.arange(1, T + 1)
    cmap = plt.get_cmap('tab10')
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"{tag}: uniform-MC best u, overlay of kmaps {[r['k'] for r in recs]}")
    for j, r in enumerate(recs):
        c = cmap(j % 10); k = r['k']
        ax[0, 0].step(*_step_xy(r['u_best']), where='post', color=c, lw=2, label=f"k{k} (J={r['J_best']:.1f})")
        ax[0, 1].plot(w, r['p_max_w'], 'o-', color=c, lw=2, label=f"k{k} (peak={np.max(r['p_max_w']):.1f})")
        ax[1, 0].plot(w, r['prod_w'], 'o-', color=c, lw=2, label=f"k{k} produced")
        ax[1, 0].plot(w, r['inj_w'], 's--', color=c, lw=1.5, alpha=.8, label=f"k{k} injected")
        ax[1, 1].plot(w, np.cumsum(r['prod_w']), 'o-', color=c, lw=2, label=f"k{k} cum. prod")
        ax[1, 1].plot(w, np.cumsum(r['prod_w'] + r['pen_w']), 's--', color=c, lw=1.5, label=f"k{k} cum. J")
    ax[0, 0].axhline(info['mult_min'], color='k', ls=':', lw=1); ax[0, 0].axhline(info['mult_max'], color='k', ls=':', lw=1)
    ax[0, 0].set_ylim(info['mult_min'] - 0.1, info['mult_max'] + 0.1)
    _limit_lines(ax[0, 1])
    for a, t in zip(ax.flat, ['Injection multiplier', 'Max pressure (bar)', 'Per window (m^3)', 'Cumulative (m^3)']):
        a.set_title(t); a.set_xlabel('Window'); a.grid(alpha=.3); a.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()


def plot_envelope(recs, tag, info, path):
    """All maps: thin line per map + mean +/- std band (4 panels)."""
    T = N_WINDOWS; w = np.arange(1, T + 1)
    Y = [np.array([r['u_best'] for r in recs]),
         np.array([r['p_max_w'] for r in recs]),
         np.array([r['prod_w'] for r in recs]),
         np.array([np.cumsum(r['prod_w'] + r['pen_w']) for r in recs])]
    titles = ['Injection multiplier', 'Max pressure (bar)', 'Production per window (m^3)', 'Cumulative J (m^3)']
    hit_frac = np.mean(Y[1] >= P_LIMIT, axis=0)
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"{tag}: uniform-MC best u over {len(recs)} maps (thin: each map, bold: mean, band: ±std, ddof={STD_DDOF})")
    for a, y, t in zip(ax.flat, Y, titles):
        for row in y:
            a.plot(w, row, color='gray', lw=.6, alpha=.4)
        mu, sd = y.mean(0), y.std(0, ddof=STD_DDOF) if len(y) > STD_DDOF else np.zeros(T)
        a.plot(w, mu, 'o-', color='k', lw=2, label='mean')
        a.fill_between(w, mu - sd, mu + sd, color='tab:blue', alpha=.25, label='mean ± std')
        a.set_title(t); a.set_xlabel('Window'); a.grid(alpha=.3)
    ax[0, 0].set_ylim(info['mult_min'] - 0.1, info['mult_max'] + 0.1)
    _limit_lines(ax[0, 1])
    b = ax[0, 1].twinx()
    b.bar(w, hit_frac, color='red', alpha=.15, width=.6, label='fraction of maps hit')
    b.set_ylim(0, 1); b.set_ylabel(f'fraction with p_max >= {P_LIMIT:.0f}')
    for a in ax.flat:
        a.legend(fontsize=8, loc='best')
    b.legend(fontsize=8, loc='lower right')
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()


def plot_grid(recs, tag, info, out_dir):
    """One small plot per map (u on the left axis, p_max on the right axis, hits = red x)."""
    T = N_WINDOWS; w = np.arange(1, T + 1)
    allp = np.concatenate([r['p_max_w'] for r in recs])
    pmin = np.nanmin(allp); pmax = max(np.nanmax(allp), P_LIMIT)
    pad = 0.05 * (pmax - pmin + 1e-9)
    paths = []
    for p, p0 in enumerate(range(0, len(recs), GRID_PER_PAGE)):
        chunk = recs[p0:p0 + GRID_PER_PAGE]
        nrow = int(np.ceil(len(chunk) / GRID_NCOL))
        fig, axs = plt.subplots(nrow, GRID_NCOL, figsize=(4.2 * GRID_NCOL, 2.9 * nrow), squeeze=False)
        for a in axs.flat[len(chunk):]:
            a.axis('off')
        for a, r in zip(axs.flat, chunk):
            pm = r['p_max_w']; hit = pm >= P_LIMIT
            a.step(*_step_xy(r['u_best']), where='post', color='green', lw=1.5)
            a.set_ylim(info['mult_min'] - 0.05, info['mult_max'] + 0.05)
            a.tick_params(labelsize=7); a.grid(alpha=.3)
            b = a.twinx()
            b.plot(w, pm, '-', color='tab:blue', lw=1.2)
            b.plot(w[hit], pm[hit], 'x', color='red', ms=6, mew=2)
            b.axhline(P_LIMIT, color='purple', ls='--', lw=1)
            b.set_ylim(pmin - pad, pmax + pad); b.tick_params(labelsize=7, colors='tab:blue')
            a.set_title(f"k{r['k']}: J={r['J_best']:.1f} peak={np.max(pm):.1f} hit={int(hit.sum())}", fontsize=8)
        handles = [plt.Line2D([], [], color='green', lw=1.5, label='multiplier u (left axis)'),
                   plt.Line2D([], [], color='tab:blue', lw=1.2, label='window p_max [bar] (right axis)'),
                   plt.Line2D([], [], color='purple', ls='--', label=f'P_lim = {P_LIMIT:.0f}'),
                   plt.Line2D([], [], color='red', marker='x', ls='', mew=2, label='p_max >= P_lim')]
        fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=9)
        fig.suptitle(f"{tag}: maps {chunk[0]['k']}–{chunk[-1]['k']} (best u per map)", fontsize=11)
        plt.tight_layout(rect=(0, 0.03, 1, 0.97))
        path = os.path.join(out_dir, f"grid_maps_p{p + 1}.png")
        plt.savefig(path, dpi=150); plt.close(); paths.append(path)
    return paths


# =====================================================================
# 5. Per-map output + report
# =====================================================================
def samples_table(rec):
    import pandas as pd
    J = rec['J_all']; ok = np.isfinite(J)
    sm = pd.DataFrame(dict(sample=np.arange(1, len(J) + 1), J=np.where(ok, J, np.nan),
                           best_so_far=np.maximum.accumulate(np.where(ok, J, -np.inf)),
                           running_mean=np.cumsum(np.where(ok, J, 0)) / np.maximum(np.cumsum(ok), 1),
                           p_peak=np.nanmax(rec['PM_all'], axis=1),
                           produced=np.nansum(rec['PR_all'], axis=1),
                           penalty=np.nansum(rec['PE_all'], axis=1)))
    for t in range(N_WINDOWS):
        sm[f"u{t+1}"] = rec['U_all'][:, t]
    return sm


def finalize_map(rec, tag, info, out_dir):
    """Write the per-map figures/sample table, then drop the large arrays (only J_all is kept)."""
    pdir = os.path.join(out_dir, 'per_map'); os.makedirs(pdir, exist_ok=True)
    k = rec['k']
    plot_episode(rec, tag, info, os.path.join(pdir, f"episode_k{k}.png"))
    plot_samples(rec, tag, os.path.join(pdir, f"samples_k{k}.png"))
    samples_table(rec).to_csv(os.path.join(pdir, f"k{k}_samples.csv"), index=False)
    for key in ('U_all', 'PM_all', 'PR_all', 'PE_all'):
        rec.pop(key, None)
    return rec


def best_episode_df(r):
    import pandas as pd
    T = N_WINDOWS
    return pd.DataFrame(dict(map_idx=r['k'], window=np.arange(1, T + 1), u=r['u_best'], p_max_bar=r['p_max_w'],
                             hit=(r['p_max_w'] >= P_LIMIT).astype(int), produced=r['prod_w'], injected=r['inj_w'],
                             penalty=r['pen_w'], reward=r['prod_w'] + r['pen_w'],
                             cum_production=np.cumsum(r['prod_w']), cum_J=np.cumsum(r['prod_w'] + r['pen_w'])))


def write_report(recs, info, tag, out_dir, overlay_maps):
    import pandas as pd
    recs = sorted(recs, key=lambda r: r['k'])
    M = len(recs)
    df = baseline_test_table(recs)
    agg = aggregate_stats(df, extra_cols=['J_rand_mean'])
    long = pd.concat([best_episode_df(r) for r in recs], ignore_index=True)
    settings = pd.DataFrame(dict(
        item=['P_limit [bar]', 'N_WINDOWS', 'DAYS_PER_WINDOW', 'lambdaP',
              'u range', 'warm-up', 'N_MIN', 'N_MAX', 'PATIENCE', 'TOL', 'std ddof', 'n_maps', 'maps',
              'hit definition'],
        value=[P_LIMIT, N_WINDOWS, DAYS_PER_WINDOW, info['lambdaP'],
               f"U[{info['mult_min']}, {info['mult_max']}] iid", f"{info['n_warmup']} win, u={info['warmup_mults']}",
               N_MIN, N_MAX, PATIENCE, TOL, STD_DDOF, M, ' '.join(str(r['k']) for r in recs),
               'window t hit <=> p_max_t >= P_limit']))

    tables = [('test_table', df), ('test_stats', agg), ('settings', settings), ('best_episodes_long', long)]
    for name, t in tables:
        t.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)
    write_latex(df, agg, os.path.join(out_dir, "table_test.tex"),
                caption=(f"Uniform-MC baseline (best of $n$ samples per map): control windows (out of {N_WINDOWS}) "
                         f"whose window max pressure reaches $P_{{\\lim}}={P_LIMIT:.0f}$ bar, over {M} test maps. "
                         f"std with ddof={STD_DDOF}."))

    ov = [r for r in recs if r['k'] in set(overlay_maps)]
    figs = []
    if ov:
        plot_overlay(ov, tag, info, os.path.join(out_dir, "overlay_maps.png")); figs.append("overlay_maps.png")
    plot_envelope(recs, tag, info, os.path.join(out_dir, "envelope_all.png")); figs.append("envelope_all.png")
    figs += [os.path.basename(p) for p in plot_grid(recs, tag, info, out_dir)]
    for r in ov:
        figs += [os.path.join('per_map', f"episode_k{r['k']}.png"), os.path.join('per_map', f"samples_k{r['k']}.png")]

    try:
        import openpyxl  # noqa: F401
    except ImportError:
        print("[report] openpyxl not installed -> only CSV/TeX written (pip install openpyxl pillow for the xlsx)", flush=True)
        return None
    path = os.path.join(out_dir, f"report_{tag}.xlsx")
    with pd.ExcelWriter(path, engine='openpyxl') as xw:
        for name, t in tables:
            t.to_excel(xw, sheet_name=name, index=False)
        for r in ov:
            best_episode_df(r).to_excel(xw, sheet_name=f"k{r['k']}_best_episode", index=False)
            sp = os.path.join(out_dir, 'per_map', f"k{r['k']}_samples.csv")
            if os.path.isfile(sp):
                pd.read_csv(sp).to_excel(xw, sheet_name=f"k{r['k']}_samples", index=False)
        pd.DataFrame().to_excel(xw, sheet_name='figures')
    try:
        from openpyxl import load_workbook
        from openpyxl.drawing.image import Image as XLImage
        wb = load_workbook(path); ws = wb['figures']; row = 1
        for name in figs:
            img = XLImage(os.path.join(out_dir, name))
            img.width, img.height = img.width * 0.35, img.height * 0.35
            ws.add_image(img, f"A{row}"); row += int(img.height / 20) + 2
        wb.save(path)
    except Exception as e:
        print(f"[report] could not embed figures ({e}); tables were saved", flush=True)
    return path


def print_stats(df, agg):
    cc = count_cols()
    m, s, h = agg.iloc[0], agg.iloc[1], agg.iloc[-1]
    print(f"\n==== test-set stats over {len(df)} maps (std ddof={STD_DDOF}) ====")
    for c in cc + VALUE_COLS + ['J_rand_mean']:
        hit = f"   maps with hit: {int(h[c])}/{len(df)}" if c in cc else ""
        print(f"  {c:12s}: {m[c]:10.3f} ± {s[c]:9.3f}   [min {agg.iloc[3][c]:.3f}, max {agg.iloc[4][c]:.3f}]{hit}")


# =====================================================================
# 6. main
# =====================================================================
FALLBACK_INFO = dict(lambdaP=float('nan'), n_warmup=-1, warmup_mults=[], mult_min=0.5, mult_max=1.5,
                     dt_days=float('nan'))


def main():
    global N_MIN, N_MAX, PATIENCE, TOL, RL_MODULE
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", type=int, nargs="*", default=None)
    ap.add_argument("--map-range", type=int, nargs="+", default=None, metavar="A B [STEP]",
                    help="range(A, B+1, STEP), e.g. --map-range 951 1000")
    ap.add_argument("--overlay-maps", type=int, nargs="*", default=OVERLAY_MAPS_DEFAULT)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 16)))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", type=str, default=None,
                    help="path pattern of earlier npz files, e.g. 'results_<old tag>/samples_k{k}.npz'")
    ap.add_argument("--report-only", type=str, default=None, metavar="DIR",
                    help="rebuild the report from DIR/samples_k*.npz without simulating")
    ap.add_argument("--n-min", type=int, default=N_MIN)
    ap.add_argument("--n-max", type=int, default=N_MAX)
    ap.add_argument("--patience", type=int, default=PATIENCE)
    ap.add_argument("--tol", type=float, default=TOL)
    ap.add_argument("--env-module", type=str, default=RL_MODULE,
                    help="Python module that defines QFSSinglePhaseEnv")
    args = ap.parse_args()
    N_MIN, N_MAX, PATIENCE, TOL = args.n_min, args.n_max, args.patience, args.tol
    RL_MODULE = args.env_module

    if args.map_range:
        a, b = args.map_range[:2]; st = args.map_range[2] if len(args.map_range) > 2 else 1
        maps = list(range(a, b + 1, st))
    elif args.maps:
        maps = list(args.maps)
    else:
        maps = None
    if maps is not None:
        maps = list(dict.fromkeys(maps))            # remove duplicates, keep order

    # ---------------- report-only ----------------
    if args.report_only:
        out_dir = args.report_only
        tag = os.path.basename(os.path.normpath(out_dir))
        tag = tag[len("results_"):] if tag.startswith("results_") else tag
        jp = os.path.join(out_dir, "env_info.json")
        if os.path.isfile(jp):
            with open(jp) as f:
                info = json.load(f)
        else:
            info = dict(FALLBACK_INFO)
            print(f"[report-only] {jp} not found -> using u range [0.5, 1.5], lambdaP=nan", flush=True)
        if maps is None:
            maps = sorted(int(re.search(r"samples_k(\d+)\.npz$", f).group(1))
                          for f in glob.glob(os.path.join(out_dir, "samples_k*.npz")))
        recs = []
        for k in maps:
            npz = os.path.join(out_dir, f"samples_k{k}.npz")
            if not os.path.isfile(npz):
                print(f"  [k={k}] {npz} not found -> skipped", flush=True); continue
            rec = make_rec(k, *load_npz(npz), sec=float('nan'), npz=npz)
            if not np.isfinite(rec['J_best']):
                print(f"  [k={k}] all samples failed -> skipped", flush=True); continue
            recs.append(finalize_map(rec, tag, info, out_dir))
        report = write_report(recs, info, tag, out_dir, args.overlay_maps)
        df = baseline_test_table(recs); print_stats(df, aggregate_stats(df, extra_cols=['J_rand_mean']))
        print(f"Done. outputs in {out_dir}/  (report: {report})")
        return

    # ---------------- sampling ----------------
    maps = maps or MAPS_DEFAULT
    job_id = os.environ.get('SLURM_JOB_ID', 'local')
    tag = f"qfs_sp_uniform_P{P_LIMIT:.0f}_T{N_WINDOWS}_M{len(maps)}_job{job_id}"
    batch = args.workers * BATCH_PER_WORKER
    out_dir = f"results_{tag}"
    os.makedirs(out_dir, exist_ok=True)
    recs = []

    with mp.get_context("spawn").Pool(processes=args.workers, initializer=_init_worker,
                                    initargs=(RL_MODULE,)) as pool:
        info = pool.apply(_env_info)
        with open(os.path.join(out_dir, "env_info.json"), "w") as f:
            json.dump(info, f, indent=2)
        print(f"[uniform-MC] {len(maps)} maps={maps}, workers={args.workers}, batch={batch}, "
              f"T={N_WINDOWS} x {DAYS_PER_WINDOW:g} d, P_limit={P_LIMIT:g}, lambdaP={info['lambdaP']:g}, "
              f"u_t ~ U[{info['mult_min']}, {info['mult_max']}]", flush=True)
        print(f"[uniform-MC] stop: n>={N_MIN} and best-J gain over last {PATIENCE} <= {TOL}, "
              f"or n>={N_MAX}", flush=True)
        for k in maps:
            try:
                rec = run_map(pool, k, args.seed, info, batch, out_dir, args.resume)
            except Exception as e:
                print(f"  [k={k}] ERROR: {e}", flush=True)
                continue
            if not np.isfinite(rec['J_best']):
                print(f"  [k={k}] all samples failed -> excluded from the statistics", flush=True)
                continue
            print(f"  [k={k}] DONE ({rec['status']}, n={rec['n']}, {rec['sec']:.0f}s) | "
                  f"max J={rec['J_best']:.3f} | p_peak={np.max(rec['p_max_w']):.2f} | "
                  f"hit={int(np.sum(rec['p_max_w'] >= P_LIMIT))} | u*={np.round(rec['u_best'], 3)}", flush=True)
            recs.append(finalize_map(rec, tag, info, out_dir))
            report = write_report(recs, info, tag, out_dir, args.overlay_maps)   # refreshed after every map
    if recs:
        df = baseline_test_table(recs); print_stats(df, aggregate_stats(df, extra_cols=['J_rand_mean']))
    print(f"Done. all outputs in {out_dir}/  (report: report_{tag}.xlsx, table_test.tex)")


if __name__ == "__main__":
    main()