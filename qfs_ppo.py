"""
qfs_ppo.py
=====================================================================
PPO agent for injection-rate control in a single-phase, incompressible
quarter five-spot reservoir (MRST simulator called through the MATLAB
Engine for Python).

What this script does (in order):
  1. Splits the permeability maps into train maps (1 .. n_unique-50) and
     test maps (the last 50).
  2. Picks 9 evaluation maps from the train set: for every train map it
     measures the pressure sensitivity a_k (see probe_a_k) and takes the maps
     closest to the 10th, 20th, ..., 90th percentile of a_k.
  3. Trains PPO (8 parallel environments, each with its own MATLAB engine).
  4. Runs the best saved model deterministically on every test map and writes
     tables (CSV / XLSX / LaTeX) and figures in the same format as the
     Monte Carlo baseline (qfs_mc_baseline.py), so both can be compared.

Episode (one permeability map):
  reset : load map k, run n_warmup = 3 warm-up windows with fixed u = [0.5, 0.7, 1.0]
          (no reward; they only fill the observation history)
  step  : one control window of 30 days with u_t in [0.5, 1.5], solved with 1-day steps
  reward: r_t = prod_t - lambda_P * pviol_t   (sum over t gives the objective J)
  T = 10 control windows (300 days). Episodes end early only if the solver fails.

Run:  python qfs_ppo.py                      (full run, defaults below)
      python qfs_ppo.py --help               (options for a quick smoke test)
"""
import argparse
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless (OSC)
import matplotlib.pyplot as plt
import os
import json
import time
from collections import deque
import matlab.engine

import torch as th
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# =====================================================================
# 0. Data split / evaluation maps / plotting and report settings
#    (kept identical to qfs_mc_baseline.py so the outputs can be compared)
# =====================================================================
KMAP_FILE = "k128x128_unique.mat"
N_TEST = 50                               # test maps = the last 50 maps (951..1000 when n_unique = 1000)
EVAL_PERCENTILES = [10, 20, 30, 40, 50, 60, 70, 80, 90]  # a_k percentiles used to pick the 9 evaluation maps
PLOT_TEST_MAPS = [980, 990]               # same as OVERLAY_MAPS_DEFAULT in qfs_mc_baseline.py

STEP_MA_WINDOW = 50
EP_MA_WINDOW   = 50      # moving-average window (episodes) for the training curve

# Statistics / plotting constants (same as qfs_mc_baseline.py)
STD_DDOF = 0             # 0: population std (numpy default), 1: sample std
GRID_PER_PAGE, GRID_NCOL = 25, 5


# =====================================================================
# 1. POMDP Reservoir Environment  (SINGLE-PHASE water quarter five-spot)
# =====================================================================
class QFSSinglePhaseEnv(gym.Env):
    """Gymnasium environment. One episode = one permeability map.

    Observation (Dict):
      grid : (H, 128, 128) last H pressure fields, normalized p_norm = (p - 90) / (300 - 90)
      vec  : (H + 1,)      last H normalized actions u_norm in [0, 1], plus (T - t) / T
    Action:
      a in [-1, 1]  ->  u_norm = (a + 1) / 2  ->  u = 0.5 + u_norm * (1.5 - 0.5)
      injection rate = u * base_rate  [m^3/day], held constant for one 30-day window
    Reward:
      r_t = prod_t - lambda_P * pviol_t
      prod_t  : water produced in window t [m^3]
      pviol_t : time integral (trapezoid rule, 1-day samples) of max(p_max - P_lim, 0)^2 [bar^2 day]
    """
    def __init__(self):
        super().__init__()

        # ---- physics / control ----------------------------------------
        self.PHYS_NX   = 128
        self.base_rate = 5.0               # q_base [m^3/day]
        self.mult_min  = 0.5
        self.mult_max  = 1.5
        self.days_per_window = 30.0        # Δt_control [day]
        self.n_windows       = 10
        self.T               = self.n_windows                          # number of control steps (T in the observation)
        self.total_days      = self.days_per_window * self.n_windows   # T_d = 300 days (time horizon of J)
        self.dt_days         = 1.0         # Δτ [day]

        # ---- reward function limit -------------
        #   J = cumProduced - lambdaP * trapz(time, max(p_max - Pmax_limit, 0).^2)
        #   step reward  r_t = prod_t - lambdaP * pviol_t
        self.P_bhp      = 100.0            # producer BHP [bar]
        self.Pmax_limit = 230.0
        self.lambdaP    = 0.01

        # ---- observation / action space-----------------------------------
        self.window_size = 3               # H
        self.OBS_NX = self.PHYS_NX
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(low=-10.0, high=10.0,
                               shape=(self.window_size, self.OBS_NX, self.OBS_NX), dtype=np.float32),
            "vec":  spaces.Box(low=0.0, high=1.0,
                               shape=(self.window_size + 1,), dtype=np.float32),
        })
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # p_norm = (p - 90) / (300 - 90)
        self.p_min_bar, self.p_max_bar = 90.0, 300.0

        # ---- warm-up period------------------------------------------
        self.n_warmup     = self.window_size
        self.warmup_mults = [0.5, 0.7, 1.0]                 # [0.5, 0.7, 1.0]

        if not isinstance(self.n_warmup, int) or self.n_warmup < 0:
            raise ValueError("n_warmup needs to be a number bigger than 0.")
        if len(self.warmup_mults) != self.n_warmup:
            raise ValueError("len(warmup_mults) must equal n_warmup.")
        for m in self.warmup_mults:
            if not self.mult_min <= m <= self.mult_max:
                raise ValueError("each warmup_mult must be inside [mult_min, mult_max].")
        self.u_norm_warm = [(m - self.mult_min) / (self.mult_max - self.mult_min)
                            for m in self.warmup_mults]

        # ---- MATLAB engine + MRST --------------------------------------
        print("Connecting to MATLAB...")
        self.eng = matlab.engine.start_matlab(
            "-nodisplay -nosplash -nodesktop -singleCompThread"
        )
        # Folder that contains MRST's startup.m. Set it once before running, e.g.
        #   export MRST_PATH=/path/to/mrst-2025a
        mrst_path = os.environ.get("MRST_PATH")
        if not mrst_path:
            raise RuntimeError("Environment variable MRST_PATH is not set. "
                               "Point it to the MRST folder that contains startup.m.")
        current_dir = os.getcwd()
        self.eng.addpath(mrst_path, nargout=0)
        self.eng.cd(mrst_path, nargout=0)
        self.eng.startup(nargout=0)
        self.eng.cd(current_dir, nargout=0)
        self.eng.addpath(current_dir, nargout=0)
        print("MATLAB engine & MRST ready.")

        # ---- kmap index range: define the train / test split -------------------
        S = self.eng.load(KMAP_FILE, "n_unique", nargout=1)
        if "n_unique" not in S:
            raise RuntimeError(f"{KMAP_FILE} has no variable n_unique (expected variables: kmap, n_unique).")
        self.n_unique = int(S["n_unique"])
        if not 0 < N_TEST < self.n_unique:
            raise ValueError(f"N_TEST={N_TEST} must be between 1 and {self.n_unique - 1}.")
        self.train_map_lo, self.train_map_hi = 1, self.n_unique - N_TEST
        self.test_map_indices = list(range(self.train_map_hi + 1, self.n_unique + 1))

        self.steps_taken = 0
        self.map_idx = float(self.train_map_lo)

    def _solver(self, rate_mult, dt_days, is_reset, map_idx, clear_carry=False):
        """Call qfs_sp_solver_warmup.m. is_reset=True loads map map_idx;
        otherwise simulates one window with multiplier rate_mult."""
        return self.eng.qfs_sp_solver_warmup(
            float(rate_mult), float(dt_days), float(self.days_per_window),
            bool(is_reset), float(map_idx), float(self.base_rate),
            float(self.Pmax_limit), bool(clear_carry), nargout=1
        )

    def _pressure_map(self, res):
        """Pressure field [Pa] -> normalized 128x128 map, and its maximum [bar]."""
        p = np.array(res['pressure']).flatten() / 1e5              # Pa -> bar
        p2d = p.reshape((self.PHYS_NX, self.PHYS_NX), order='F')
        p_norm = (p2d - self.p_min_bar) / (self.p_max_bar - self.p_min_bar)
        return p_norm.astype(np.float32), float(np.max(p))

    def _obs(self):
        grid = np.stack(list(self.p_history), axis=0).astype(np.float32)          # (H,128,128)
        tfeat = (self.T - self.steps_taken) / self.T                               # (T-t)/T
        vec = np.array(list(self.u_history) + [tfeat], dtype=np.float32)          # (H+1,)
        return {"grid": grid, "vec": vec}

    def probe_a_k(self, map_idx):
        """Pressure sensitivity of map k:
            a_k = (p_max(u_hi) - p_max(u_lo)) / (u_hi - u_lo)   [bar]

        Each probe resets the map and runs ONE window (no warm-up) as a single
        30-day step. Because the flow is incompressible with a fixed producer BHP,
        p_max = P_bhp + a_k * u exactly, so two probes are enough. The intercept
        check below prints a message if this linearity does not hold.
        Note: this overwrites the simulator state; always call reset() afterwards.
        """
        u_lo, u_hi = self.mult_min, self.mult_max
        p = {}
        for u in (u_lo, u_hi):
            self._solver(0.0, self.days_per_window, True, map_idx)
            res = self._solver(u, self.days_per_window, False, 0.0)
            if bool(res['failed']) or not np.isfinite(float(res['p_max'])):
                return float('nan')
            p[u] = float(res['p_max'])

        a_k = (p[u_hi] - p[u_lo]) / (u_hi - u_lo)
        intercept = p[u_lo] - u_lo * a_k
        if abs(intercept - self.P_bhp) > 1e-3 * max(abs(self.P_bhp), 1.0):
            print(f"[probe] map {int(map_idx)}: intercept {intercept:.4f} != P_bhp {self.P_bhp:g} "
                  f"(p_max is not linear in u)", flush=True)
        return float(a_k)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps_taken = 0
        if options is not None and options.get('map_idx') is not None:
            self.map_idx = float(options['map_idx'])
        else:
            self.map_idx = float(self.np_random.integers(self.train_map_lo, self.train_map_hi + 1))

        # 1. Build the model: load the map, initial pressure field, qfs_pcarry = []
        res = self._solver(0.0, self.dt_days, True, self.map_idx)
        self.k_grid = np.array(res['k_mD']).flatten()
        p_init, p_max_init = self._pressure_map(res)

        # 2. WARM-UP windows with fixed multipliers (no reward computed)
        warm_p, warm_pmax, warm_u = [], [], []
        warmup_failed = False
        for i in range(self.n_warmup):
            r = self._solver(self.warmup_mults[i], self.dt_days, False, 0.0)
            if bool(r['failed']):
                warmup_failed = True
                break
            p_w, p_max_w = self._pressure_map(r)
            warm_p.append(p_w)
            warm_pmax.append(p_max_w)
            warm_u.append(self.u_norm_warm[i])

        # 3. Observation history. Normally it is filled by the H warm-up windows;
        #    the padding with the initial field and u_norm = 0.5 is used only if warm-up failed.
        self.p_history = deque(maxlen=self.window_size)
        self.u_history = deque(maxlen=self.window_size)
        for _ in range(max(self.window_size - len(warm_p), 0)):
            self.p_history.append(p_init.copy())
            self.u_history.append(0.5)
        for p_w, u_w in zip(warm_p, warm_u):
            self.p_history.append(p_w)
            self.u_history.append(u_w)

        info = {
            'map_idx': int(self.map_idx),
            'max_pressure': warm_pmax[-1] if warm_pmax else p_max_init,
            'warmup_windows': len(warm_p),
            'warmup_failed': warmup_failed,
            'warmup_pmax': max(warm_pmax) if warm_pmax else float('nan'),
            'warmup_over_limit': bool(warm_pmax) and max(warm_pmax) > self.Pmax_limit,
        }
        return self._obs(), info

    def step(self, action):
        a_clip = float(np.clip(action[0], -1.0, 1.0))
        u_norm = (a_clip + 1.0) / 2.0                              # [0,1]
        rate_mult = self.mult_min + u_norm * (self.mult_max - self.mult_min)

        # First control window: drop the trapezoid carry left by the warm-up,
        # so the warm-up never contributes to the reward.
        is_first_control = (self.steps_taken == 0)
        res = self._solver(rate_mult, self.dt_days, False, 0.0, clear_carry=is_first_control)

        prod     = float(res['prod'])        # sum_tau q_prod * dtau        [m^3]
        inj      = float(res['inj'])
        pviol    = float(res['pviol'])       # carry-trapz of y_tau         [bar^2 day]
        p_max    = float(res['p_max'])       # max over the window's daily samples [bar]
        inj_rate = float(res['inj_rate'])
        failed   = bool(res['failed'])
        if not failed and not np.all(np.isfinite([prod, inj, pviol, p_max])):
            failed = True

        self.steps_taken += 1
        at_horizon = self.steps_taken >= self.T

        if failed:
            terminated, truncated = True, False
            r_prod, r_penalty = 0.0, 0.0
            info_status = "solver_failed"
            p_norm = self.p_history[-1].copy()
        else:
            terminated, truncated = at_horizon, False
            r_prod    = prod
            r_penalty = -self.lambdaP * pviol
            info_status = "normal"
            p_norm, _ = self._pressure_map(res)
        total_reward = r_prod + r_penalty

        self.p_history.append(p_norm)
        self.u_history.append(u_norm)

        info = {
            "rate_mult": rate_mult, "inj_rate": inj_rate, "max_pressure": p_max,
            "prod": prod, "inj": inj, "pviol": pviol,
            "r_prod": r_prod, "r_penalty": r_penalty,
            "total_reward": total_reward,
            "status": info_status,
            "map_idx": int(self.map_idx),
            "t": self.steps_taken,
        }
        return self._obs(), float(total_reward), terminated, truncated, info

    def close(self):
        if getattr(self, "eng", None) is not None:
            self.eng.quit()


# =====================================================================
# 2. Feature extractor : grid -> CNN(256), vec(H+1)
# =====================================================================
class CombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, cnn_features_dim: int = 256):
        n_vec = int(observation_space["vec"].shape[0])
        super().__init__(observation_space, features_dim=cnn_features_dim + n_vec)
        n_in = int(observation_space["grid"].shape[0])
        self.cnn = nn.Sequential(
            nn.Conv2d(n_in, 32, 3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),  nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.AdaptiveMaxPool2d((4, 4)),
            nn.Flatten(),
        )
        self.linear = nn.Sequential(nn.Linear(64 * 4 * 4, cnn_features_dim), nn.ReLU())

    def forward(self, obs):
        return th.cat([self.linear(self.cnn(obs["grid"])), obs["vec"]], dim=1)


# =====================================================================
# 3. Callbacks
# =====================================================================
class TrainingCurveCallback(BaseCallback):
    """Stores per-step rewards and per-episode return / production / penalty for the plots."""
    def __init__(self, num_envs=8, verbose=0):
        super().__init__(verbose)
        self.num_envs = num_envs
        self.timesteps, self.step_rewards = [], []
        self.ep_return, self.ep_prod, self.ep_pen = [], [], []
        self.ep_len, self.ep_failed, self.ep_map = [], [], []
        self._acc = np.zeros((num_envs, 3))
        self._len = np.zeros(num_envs, dtype=int)

    def _on_step(self) -> bool:
        dones = self.locals['dones']
        infos = self.locals['infos']

        raw_r = np.array([i.get('total_reward', 0.0) for i in infos], dtype=float)
        self.step_rewards.append(float(np.mean(raw_r)))
        self.timesteps.append(self.num_timesteps)

        for i in range(self.num_envs):
            inf = infos[i]
            self._acc[i] += (raw_r[i], inf.get('r_prod', 0.0), inf.get('r_penalty', 0.0))
            self._len[i] += 1
            if dones[i]:
                ret, pr, pe = self._acc[i]
                self.ep_return.append(float(ret)); self.ep_prod.append(float(pr))
                self.ep_pen.append(float(pe))
                self.ep_len.append(int(self._len[i]))
                self.ep_failed.append(inf.get('status') == "solver_failed")
                self.ep_map.append(int(inf.get('map_idx', -1)))
                self._acc[i] = 0.0
                self._len[i] = 0
        return True


class EntCoefScheduleCallback(BaseCallback):
    """Linear entropy-coefficient schedule from `start` to `end` over all PPO updates."""
    def __init__(self, start, end, total_timesteps, verbose=0):
        super().__init__(verbose)
        if total_timesteps <= 0:
            raise ValueError("total_timesteps must be positive.")
        self.start, self.end = float(start), float(end)
        self.total_timesteps = int(total_timesteps)
        self.n_updates = None
        self._j = 0
        self.history = []

    def value(self, j):
        if self.n_updates is None or self.n_updates <= 1:
            return self.end
        frac = min(max(j / (self.n_updates - 1), 0.0), 1.0)
        return self.start + (self.end - self.start) * frac

    def _on_training_start(self):
        per_update = int(self.model.n_steps) * int(self.model.n_envs)
        self.n_updates = int(np.ceil(self.total_timesteps / per_update))
        self._j = 0
        self.model.ent_coef = self.start

    def _on_rollout_end(self):
        v = self.value(self._j)
        self.model.ent_coef = v
        self.history.append((self._j, int(self.model.num_timesteps), v))
        self.logger.record("train/ent_coef", v)
        self._j += 1

    def _on_step(self) -> bool:
        return True


def make_env():
    def _init():
        return QFSSinglePhaseEnv()
    return _init


# =====================================================================
# 3b. Evaluation-map selection (by a_k percentiles of the train maps)
# =====================================================================
def select_eval_maps(env, train_maps, percentiles, csv_path):
    """Probe a_k on every train map and return the maps closest to the given a_k percentiles."""
    train_maps = np.asarray(train_maps, dtype=int)
    print(f"\n[eval maps] train maps total {len(train_maps)} a_k computing (1 substep per map)...")
    t0 = time.time()
    a_k = np.full(len(train_maps), np.nan)
    for i, k in enumerate(train_maps):
        a_k[i] = env.probe_a_k(int(k))
        if (i + 1) % 50 == 0 or i + 1 == len(train_maps):
            print(f"  {i+1}/{len(train_maps)}  {time.time()-t0:7.1f}s", flush=True)

    np.savetxt(csv_path, np.column_stack([train_maps, a_k]), delimiter=",",
               header="map_idx,a_k_bar", comments="", fmt=["%d", "%.6g"])
    print(f"[eval maps] a_k table -> {csv_path}")

    ok = np.isfinite(a_k)
    if ok.sum() < len(percentiles):
        raise RuntimeError(f"a_k was obtained for only {ok.sum()} maps; cannot pick {len(percentiles)} evaluation maps.")
    maps_ok, ak_ok = train_maps[ok], a_k[ok]
    if ok.sum() < len(train_maps):
        print(f"[eval maps] unsolved: {train_maps[~ok].tolist()}")

    targets = np.percentile(ak_ok, percentiles)
    chosen = []
    for q in targets:
        for j in np.argsort(np.abs(ak_ok - q), kind="stable"):
            if int(maps_ok[j]) not in chosen:
                chosen.append(int(maps_ok[j]))
                break

    ak_of = dict(zip(maps_ok.tolist(), ak_ok.tolist()))
    rows = sorted(zip(percentiles, targets, chosen), key=lambda r: ak_of[r[2]])
    print(f"[eval maps] whole a_k : min={ak_ok.min():.1f}, median={np.median(ak_ok):.1f}, max={ak_ok.max():.1f} bar")
    print("   pct   target a_k    map     a_k")
    for pct, q, k in rows:
        print(f"  {pct:4.0f}   {q:10.1f}   {k:5d}  {ak_of[k]:7.1f}")
    return [r[2] for r in rows], a_k


class FixedMapEvalEnv(QFSSinglePhaseEnv):
    """Same environment, but reset() cycles through a fixed list of evaluation maps."""
    def __init__(self, map_list=None):
        super().__init__()
        self._eval_maps = list(map_list) if map_list else []
        self._eval_ptr = 0

    def set_eval_maps(self, map_list):
        self._eval_maps = list(map_list)
        self._eval_ptr = 0

    def reset(self, seed=None, options=None):
        if options is None or options.get('map_idx') is None:
            if not self._eval_maps:
                raise RuntimeError("The evaluation-map list is empty. Call set_eval_maps() first.")
            k = self._eval_maps[self._eval_ptr % len(self._eval_maps)]
            self._eval_ptr += 1
            options = {'map_idx': k}
        return super().reset(seed=seed, options=options)


# =====================================================================
# 3c. Statistics and report functions (same definitions and format as qfs_mc_baseline.py)
# =====================================================================
def count_cols():
    return ['n_ge_Plim']


VALUE_COLS = ['p_peak', 'J', 'prod', 'penalty', 'inj', 'u_mean']


def episode_stats(k, u, p_max_w, prod_w, pen_w, inj_w, p_limit):
    """One table row for one episode (= one map, T windows)."""
    pm = np.asarray(p_max_w, float)
    d = dict(map_idx=int(k), n_ge_Plim=int(np.sum(pm >= p_limit)))
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


def rl_test_table(recs, p_limit):
    import pandas as pd
    rows = []
    for r in recs:
        if r['status'] != 'normal':
            continue
        d = episode_stats(r['k'], r['u_best'], r['p_max_w'], r['prod_w'], r['pen_w'], r['inj_w'], p_limit)
        assert np.isclose(d['J'], r['J_best'], rtol=1e-12, atol=1e-9), (d['J'], r['J_best'])
        d.update(a_k=r['a_k'], n_windows=r['n_win'], status=r['status'],
                 failed=int(r['status'] != 'normal'), sec=r['sec'])
        rows.append(d)
    return pd.DataFrame(rows).sort_values('map_idx').reset_index(drop=True)


def write_latex(df, agg, path, caption):
    cc = count_cols()
    heads = {'n_ge_Plim': r'$\ge P_{\lim}$', 'p_peak': r'max $p_{\max}$ [bar]',
             'J': r'$J$ [m$^3$]', 'prod': r'Prod. [m$^3$]', 'penalty': r'Penalty [m$^3$]'}
    cols = cc + ['p_peak', 'J', 'prod', 'penalty']
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


def best_episode_df(r, p_limit):
    import pandas as pd
    T = len(r['u_best'])
    return pd.DataFrame(dict(
        map_idx=r['k'], window=np.arange(1, T + 1), u=r['u_best'], p_max_bar=r['p_max_w'],
        hit=(r['p_max_w'] >= p_limit).astype(int), produced=r['prod_w'], injected=r['inj_w'],
        penalty=r['pen_w'], reward=r['prod_w'] + r['pen_w'],
        cum_production=np.cumsum(r['prod_w']), cum_J=np.cumsum(r['prod_w'] + r['pen_w'])
    ))


def print_stats(df, agg):
    cc = count_cols()
    m, s, h = agg.iloc[0], agg.iloc[1], agg.iloc[-1]
    print(f"\n==== test-set stats over {len(df)} maps (std ddof={STD_DDOF}) ====")
    for c in cc + VALUE_COLS:
        hit = f"   maps with hit: {int(h[c])}/{len(df)}" if c in cc else ""
        print(f"  {c:12s}: {m[c]:10.3f} ± {s[c]:9.3f}   [min {agg.iloc[3][c]:.3f}, max {agg.iloc[4][c]:.3f}]{hit}")


# =====================================================================
# 3d. Plotting (RL-specific plots + plots in the same style as the MC baseline)
# =====================================================================
def _ma(x, w):
    x = np.asarray(x, dtype=float)
    if len(x) < w:
        return None, None
    return np.convolve(x, np.ones(w) / w, mode='valid'), np.arange(w, len(x) + 1)


def _step_xy(u):
    return np.arange(len(u) + 1) + 1.0, np.append(u, u[-1])


def _limit_lines(a, p_limit):
    a.axhline(p_limit, color='purple', ls='--', lw=2, label=f'$P_{{lim}}$ = {p_limit:.0f} bar (penalty)')


def plot_policy(env, tag, k_idx, mult, p_maxh, prod_w, pen_w, inj_w, J, status, save_paths):
    """2x2 figure for one map: multiplier, max pressure, per-window volumes, cumulative production / J."""
    T = env.T
    win_axis = np.arange(1, len(mult) + 1)
    prod_arr = np.asarray(prod_w, dtype=float)
    pen_arr  = np.asarray(pen_w, dtype=float)

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    title = f'{tag}: learned policy on kmap #{k_idx}  (J={J:.1f})'
    if status != "normal":
        title += f'  [{status}]'
    fig.suptitle(title)

    n_w = len(mult)
    edges = np.arange(n_w + 1) + 1.0
    ax[0, 0].step(edges, np.append(mult, mult[-1]) if n_w else edges * np.nan,
                  where='post', color='darkgreen', lw=2, zorder=3, label='policy u')
    for e in edges:
        ax[0, 0].axvline(e, color='gray', lw=0.5, alpha=0.4, zorder=1)
    ax[0, 0].set_ylim(env.mult_min - 0.1, env.mult_max + 0.1)
    ax[0, 0].axhline(env.mult_min, color='k', ls=':', lw=1)
    ax[0, 0].axhline(env.mult_max, color='k', ls=':', lw=1)
    ax[0, 0].set_xlim(0.5, T + 1.5)
    ax[0, 0].set_title('Injection multiplier')
    ax[0, 0].set_xlabel(f'Window  (u held constant for {env.days_per_window:g} d each)')
    ax[0, 0].set_xticks(np.arange(1, T + 1))
    ax[0, 0].legend(loc='center right'); ax[0, 0].grid(alpha=0.3, axis='y')

    ax[0, 1].plot(win_axis, p_maxh, 'o-', color='red', lw=2)
    _limit_lines(ax[0, 1], env.Pmax_limit)
    ax[0, 1].set_xlim(0.5, T + 0.5)
    ax[0, 1].set_title('Max pressure (bar)'); ax[0, 1].set_xlabel('Window')
    ax[0, 1].legend(); ax[0, 1].grid(alpha=0.3)

    ax[1, 0].plot(win_axis, prod_arr, 'o-', color='green', lw=2, label='Produced')
    ax[1, 0].plot(win_axis, inj_w, 's--', color='blue', lw=2, label='Injected')
    ax[1, 0].set_xlim(0.5, T + 0.5)
    ax[1, 0].set_title('Per window (m^3)'); ax[1, 0].set_xlabel('Window')
    ax[1, 0].legend(); ax[1, 0].grid(alpha=0.3)

    ax[1, 1].plot(win_axis, np.cumsum(prod_arr), 'o-', color='green', lw=2, label='cum. production')
    ax[1, 1].plot(win_axis, np.cumsum(prod_arr + pen_arr), 's--', color='k', lw=1.5, label='cum. J (prod + penalty)')
    ax[1, 1].set_xlim(0.5, T + 0.5)
    ax[1, 1].set_title('Cumulative (m^3)'); ax[1, 1].set_xlabel('Window')
    ax[1, 1].legend(); ax[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    for p in save_paths:
        plt.savefig(p, dpi=200)
    plt.close()


def plot_overlay_mc_style(recs, tag, info, path):
    """Overlay of the 2x2 panels for a few selected maps."""
    T = info['n_windows']; w = np.arange(1, T + 1)
    cmap = plt.get_cmap('tab10')
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"{tag}: RL policy u, overlay of kmaps {[r['k'] for r in recs]}")
    for j, r in enumerate(recs):
        c = cmap(j % 10); k = r['k']
        ax[0, 0].step(*_step_xy(r['u_best']), where='post', color=c, lw=2, label=f"k{k} (J={r['J_best']:.1f})")
        ax[0, 1].plot(w, r['p_max_w'], 'o-', color=c, lw=2, label=f"k{k} (peak={np.max(r['p_max_w']):.1f})")
        ax[1, 0].plot(w, r['prod_w'], 'o-', color=c, lw=2, label=f"k{k} produced")
        ax[1, 0].plot(w, r['inj_w'], 's--', color=c, lw=1.5, alpha=.8, label=f"k{k} injected")
        ax[1, 1].plot(w, np.cumsum(r['prod_w']), 'o-', color=c, lw=2, label=f"k{k} cum. prod")
        ax[1, 1].plot(w, np.cumsum(r['prod_w'] + r['pen_w']), 's--', color=c, lw=1.5, label=f"k{k} cum. J")
    ax[0, 0].axhline(info['mult_min'], color='k', ls=':', lw=1)
    ax[0, 0].axhline(info['mult_max'], color='k', ls=':', lw=1)
    ax[0, 0].set_ylim(info['mult_min'] - 0.1, info['mult_max'] + 0.1)
    _limit_lines(ax[0, 1], info['Pmax_limit_env'])
    for a, t in zip(ax.flat, ['Injection multiplier', 'Max pressure (bar)', 'Per window (m^3)', 'Cumulative (m^3)']):
        a.set_title(t); a.set_xlabel('Window'); a.grid(alpha=.3); a.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()


def plot_envelope_mc_style(recs, tag, info, path):
    """All maps: thin line per map, bold mean, shaded mean +/- std."""
    T = info['n_windows']; w = np.arange(1, T + 1)
    p_lim = info['Pmax_limit_env']
    Y = [np.array([r['u_best'] for r in recs]),
         np.array([r['p_max_w'] for r in recs]),
         np.array([r['prod_w'] for r in recs]),
         np.array([np.cumsum(r['prod_w'] + r['pen_w']) for r in recs])]
    titles = ['Injection multiplier', 'Max pressure (bar)', 'Production per window (m^3)', 'Cumulative J (m^3)']
    hit_frac = np.mean(Y[1] >= p_lim, axis=0)
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"{tag}: RL policy u over {len(recs)} maps (thin: each map, bold: mean, band: ±std, ddof={STD_DDOF})")
    for a, y, t in zip(ax.flat, Y, titles):
        for row in y:
            a.plot(w, row, color='gray', lw=.6, alpha=.4)
        mu, sd = y.mean(0), y.std(0, ddof=STD_DDOF) if len(y) > STD_DDOF else np.zeros(T)
        a.plot(w, mu, 'o-', color='k', lw=2, label='mean')
        a.fill_between(w, mu - sd, mu + sd, color='tab:blue', alpha=.25, label='mean ± std')
        a.set_title(t); a.set_xlabel('Window'); a.grid(alpha=.3)
    ax[0, 0].set_ylim(info['mult_min'] - 0.1, info['mult_max'] + 0.1)
    _limit_lines(ax[0, 1], p_lim)
    b = ax[0, 1].twinx()
    b.bar(w, hit_frac, color='red', alpha=.15, width=.6, label='fraction of maps hit')
    b.set_ylim(0, 1); b.set_ylabel(f'fraction with p_max >= {p_lim:.0f}')
    for a in ax.flat:
        a.legend(fontsize=8, loc='best')
    b.legend(fontsize=8, loc='lower right')
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()


def plot_grid_mc_style(recs, tag, info, out_dir):
    """Small-multiples page per 25 maps: u (left axis) and window p_max (right axis)."""
    T = info['n_windows']; w = np.arange(1, T + 1)
    p_lim = info['Pmax_limit_env']
    allp = np.concatenate([r['p_max_w'] for r in recs])
    pmin = np.nanmin(allp); pmax = max(np.nanmax(allp), p_lim)
    pad = 0.05 * (pmax - pmin + 1e-9)
    paths = []
    for p, p0 in enumerate(range(0, len(recs), GRID_PER_PAGE)):
        chunk = recs[p0:p0 + GRID_PER_PAGE]
        nrow = int(np.ceil(len(chunk) / GRID_NCOL))
        fig, axs = plt.subplots(nrow, GRID_NCOL, figsize=(4.2 * GRID_NCOL, 2.9 * nrow), squeeze=False)
        for a in axs.flat[len(chunk):]:
            a.axis('off')
        for a, r in zip(axs.flat, chunk):
            pm = r['p_max_w']; hit = pm >= p_lim
            a.step(*_step_xy(r['u_best']), where='post', color='green', lw=1.5)
            a.set_ylim(info['mult_min'] - 0.05, info['mult_max'] + 0.05)
            a.tick_params(labelsize=7); a.grid(alpha=.3)
            b = a.twinx()
            b.plot(w, pm, '-', color='tab:blue', lw=1.2)
            b.plot(w[hit], pm[hit], 'x', color='red', ms=6, mew=2)
            b.axhline(p_lim, color='purple', ls='--', lw=1)
            b.set_ylim(pmin - pad, pmax + pad); b.tick_params(labelsize=7, colors='tab:blue')
            a.set_title(f"k{r['k']}: J={r['J_best']:.1f} peak={np.max(pm):.1f} hit={int(hit.sum())}", fontsize=8)
        handles = [plt.Line2D([], [], color='green', lw=1.5, label='multiplier u (left axis)'),
                   plt.Line2D([], [], color='tab:blue', lw=1.2, label='window p_max [bar] (right axis)'),
                   plt.Line2D([], [], color='purple', ls='--', label=f'P_lim = {p_lim:.0f}'),
                   plt.Line2D([], [], color='red', marker='x', ls='', mew=2, label='p_max >= P_lim')]
        fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=9)
        fig.suptitle(f"{tag}: maps {chunk[0]['k']}–{chunk[-1]['k']} (RL policy per map)", fontsize=11)
        plt.tight_layout(rect=(0, 0.03, 1, 0.97))
        path = os.path.join(out_dir, f"grid_maps_p{p + 1}.png")
        plt.savefig(path, dpi=150); plt.close(); paths.append(path)
    return paths


def write_rl_report(recs, info, tag, out_dir, overlay_maps, total_timesteps, ent_start, ent_end):
    """Write CSV / XLSX / LaTeX tables and figures (same layout as the MC baseline report)."""
    import pandas as pd
    recs_ok = sorted([r for r in recs if r['status'] == 'normal'], key=lambda r: r['k'])
    if not recs_ok:
        print("[report] no test map finished normally; report skipped.", flush=True)
        return None
    M = len(recs_ok)
    p_lim = info['Pmax_limit_env']
    n_win = info['n_windows']
    days_w = info['days_per_window']

    df = rl_test_table(recs_ok, p_lim)
    agg = aggregate_stats(df)
    long = pd.concat([best_episode_df(r, p_lim) for r in recs_ok], ignore_index=True)
    settings = pd.DataFrame(dict(
        item=['Method', 'P_limit [bar]', 'N_WINDOWS', 'DAYS_PER_WINDOW', 'lambdaP',
              'u range', 'warm-up', 'total_timesteps', 'ent_coef_schedule', 'std ddof',
              'n_maps', 'maps', 'hit definition'],
        value=['PPO (deterministic policy)', p_lim, n_win, days_w, info['lambdaP'],
               f"[{info['mult_min']}, {info['mult_max']}]",
               f"{info['n_warmup']} win, u={info['warmup_mults']}",
               total_timesteps, f"linear {ent_start} -> {ent_end}", STD_DDOF,
               M, ' '.join(str(r['k']) for r in recs_ok),
               'window t hit <=> p_max_t >= P_limit']))

    tables = [('test_table', df), ('test_stats', agg), ('settings', settings), ('best_episodes_long', long)]
    for name, t in tables:
        t.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)
    write_latex(df, agg, os.path.join(out_dir, "table_test.tex"),
                caption=(f"PPO learned policy (deterministic): control windows (out of {n_win}) "
                         f"whose window max pressure reaches $P_{{\\lim}}={p_lim:.0f}$ bar, over {M} test maps. "
                         f"std with ddof={STD_DDOF}."))

    ov = [r for r in recs_ok if r['k'] in set(overlay_maps)]
    figs = []
    if ov:
        plot_overlay_mc_style(ov, tag, info, os.path.join(out_dir, "overlay_maps.png"))
        figs.append("overlay_maps.png")
    plot_envelope_mc_style(recs_ok, tag, info, os.path.join(out_dir, "envelope_all.png"))
    figs.append("envelope_all.png")
    figs += [os.path.basename(p) for p in plot_grid_mc_style(recs_ok, tag, info, out_dir)]
    for r in ov:
        ep_rel = os.path.join('per_map', f"episode_k{r['k']}.png")
        if os.path.isfile(os.path.join(out_dir, ep_rel)):
            figs.append(ep_rel)

    try:
        import openpyxl  # noqa: F401
    except ImportError:
        print("[report] openpyxl not installed -> only CSV/TeX written", flush=True)
        return None

    path = os.path.join(out_dir, f"report_{tag}.xlsx")
    with pd.ExcelWriter(path, engine='openpyxl') as xw:
        for name, t in tables:
            t.to_excel(xw, sheet_name=name, index=False)
        for r in ov:
            best_episode_df(r, p_lim).to_excel(xw, sheet_name=f"k{r['k']}_best_episode", index=False)
        pd.DataFrame().to_excel(xw, sheet_name='figures')
    try:
        from openpyxl import load_workbook
        from openpyxl.drawing.image import Image as XLImage
        wb = load_workbook(path); ws = wb['figures']; row = 1
        for name in figs:
            img_p = os.path.join(out_dir, name)
            if os.path.isfile(img_p):
                img = XLImage(img_p)
                img.width, img.height = img.width * 0.35, img.height * 0.35
                ws.add_image(img, f"A{row}"); row += int(img.height / 20) + 2
        wb.save(path)
    except Exception as e:
        print(f"[report] could not embed figures ({e}); tables were saved", flush=True)
    return path


def plot_reward_breakdown(env, tag, cb, w=EP_MA_WINDOW):
    n_ep = len(cb.ep_return)
    fig, ax = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)
    ax[0].set_title(f'{tag}: reward components per episode (moving avg, {w} episodes)')

    if n_ep >= w:
        m, x = _ma(cb.ep_prod, w)
        ax[0].plot(x, m, color='green', lw=1.8, label=r'$\sum_t r_{prod,t}$  (cumProduced)')
        m, x = _ma(cb.ep_pen, w)
        ax[1].plot(x, m, color='purple', lw=1.8,
                   label=r'$\sum_t r_{penalty,t}$  ($-\lambda_P\,$pressurePenalty)')

    ax[0].set_ylabel('m^3 / episode'); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=9)
    if n_ep >= w:
        lo = float(np.min(_ma(cb.ep_pen, w)[0]))
        ax[1].set_ylim(1.05 * lo if lo < 0 else -1.0, max(0.05 * abs(lo), 1.0))
    ax[1].set_ylabel('m^3 / episode'); ax[1].set_xlabel('Episode')
    ax[1].grid(alpha=0.3); ax[1].legend(fontsize=9, loc='lower left')
    plt.tight_layout(); plt.savefig(f"reward_breakdown_{tag}.png", dpi=200); plt.close()


def plot_training_curve(tag, cb, w_step=STEP_MA_WINDOW, w_ep=EP_MA_WINDOW):
    fig, ax = plt.subplots(2, 1, figsize=(12, 9))

    ax[0].plot(cb.timesteps, cb.step_rewards, color='blue', alpha=0.3, label='Step reward (mean over envs)')
    m, x = _ma(cb.step_rewards, w_step)
    if m is not None:
        ax[0].plot(np.asarray(cb.timesteps)[x - 1], m, color='red', lw=2, label=f'Moving avg ({w_step})')
    ax[0].set_xlabel('Timesteps'); ax[0].set_ylabel('Step reward (m^3)')
    ax[0].legend(); ax[0].grid(alpha=0.3); ax[0].set_title(f'{tag}: training')

    ep_ax = np.arange(1, len(cb.ep_return) + 1)
    ax[1].plot(ep_ax, cb.ep_return, 'o', ms=2, color='green', alpha=0.4, label='Episode return')
    m, x = _ma(cb.ep_return, w_ep)
    if m is not None:
        ax[1].plot(x, m, color='darkgreen', lw=2, label=f'Moving avg ({w_ep})')
    ax[1].set_xlabel('Episode'); ax[1].set_ylabel('Episode return = J (m^3)')
    ax[1].legend(); ax[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"trainingcurve_{tag}.png", dpi=200); plt.close()


def plot_reward_per_window(env, tag, recs):
    n = len(recs)
    if n == 0:
        return
    T = env.T
    fig, ax = plt.subplots(n, 2, figsize=(13, 4.2 * n), squeeze=False)
    fig.suptitle(f'{tag}: per-window reward on fixed test maps (deterministic policy)')

    for i, r in enumerate(recs):
        rp = np.asarray(r['prod_w'], dtype=float)
        rn = np.asarray(r['pen_w'],  dtype=float)
        rt = np.asarray(r['r_w'],    dtype=float)
        w  = np.arange(1, len(rt) + 1)
        st = '' if r['status'] == 'normal' else f"  [{r['status']}]"

        a = ax[i, 0]
        a.bar(w, rp, width=0.6, color='green',  alpha=0.55, label=r'$r_{prod,t}$')
        a.bar(w, rn, width=0.6, color='purple', alpha=0.55, label=r'$r_{penalty,t}=-\lambda_P\,$pviol$_t$')
        a.plot(w, rt, 'o-', color='black', lw=1.8, ms=5, label=r'$r_t=r_{prod,t}+r_{penalty,t}$')
        a.axhline(0.0, color='k', lw=0.8)
        if len(rt):
            y_hi = max(float(np.max(rp)), float(np.max(rt)), 0.0)
            y_lo = min(float(np.min(rn)), float(np.min(rt)), 0.0)
            span = max(y_hi - y_lo, 1.0)
            a.set_ylim(y_lo - 0.05 * span, y_hi + 0.30 * span)
        a.set_xlim(0.5, T + 0.5); a.set_xticks(np.arange(1, T + 1))
        a.set_title(f"kmap #{r['k']}: reward breakdown per window{st}")
        a.set_xlabel('Window'); a.set_ylabel('m^3 / window')
        a.grid(alpha=0.3); a.legend(fontsize=8, loc='upper center', ncol=3)

        a = ax[i, 1]
        a.plot(w, np.cumsum(rt), 'o-', color='black', lw=2, ms=5, label=r'$\sum_{s\leq t} r_s$')
        a.plot(w, np.cumsum(rp), '--', color='green',  lw=1.2, label=r'$\sum_{s\leq t} r_{prod,s}$')
        a.plot(w, np.cumsum(rn), '--', color='purple', lw=1.2, label=r'$\sum_{s\leq t} r_{penalty,s}$')
        a.axhline(0.0, color='k', lw=0.8)
        a.set_xlim(0.5, T + 0.5); a.set_xticks(np.arange(1, T + 1))
        a.set_title(f"kmap #{r['k']}: cumulative total reward (J={r['J_best']:.1f}){st}")
        a.set_xlabel('Window'); a.set_ylabel('m^3')
        a.grid(alpha=0.3); a.legend(fontsize=8, loc='upper left')

    plt.tight_layout()
    plt.savefig(f"reward_per_window_{tag}.png", dpi=200); plt.close()


def plot_test_overlay(env, tag, rec):
    if not rec:
        return
    T = env.T
    ak = np.array([r['a_k'] for r in rec], dtype=float)
    fin = np.isfinite(ak)
    cmap = plt.get_cmap('viridis')
    if fin.any():
        lo, hi = float(ak[fin].min()), float(ak[fin].max())
        norm = matplotlib.colors.Normalize(vmin=lo, vmax=hi if hi > lo else lo + 1.0)
    else:
        norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(f'{tag}: all {len(rec)} test maps '
                 f'(kmap {rec[0]["k"]}-{rec[-1]["k"]}, deterministic policy)')

    full_p, full_c = [], []
    for r, a_k in zip(rec, ak):
        p = np.asarray(r['p_max_w'], dtype=float)
        c = np.cumsum(np.asarray(r['prod_w'], dtype=float))
        w = np.arange(1, len(p) + 1)
        col = cmap(norm(a_k)) if np.isfinite(a_k) else (0.5, 0.5, 0.5, 1.0)
        ax[0].plot(w, p, '-', color=col, lw=1.0, alpha=0.7)
        ax[1].plot(w, c, '-', color=col, lw=1.0, alpha=0.7)
        if len(p) == T and r['status'] == 'normal':
            full_p.append(p); full_c.append(c)

    if full_p:
        w = np.arange(1, T + 1)
        ax[0].plot(w, np.mean(full_p, axis=0), 'k--', lw=2.2, label=f'mean ({len(full_p)} maps)')
        ax[1].plot(w, np.mean(full_c, axis=0), 'k--', lw=2.2, label=f'mean ({len(full_c)} maps)')

    ax[0].axhline(env.Pmax_limit, color='purple', ls='--', lw=2,
                  label=f'$P_{{max,limit}}$ = {env.Pmax_limit:.0f} (penalty threshold)')
    ax[0].set_title('Max pressure (bar)'); ax[1].set_title('Cumulative production (m^3)')
    for a in ax:
        a.set_xlim(0.5, T + 0.5); a.set_xticks(np.arange(1, T + 1))
        a.set_xlabel('Window'); a.grid(alpha=0.3); a.legend(fontsize=9, loc='best')

    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    fig.colorbar(sm, ax=list(ax), shrink=0.9, pad=0.02, label=r'$a_k$ (bar)  [gray = not probed]')
    plt.savefig(f"test_overlay_{tag}.png", dpi=200, bbox_inches='tight'); plt.close()


# =====================================================================
# 4. Train & Test
# =====================================================================
if __name__ == '__main__':
    # Defaults reproduce the reported run. The options only exist to make a short smoke test possible.
    ap = argparse.ArgumentParser(description="Train PPO and evaluate it on the test maps.")
    ap.add_argument("--total-timesteps", type=int, default=250000)
    ap.add_argument("--num-envs", type=int, default=8,
                    help="parallel training environments (one MATLAB engine each)")
    ap.add_argument("--eval-maps", type=int, nargs="*", default=None,
                    help="give the evaluation maps directly and skip the a_k probe of all train maps")
    ap.add_argument("--test-maps", type=int, nargs="*", default=None,
                    help="subset of test maps to roll out (default: all test maps)")
    args = ap.parse_args()

    job_id = os.environ.get('SLURM_JOB_ID', 'local')
    TAG      = f"qfs_sp_pomdp3_warmup_schedule_job{job_id}"
    num_cpu  = args.num_envs
    total_timesteps = args.total_timesteps
    ENT_COEF_START, ENT_COEF_END = 0.05, 0.005

    # VecMonitor only adds episode statistics to the SB3 log (rollout/ep_rew_mean);
    # it does not change the environment or the training.
    env = VecMonitor(SubprocVecEnv([make_env() for _ in range(num_cpu)]))

    raw_eval_env = FixedMapEvalEnv()
    print(f"[split] {KMAP_FILE}: n_unique={raw_eval_env.n_unique} -> "
          f"train {raw_eval_env.train_map_lo}-{raw_eval_env.train_map_hi}, "
          f"test {raw_eval_env.test_map_indices[0]}-{raw_eval_env.test_map_indices[-1]} "
          f"(N={len(raw_eval_env.test_map_indices)})")
    print(f"[reward] extractQuarterFiveSpotQoI J: r_t = prod_t - lambda_P * pviol_t, "
          f"Pmax_limit={raw_eval_env.Pmax_limit:g} bar, lambda_P={raw_eval_env.lambdaP:g} m^3/(bar^2 day) "
          f"| no hard limit, no early termination")
    print(f"[time  ] Δt_control={raw_eval_env.days_per_window:g} d x {raw_eval_env.n_windows} windows "
          f"= {raw_eval_env.total_days:g} d, Δτ={raw_eval_env.dt_days:g} d")
    print(f"[warmup] n_warmup={raw_eval_env.n_warmup} windows "
          f"(= {raw_eval_env.n_warmup * raw_eval_env.days_per_window:g} d, excluded from reward/J), "
          f"warmup_mults={raw_eval_env.warmup_mults} "
          f"(u_norm={[round(u, 3) for u in raw_eval_env.u_norm_warm]})")
    if raw_eval_env.dt_days != 1.0:
        print("[warn  ] dt_days != 1 -> sum of pviol is a trapz on a coarser grid, "
              "not identical to the reference J evaluated with 1-day steps.")

    train_maps = np.arange(raw_eval_env.train_map_lo, raw_eval_env.train_map_hi + 1)
    if args.eval_maps:
        EVAL_MAPS, a_k_train = list(args.eval_maps), np.array([])
        print("[eval maps] given on the command line -> a_k probe of the train maps skipped")
    else:
        EVAL_MAPS, a_k_train = select_eval_maps(raw_eval_env, train_maps, EVAL_PERCENTILES,
                                                csv_path=f"a_k_train_{TAG}.csv")
    raw_eval_env.set_eval_maps(EVAL_MAPS)
    print(f"[eval maps] EVAL_MAPS = {EVAL_MAPS}\n")

    if raw_eval_env.n_warmup > 0 and len(a_k_train) > 0:
        m_warm_max = max(raw_eval_env.warmup_mults)
        ak_fin  = a_k_train[np.isfinite(a_k_train)]
        ak_crit = (raw_eval_env.Pmax_limit - raw_eval_env.P_bhp) / m_warm_max
        n_over  = int(np.sum(raw_eval_env.P_bhp + m_warm_max * ak_fin > raw_eval_env.Pmax_limit))
        print(f"[warmup] train maps whose warm-up p_max > Pmax_limit: {n_over}/{len(ak_fin)} "
              f"({100 * n_over / max(len(ak_fin), 1):.1f} %, a_k > {ak_crit:.1f} bar)\n")

    eval_env = DummyVecEnv([lambda: Monitor(raw_eval_env)])

    callback = TrainingCurveCallback(num_envs=num_cpu)

    save_dir = f"./best_model_dir_{TAG}"
    os.makedirs(save_dir, exist_ok=True)
    eval_callback = EvalCallback(
        eval_env, best_model_save_path=save_dir, log_path=save_dir,
        eval_freq=1000,
        n_eval_episodes=len(EVAL_MAPS),
        deterministic=True, render=False,
    )
    ent_callback = EntCoefScheduleCallback(ENT_COEF_START, ENT_COEF_END, total_timesteps)
    callback_list = CallbackList([callback, ent_callback, eval_callback])

    policy_kwargs = dict(
        features_extractor_class=CombinedExtractor,
        features_extractor_kwargs=dict(cnn_features_dim=256),
        normalize_images=False,
    )

    model = PPO(
        "MultiInputPolicy", env, verbose=1,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=256,
        gamma=1.0,
        ent_coef=ENT_COEF_START,
        policy_kwargs=policy_kwargs,
        seed=0,
    )

    print(f"[ppo   ] total_timesteps={total_timesteps}, "
          f"ent_coef linear {ENT_COEF_START:g} -> {ENT_COEF_END:g} over PPO updates")
    print("Training start...")
    model.learn(total_timesteps=total_timesteps, callback=callback_list)
    print("Finished.")
    if ent_callback.history:
        h = ent_callback.history
        print(f"[ppo   ] {len(h)} PPO updates, final num_timesteps={model.num_timesteps} | ent_coef: "
              f"first={h[0][2]:.4f}, mid={h[len(h) // 2][2]:.4f}, last={h[-1][2]:.4f}")

    # =================================================================
    # 5. Test: deterministic rollouts on the test maps + report (same format as the MC baseline)
    # =================================================================
    model.save(os.path.join(save_dir, "final_model"))

    print("\nLoading best model for test rollouts...")
    best_path = os.path.join(save_dir, "best_model.zip")
    if os.path.isfile(best_path):
        best_model = PPO.load(best_path)
        print(f"[test] using {best_path}")
    else:
        best_model = model
        print(f"[warn] {best_path} not found -> testing the final model.")

    TEST_MAPS = list(args.test_maps) if args.test_maps else list(raw_eval_env.test_map_indices)
    test_set  = set(TEST_MAPS)
    plot_maps = [k for k in PLOT_TEST_MAPS if k in test_set]
    if not plot_maps:
        plot_maps = [TEST_MAPS[0], TEST_MAPS[-1]]
        print(f"[test] PLOT_TEST_MAPS={PLOT_TEST_MAPS} is out of range of test -> {plot_maps} is substituted")
    print(f"[test] {len(TEST_MAPS)} maps rollout, overlay/plot maps = {plot_maps}\n")

    # Output folder results_<TAG>/ with a per_map/ subfolder (same layout as the MC baseline)
    out_dir = f"results_{TAG}"
    pdir = os.path.join(out_dir, "per_map")
    os.makedirs(pdir, exist_ok=True)

    env_info_dict = dict(
        lambdaP=float(raw_eval_env.lambdaP),
        n_warmup=int(raw_eval_env.n_warmup),
        warmup_mults=[float(m) for m in raw_eval_env.warmup_mults],
        mult_min=float(raw_eval_env.mult_min),
        mult_max=float(raw_eval_env.mult_max),
        dt_days=float(raw_eval_env.dt_days),
        Pmax_limit_env=float(raw_eval_env.Pmax_limit),
        n_windows=int(raw_eval_env.n_windows),
        days_per_window=float(raw_eval_env.days_per_window),
    )
    with open(os.path.join(out_dir, "env_info.json"), "w") as f:
        json.dump(env_info_dict, f, indent=2)

    rec  = []
    t_t0 = time.time()
    for k_idx in TEST_MAPS:
        t_m0 = time.time()
        a_k = raw_eval_env.probe_a_k(k_idx)

        obs, info0 = raw_eval_env.reset(options={'map_idx': k_idx})
        mult, p_maxh, prod_w, inj_w = [], [], [], []
        r_prod_w, r_pen_w, r_w = [], [], []
        J, sum_prod, sum_pen, sum_inj, status = 0.0, 0.0, 0.0, 0.0, "normal"

        if info0['warmup_failed']:
            status = "solver_failed"
        else:
            for t in range(raw_eval_env.T):
                action, _ = best_model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = raw_eval_env.step(action)
                mult.append(info['rate_mult'])
                p_maxh.append(info['max_pressure'])
                prod_w.append(info['prod'])
                inj_w.append(info['inj'])
                r_prod_w.append(info['r_prod'])
                r_pen_w.append(info['r_penalty'])
                r_w.append(float(reward))
                J        += reward
                sum_prod += info['r_prod']
                sum_pen  += info['r_penalty']
                sum_inj  += info['inj']
                status = info['status']
                if terminated or truncated:
                    break

        sec_map = time.time() - t_m0
        u_arr  = np.asarray(mult, dtype=float)
        pm_arr = np.asarray(p_maxh, dtype=float)
        pr_arr = np.asarray(r_prod_w, dtype=float)
        pe_arr = np.asarray(r_pen_w, dtype=float)
        in_arr = np.asarray(inj_w, dtype=float)
        rw_arr = np.asarray(r_w, dtype=float)

        n_hit = int(np.sum(pm_arr >= raw_eval_env.Pmax_limit)) if len(pm_arr) else 0
        p_pk  = float(np.max(pm_arr)) if len(pm_arr) else float('nan')
        u_mn  = float(np.mean(u_arr)) if len(u_arr) else float('nan')

        rec.append(dict(
            k=int(k_idx), a_k=a_k,
            J=J, J_best=J, prod=sum_prod, pen=sum_pen, inj=sum_inj,
            p_peak=p_pk, u_mean=u_mn, n_ge_Plim=n_hit,
            n_win=len(mult), status=status, sec=sec_map,
            # keys shared with the MC baseline (u_best, p_max_w, prod_w, pen_w, inj_w)
            u_best=u_arr, p_max_w=pm_arr, prod_w=pr_arr, pen_w=pe_arr, inj_w=in_arr, r_w=rw_arr
        ))

        print(f"  k={k_idx:4d} | a_k={a_k:6.1f} | J={J:9.3f} | prod={sum_prod:9.3f} "
              f"| pen={sum_pen:9.3f} | inj={sum_inj:9.3f} | p_peak={p_pk:6.2f} "
              f"| hit={n_hit:2d} | u_mean={u_mn:.3f} | status={status} | u*={np.round(u_arr, 3)}",
              flush=True)

        # per_map/episode_k<map>.png for every test map; the plot_maps are also saved in the run folder
        save_paths = [os.path.join(pdir, f"episode_k{k_idx}.png")]
        if k_idx in plot_maps:
            print(f"           warm-up: {info0['warmup_windows']}/{raw_eval_env.n_warmup} windows, "
                  f"p_max={info0['warmup_pmax']:.1f} bar, failed={info0['warmup_failed']}")
            save_paths.append(f"policy_k_{k_idx}_{TAG}.png")
        if status == "normal":
            plot_policy(raw_eval_env, TAG, k_idx, u_arr, pm_arr, pr_arr, pe_arr, in_arr, J, status, save_paths)

    print(f"[test] rollout done ({time.time() - t_t0:.1f}s)")

    # 1) Report in the same format as the MC baseline (CSV, XLSX, LaTeX, overlay, envelope, grid)
    report_path = write_rl_report(rec, env_info_dict, TAG, out_dir, plot_maps,
                                  total_timesteps, ENT_COEF_START, ENT_COEF_END)

    # 2) RL-specific plots
    rec_by_k = {r['k']: r for r in rec}
    plot_reward_per_window(raw_eval_env, TAG, [rec_by_k[k] for k in plot_maps if k in rec_by_k])
    plot_test_overlay(raw_eval_env, TAG, rec)

    # =================================================================
    # 6. Training-history plots
    # =================================================================
    print("\nPlotting training history...")
    plot_reward_breakdown(raw_eval_env, TAG, callback)
    plot_training_curve(TAG, callback)

    # =================================================================
    # 7. Final statistics (training summary + test summary)
    # =================================================================
    if len(callback.ep_return) > 0:
        n  = min(300, len(callback.ep_return))
        fl = np.asarray(callback.ep_failed[-n:], dtype=float)
        pr = float(np.mean(callback.ep_prod[-n:]))
        pe = float(np.mean(callback.ep_pen[-n:]))
        print(f"\n[train summary] Previous {n} episodes: mean return (=J)={np.mean(callback.ep_return[-n:]):.1f}, "
              f"solver-failure rate={fl.mean():.3f}")
        print(f"[train summary] mean sum r_prod={pr:.1f}, mean sum r_penalty={pe:.1f}, "
              f"|penalty|/prod={(abs(pe) / pr if pr > 0 else float('nan')):.3f}")

    df_test = rl_test_table(rec, raw_eval_env.Pmax_limit)
    if len(df_test) > 0:
        agg_test = aggregate_stats(df_test)
        print_stats(df_test, agg_test)

    sfail = np.array([r['status'] == 'solver_failed' for r in rec])
    N_all = len(rec)
    print("\n" + "=" * 78)
    print(f" TEST SUMMARY (Valid={len(df_test)}/{N_all}) | maps {TEST_MAPS[0]}-{TEST_MAPS[-1]} "
          f"| Pmax_limit={raw_eval_env.Pmax_limit:g}, lambda_P={raw_eval_env.lambdaP:g}")
    print("=" * 78)
    if len(df_test) > 0:
        print(f"  avg n_ge_Plim     (windows >= P_lim)       = {df_test['n_ge_Plim'].mean():10.3f} ± {df_test['n_ge_Plim'].std(ddof=STD_DDOF):8.3f}")
        print(f"  avg peak p_max                             = {df_test['p_peak'].mean():10.3f} ± {df_test['p_peak'].std(ddof=STD_DDOF):8.3f} bar")
        print(f"  avg test J        (sum_t r_prod + r_pen)   = {df_test['J'].mean():10.3f} ± {df_test['J'].std(ddof=STD_DDOF):8.3f} m^3")
        print(f"  avg sum r_prod    (cumProduced)            = {df_test['prod'].mean():10.3f} ± {df_test['prod'].std(ddof=STD_DDOF):8.3f} m^3")
        print(f"  avg sum r_penalty (-lambda_P*penalty)      = {df_test['penalty'].mean():10.3f} ± {df_test['penalty'].std(ddof=STD_DDOF):8.3f} m^3")
        print(f"  avg sum inj       (cumInjected)            = {df_test['inj'].mean():10.3f} ± {df_test['inj'].std(ddof=STD_DDOF):8.3f} m^3")
        print(f"  avg mean multiplier u                      = {df_test['u_mean'].mean():10.4f} ± {df_test['u_mean'].std(ddof=STD_DDOF):8.4f}")
    print(f"  solver failure                             = {int(sfail.sum()):4d}/{N_all} = {100 * sfail.mean():5.1f} %")
    print("=" * 78)
    print(f"Done. All MC-compatible outputs in {out_dir}/ (report: {report_path})")

    env.close()
    raw_eval_env.close()