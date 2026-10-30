# RL control of injection rate in a quarter five-spot reservoir (single-phase, MRST)

This repository trains a PPO agent to choose the water-injection rate in a 2D
quarter five-spot reservoir, and compares it with a uniform Monte Carlo search
baseline. The flow simulation is done by [MRST](https://www.sintef.no/projectweb/mrst/)
in MATLAB, called from Python through the MATLAB Engine for Python.

**Goal:** inject as much water as possible (more injection → more production)
while keeping the maximum reservoir pressure below a soft limit of 230 bar.
Exceeding the limit is not forbidden, but it is penalized.

---

## Repository layout

| File | What it is |
|---|---|
| `qfs_sp_solver_warmup.m` | MATLAB/MRST simulator. One call = build the model for one permeability map (reset), or simulate one 30-day control window. |
| `qfs_ppo.py` | Gymnasium environment (`QFSSinglePhaseEnv`), PPO training, and the test on the held-out maps (tables + figures). |
| `qfs_mc_baseline.py` | Baseline without RL: random search over injection schedules, separately for each test map. Writes tables in the same format as `qfs_ppo.py`. |
| `scripts/train_ppo.slurm` | Example SLURM job for OSC (training + test). |
| `scripts/mc_baseline.slurm` | Example SLURM job for OSC (baseline on the 50 test maps). |
| `requirements.txt` | Python packages. |

The permeability data file `k128x128_unique.mat` is **not** in the repository
because of its size (see [Data](#data)).

---

## Problem setup

### Physics

Single-phase water flow (Darcy's law + mass conservation,
[MRST book, Ch. 4](https://www.sintef.no/contentassets/8af8db2e42614f7fb94fb0c68f5bc256/mrst-book-2016.pdf)):

$$
\phi\, c_t \frac{\partial p}{\partial t} = \nabla \cdot \left( \frac{k}{\mu} \nabla p \right) + q .
$$

The fluid and rock are **incompressible** ($c_t = 0$), so the equation reduces to
a steady-state (elliptic) equation that is solved at every time step:

$$
-\nabla \cdot \left( \frac{k}{\mu} \nabla p \right) = q .
$$

| Quantity | Value |
|---|---|
| Grid | 128 × 128 cells, 1 m × 1 m each (128 m × 128 m) |
| Viscosity $\mu$, porosity $\phi$, density $\rho$ | 1 cP, 0.2, 1000 kg/m³ |
| Permeability $k$ | one 128 × 128 map per episode, from `k128x128_unique.mat` (values below $10^{-2}$ mD are raised to $10^{-2}$ mD to keep the linear system non-singular) |
| Initial pressure | 200 bar everywhere |
| Injector | cell (1, 1), rate-controlled: $q_{inj} = u_t \cdot q_{base}$, $q_{base} = 5$ m³/day |
| Producer | cell (128, 128), fixed bottom-hole pressure 100 bar |
| Solver time step | $\Delta\tau = 1$ day |

### Timeline of one episode

```
day 0 ─────── 90 ──────────────────────────────────────── 390
     warm-up              10 control windows (30 days each)
  3 windows, fixed     agent chooses u_t once per window
  u = 0.5, 0.7, 1.0    reward is collected here
  (no reward)
```

* **Warm-up (days 0–90).** After loading a map, 3 windows are simulated with fixed
  multipliers $u = 0.5, 0.7, 1.0$. They only fill the observation history and
  give the agent information about the map. They never count in the reward.
* **Control (days 90–390).** $T = 10$ windows of $\Delta t_{control} = 30$ days.
  In window $t$ the injection multiplier $u_t$ is held constant.

### Train / test split

The data file has $n_{unique} = 1000$ permeability maps.
Maps 1–950 are used for training, maps 951–1000 (the last 50) for testing only.
During training the agent is evaluated on 9 fixed train maps, chosen so that they
span the range of the pressure sensitivity $a_k$ (closest maps to the
10th, 20th, …, 90th percentile; see `probe_a_k` and `select_eval_maps`).

### Observation, action, reward

With history length $H = 3$ and control step $t = 0, 1, \dots, T-1$:

**Observation** (a dictionary):

* `grid` — the last $H$ pressure fields, shape $H \times 128 \times 128$, normalized as
  $p_{norm} = (p - 90) / (300 - 90)$ with $p$ in bar.
* `vec` — the last $H$ normalized actions $u_{norm} \in [0, 1]$, plus the remaining
  horizon $(T - t)/T$. Shape $H + 1$.

At $t = 0$ the history is the 3 warm-up windows
(so `vec` starts as $[0, 0.2, 0.5, 1]$).
The remaining-horizon feature is included because the objective is a finite-horizon
sum without discounting, so the best action can depend on how many windows are left.

**Action.** PPO outputs $a_t \in [-1, 1]$, mapped linearly to the multiplier:

$$
u_{norm} = \frac{a_t + 1}{2}, \qquad u_t = 0.5 + u_{norm}\,(1.5 - 0.5) \in [0.5,\ 1.5].
$$

**Reward** of window $t$ (units: m³):

$$
r_t = \underbrace{\sum_{i=1}^{m} q_{prod,i}\,\Delta\tau}_{\text{production in window } t}
\;-\; \lambda_P \cdot \text{pviol}_t ,
\qquad m = \frac{\Delta t_{control}}{\Delta\tau} = 30 ,
$$

where $q_{prod,i}$ is the producer rate on day $i$ of the window. The penalty uses
the daily squared violation of the soft limit $P_{lim} = 230$ bar,

$$
y_i = \Big(\max\big(0,\ p_{max,i} - P_{lim}\big)\Big)^2 ,
\qquad p_{max,i} = \max_{\mathbf{x} \in \Omega} p(\mathbf{x}, \text{day } i),
$$

integrated in time with the trapezoid rule. To make the sum over windows equal one
trapezoid rule over the whole control period, each window carries over the last
value $y_0$ of the previous window:

$$
\text{pviol}_t =
\begin{cases}
\Delta\tau \left[ \dfrac{y_1}{2} + \displaystyle\sum_{i=2}^{m-1} y_i + \dfrac{y_m}{2} \right], & t = 0 \ \ (\text{first control window, no carry}),\\[2ex]
\Delta\tau \left[ \dfrac{y_0}{2} + \displaystyle\sum_{i=1}^{m-1} y_i + \dfrac{y_m}{2} \right], & t \ge 1 .
\end{cases}
$$

$\lambda_P = 0.01$ m³/(bar² day).

**Objective.** The episode return is

$$
J = \sum_{t=0}^{T-1} r_t
\;\approx\; \int_{t_w}^{t_w + T_d} q_{prod}\, dt
\;-\; \lambda_P \int_{t_w}^{t_w + T_d} \max\big(p_{max}(t) - P_{lim},\, 0\big)^2 dt ,
$$

with $t_w = 90$ days (end of warm-up) and $T_d = 300$ days. Production is
discretized with a rectangle sum and the penalty with the trapezoid rule.
There is no hard pressure limit: every episode runs all 10 windows, unless the
simulator fails (then the episode ends with reward 0 for that window).

### PPO settings

| Setting | Value |
|---|---|
| Library | Stable-Baselines3 PPO, `MultiInputPolicy` |
| Feature extractor | `grid` → 2 conv layers (32, 64 channels) + pooling → 256 features; concatenated with `vec` |
| Parallel environments | 8 (each with its own MATLAB engine) |
| Total timesteps | 250,000 (= 25,000 episodes) |
| `n_steps` / `batch_size` | 256 per environment / 256 |
| Learning rate | 3 × 10⁻⁴ |
| Discount $\gamma$ | 1.0 |
| Entropy coefficient | linear schedule 0.05 → 0.005 over the PPO updates |
| Other PPO parameters | SB3 defaults (GAE λ = 0.95, clip 0.2, 10 epochs) |
| Seed | 0 |
| Evaluation | every 1000 vector-env steps (= 8000 timesteps), 9 eval maps, deterministic policy; best model is saved |

### Monte Carlo baseline

For each test map $k$, separately:

$$
\mathbf{u}^{(i)} = (u^{(i)}_1, \dots, u^{(i)}_T), \quad u^{(i)}_t \sim \mathcal{U}[0.5, 1.5] \text{ i.i.d.},
\qquad
\hat J_k = \max_{i \le n} J_k(\mathbf{u}^{(i)}) .
$$

Every sample starts from the same post-warm-up state of map $k$. Sampling stops
when the best $J$ has improved by at most 0.1 over the last 5000 samples
(with at least 5000 samples), or at 50,000 samples. This baseline knows the map
and optimizes a fixed schedule for it (an "oracle" open-loop baseline); the best
sample is a lower bound on the optimal $J$ for that map. The mean $J$ over all
samples is also reported as the expected $J$ of a uniformly random policy.

---

## Requirements

* MATLAB **R2025b** (tested on OSC) with MRST (2025 release; uses the `ad-core` and `ad-props` modules)
* MATLAB Engine for Python matching the MATLAB version (for R2025b: `matlabengine` 25.2.x)
* Python 3.10 and the packages in `requirements.txt`

## Setup on OSC

```bash
module load matlab/r2025b
module load miniconda3/24.1.2-py310

# one time: create the environment
conda create -n rl_env python=3.10 -y
conda activate rl_env
pip install -r requirements.txt
pip install "matlabengine==25.2.*"      # must match the loaded MATLAB version

# every session
module load matlab/r2025b
module load miniconda3/24.1.2-py310
conda activate rl_env
export MRST_PATH=/path/to/mrst          # folder that contains MRST's startup.m
```

On another machine, install MATLAB, MRST and the matching MATLAB Engine for
Python, then follow the same steps (without `module load`).

## Data

`k128x128_unique.mat` must contain two variables:

* `kmap` — permeability in mD, size 128 × 128 × `n_unique`
* `n_unique` — number of maps (1000)

Put the file (or a symbolic link to it) in the directory you run from.
Download link: **[to be added]**

## How to run

Run all commands from the repository root (the MATLAB function and the data file
must be in the working directory).

**1. Smoke test (a few minutes)** — checks that MATLAB, MRST and Python work together:

```bash
python qfs_ppo.py --total-timesteps 512 --num-envs 2 \
                  --eval-maps 100 200 --test-maps 980 990
```

`--eval-maps` skips the a_k probe over all 950 train maps, which alone runs
1900 short simulations.

**2. Full training + test (the reported run):**

```bash
python qfs_ppo.py
# or on OSC:  sbatch scripts/train_ppo.slurm
```

**3. Monte Carlo baseline on the 50 test maps:**

```bash
python qfs_mc_baseline.py --map-range 951 1000 --workers 16
# quick check:  python qfs_mc_baseline.py --maps 980 --workers 4 --n-min 64 --n-max 64 --patience 32
# rebuild tables/figures from saved samples:  python qfs_mc_baseline.py --report-only results_<tag>
```

## Outputs

`qfs_ppo.py` writes (`<TAG>` contains the SLURM job id):

* `best_model_dir_<TAG>/` — `best_model.zip`, `final_model.zip`, evaluation log
* `results_<TAG>/` — `test_table.csv`, `test_stats.csv`, `table_test.tex`,
  `report_<TAG>.xlsx`, overlay / envelope / grid figures, `per_map/` episode plots
* `trainingcurve_<TAG>.png`, `reward_breakdown_<TAG>.png`,
  `reward_per_window_<TAG>.png`, `test_overlay_<TAG>.png`, `a_k_train_<TAG>.csv`

`qfs_mc_baseline.py` writes `results_<tag>/` with the same table and figure format,
plus the raw samples `samples_k<map>.npz`.

## Notes

* **Incompressible flow.** Because $c_t = 0$ and the producer pressure is fixed,
  the pressure field responds instantly to the current rate and is linear in it:
  $p_{max} = P_{bhp} + a_k u_t$ for a map-dependent constant $a_k$, and produced
  volume equals injected volume. The pressure history therefore carries
  information about the map (through $a_k$) rather than a dynamic state.
  `probe_a_k` computes $a_k$ and checks this linearity.
* **Solver failure.** If MRST fails in a window, the episode ends
  (`terminated=True`) with reward 0 for that window.
* **Original file names** on OSC: `qfs_sp_pomdp3_basic_warmup_5.py` → `qfs_ppo.py`,
  `qfs_sp_baseline_mc_2maps.py` → `qfs_mc_baseline.py`.
