# Geometric Brownian Motion and Ornstein-Uhlenbeck SDE Lab

An exploratory project for simulating, estimating, and validating two foundational stochastic differential equation (SDE) models using Euler–Maruyama discretization:

- Geometric Brownian Motion (GBM)
- Ornstein-Uhlenbeck (OU)

Project layout:

```
├── models.py          # theoretical SDE/BS model primitives + simulators
├── simulate.py        # experiment runner
├── estimation.py      # parameter estimation (optimisation here)
├── analysis.py        # statistics + validation
├── first_passage.py   # hitting time analysis
├── options/           # practical option pricing + Greeks + MC apps
│   ├── black_scholes.py
│   ├── demo.py
│   ├── greeks.py
│   ├── monte_carlo.py
│   └── portfolio_risk.py
├── physics_simulation/ # practical physics simulations tied to SDE models
│   └── langevin.py
├── plots.py           # all visualisations
└── README.md
```

---

## 1. Introduction

### What SDEs are

Stochastic differential equations model systems with both deterministic and random components. A generic Itô SDE is:

$$
dX_t = a(X_t, t)\,dt + b(X_t, t)\,dW_t
$$

where:

- $a(X_t,t)$ is the drift (systematic trend),
- $b(X_t,t)$ is the diffusion (noise intensity),
- $W_t$ is standard Brownian motion.

SDEs are central in quantitative finance, control, physics, biology, and signal processing because they capture uncertainty directly at the model level.

### Why GBM / OU

#### GBM

GBM is a canonical positive-valued process for asset prices:

$$
dS_t = \mu S_t\,dt + \sigma S_t\,dW_t
$$

It enforces positivity and yields log-normal marginals in continuous time. It is used in Black–Scholes-style modeling and as a baseline for multiplicative-noise systems.

#### OU

OU is a mean-reverting Gaussian process:

$$
dX_t = \theta(\mu - X_t)\,dt + \sigma\,dW_t
$$

It is widely used for short rates, spreads, residuals, and physical relaxation dynamics where pull-back to long-run level is essential.

These two models provide a useful contrast:

- GBM: non-stationary, multiplicative noise, positive state.
- OU: stationary (under mild conditions), additive noise, mean reversion.

---

## 2. Methods

### Euler–Maruyama discretization

For time step $\Delta t$, Euler–Maruyama (EM) approximates:

$$
X_{n+1} = X_n + a(X_n,t_n)\Delta t + b(X_n,t_n)\sqrt{\Delta t}\,Z_n,
\quad Z_n \sim \mathcal{N}(0,1)
$$

Applied models in this repository:

#### GBM (EM)

$$
S_{n+1} = S_n\left(1 + \mu\Delta t + \sigma\sqrt{\Delta t}Z_n\right)
$$

Because finite-step EM can produce rare negative updates, implementation clamps at a small positive floor to maintain valid log transforms for estimation.

#### OU (EM)

$$
X_{n+1} = X_n + \theta(\mu - X_n)\Delta t + \sigma\sqrt{\Delta t}Z_n
$$

### Estimation approach

`estimation.py` contains optimization-based maximum likelihood estimators for discretized transition models.

#### GBM transform + MLE objective

Using log returns:

$$
r_n = \log\left(\frac{S_{n+1}}{S_n}\right)
\sim \mathcal{N}\left((\mu - \tfrac12\sigma^2)\Delta t,\, \sigma^2\Delta t\right)
$$

To enforce positivity in optimization:

- optimize over transformed variables $(\mu, \log\sigma)$,
- map back with $\sigma = e^{\log\sigma}$.

Objective is the Gaussian negative log-likelihood:

$$
\mathcal{L}(\mu,\sigma) =
\frac{N}{2}\log(2\pi\sigma^2\Delta t)
+ \frac{1}{2\sigma^2\Delta t}\sum_{n=1}^N\left(r_n-(\mu-\tfrac12\sigma^2)\Delta t\right)^2
$$

#### OU transformed MLE

For EM transition approximation:

$$
X_{n+1}\mid X_n \sim \mathcal{N}\left(X_n + \theta(\mu-X_n)\Delta t,\,\sigma^2\Delta t\right)
$$

Optimization uses $(\log\theta,\mu,\log\sigma)$ to preserve positivity constraints on $(\theta,\sigma)$.

#### Optimizer

The project uses a lightweight coordinate-search optimizer with adaptive step shrinkage to avoid external dependencies and keep logic transparent for exploratory work.

---

## 3. Results

The script `simulate.py` runs a full experiment:

1. Simulate many paths for GBM and OU.
2. Compute distributional diagnostics.
3. Estimate parameters on a representative path using MLE.
4. Validate by absolute/relative error against known true parameters.
5. Compute first-passage (hitting time) summaries.
6. Save visualizations in `results/`.

### Simulation

Generated artifacts:

- `results/gbm_paths.png`
- `results/gbm_exact_paths.png`
- `results/gbm_terminal_hist.png`
- `results/gbm_hitting_times.png`
- `results/gbm_mean_vs_theory.png`
- `results/gbm_moment_validation.csv`
- `results/ou_paths.png`
- `results/ou_terminal_hist.png`
- `results/ou_hitting_times.png`
- `results/ou_moments_vs_theory.png`
- `results/ou_moment_validation.csv`

### Validation

`analysis.py` provides:

- moment summaries (mean, std, skewness, excess kurtosis),
- empirical-vs-theoretical moment checks:
	- GBM empirical cross-sectional mean vs $S_0 e^{\mu t}$,
	- OU empirical mean/variance vs exact OU moments,
- estimation error metrics:
  - absolute error: $|\hat\theta - \theta|$,
  - relative error: $|\hat\theta - \theta| / |\theta|$.

`simulate.py --mode single` now prints explicit theory-vs-simulation diagnostics, including:

- GBM max relative mean error and relative RMSE,
- OU mean scaled max error,
- OU variance max relative error,
- OU terminal variance gap to stationary target $\sigma^2/(2\theta)$.

### Estimation

For GBM and OU, the runner reports:

- true vs estimated parameters,
- per-parameter absolute/relative errors,
- final negative log-likelihood objective value.

This setup allows quick empirical checks of identifiability and finite-sample behavior under controlled synthetic data.

### Monte Carlo study mode

`simulate.py` now supports a repeated-estimation study mode to quantify estimator behavior across many synthetic paths.

Outputs added in `results/`:

- `monte_carlo_gbm_raw.csv` (replication-level GBM estimates)
- `monte_carlo_ou_raw.csv` (replication-level OU estimates)
- `monte_carlo_summary.csv` (mean, bias, std, variance, RMSE per parameter)
- `monte_carlo_estimate_histograms.png` (distribution of estimated parameters)

---

## 4. Extensions

### Hitting times / first passage

`first_passage.py` computes:

- first index/time a trajectory crosses a barrier,
- hit probability,
- mean/median hitting times.

This is relevant for barrier options (GBM), threshold control, and excursion analysis in mean-reverting systems (OU).

### OU-focused diagnostics and estimation (implemented)

Implemented OU refinement stack includes:

- exact-transition OU MLE alongside EM-MLE,
- asymptotic and bootstrap confidence intervals,
- residual QQ/ACF/Ljung-Box diagnostics,
- mean and variance validation against exact OU moment formulas.

### Implemented advanced package (now available)

The project now includes all major refinement tracks discussed:

- exact GBM transition simulation (`exact_step_gbm`),
- exact-transition OU MLE (`estimate_ou_exact_mle`) alongside EM-MLE,
- asymptotic CIs (observed Fisher via numerical Hessian),
- parametric bootstrap CIs,
- empirical CI coverage study,
- grid study over $(\Delta t, T, N)$ with relative-RMSE heatmaps,
- residual diagnostics (QQ + ACF + Ljung-Box Q statistic),
- model comparison metrics (AIC/BIC + OU LR test for $\theta=0$),
- regime extensions: jump-diffusion GBM, CIR, and time-varying GBM/OU simulation.

Integrated options/risk stack (now split into practical modules under `options/` and callable from `simulate.py`):

- Black-Scholes analytics (call/put/digital + Greeks),
- GBM Monte Carlo pricers (European, arithmetic Asian, barrier),
- OU mean-reverting log-price Monte Carlo pricer,
- correlated multi-asset GBM simulation + portfolio VaR/CVaR.

Standalone options dashboard:

- European call/put and digital pricing,
- Asian option comparison,
- barrier option sensitivity,
- pathwise vs likelihood-ratio Greeks,
- saved dashboard figures in `results/`.

Integrated physics simulation stack (under `physics_simulation/` and callable from `simulate.py`):

- 1D underdamped Langevin particle,
- velocity modeled as an OU process,
- position obtained as the time-integral of velocity,
- empirical velocity moments vs OU theoretical moments.

---

## 5. Conclusion

### Numerical accuracy

Euler–Maruyama is easy to implement and effective for exploratory simulation and estimation pipelines. Accuracy improves with smaller $\Delta t$ and sufficient sample size, but discretization error remains model- and horizon-dependent.

### Limitations

- EM bias for coarse time grids.
- GBM EM update can violate positivity without safeguards.
- Single-path estimation can be high variance.
- Coordinate-search optimization is robust but not as sample-efficient as gradient/Hessian methods.

---

## Suggested refinements (next iterations)

1. Add automated regression tests that assert acceptable GBM/OU moment-validation tolerances from `gbm_moment_validation.csv` and `ou_moment_validation.csv`.
2. Add panel-based moment validation summaries (across different $(\Delta t, T, N)$ settings) directly into `grid_study_summary.csv`.
3. Add robust optimizer backends (e.g. SciPy L-BFGS-B) with a configurable fallback to coordinate search.
4. Add optional exact-transition OU simulation mode in `models.py` for side-by-side simulation bias studies (not only exact likelihood estimation).
5. Add richer data-generating models (e.g. stochastic volatility and jump-intensity dynamics) for stress tests beyond constant-parameter GBM/OU.
6. Add a lightweight report generator that collates key metrics and figures into a single markdown or PDF summary.

---

## Quickstart

Install dependencies:

```bash
pip install numpy matplotlib
```

Run baseline single experiment:

```bash
python simulate.py --mode single
```

Run Monte Carlo estimation study only:

```bash
python simulate.py --mode study --mc-reps 500
```

Run single + Monte Carlo (default behavior):

```bash
python simulate.py --mode all --mc-reps 300
```

Run full advanced suite (all features):

```bash
python simulate.py --mode full --mc-reps 300 --coverage-reps 60 --bootstrap-reps 80 --grid-reps 40
```

Run integrated options/risk module with the main pipeline:

```bash
python simulate.py --mode full --with-options
```

Run the standalone options dashboard without the GBM/OU baseline:

```bash
python simulate.py --mode options
```

Run integrated physics module with the main pipeline:

```bash
python simulate.py --mode full --with-physics
```

Run the standalone physics dashboard without the GBM/OU baseline:

```bash
python simulate.py --mode physics
```

Quick one-command run (legacy/default):

```bash
python simulate.py
```

The script prints diagnostics and saves all artifacts to `results/`, including CI coverage and grid-study summaries in full mode.