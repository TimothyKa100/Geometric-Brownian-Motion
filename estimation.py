from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from models import GBMParams, OUParams, exact_step_gbm, euler_maruyama_ou


Array = np.ndarray


@dataclass(frozen=True)
class GBMMLE:
    mu: float
    sigma: float
    nll: float


@dataclass(frozen=True)
class OUMLE:
    theta: float
    mu: float
    sigma: float
    nll: float


@dataclass(frozen=True)
class ParameterCI:
    lower: float
    upper: float


@dataclass(frozen=True)
class InformationCriteria:
    aic: float
    bic: float


def _minimize_grid(
    objective: Callable[[Array], float],
    start: Array,
    step: Array,
    n_iter: int = 120,
) -> tuple[Array, float]:
    """Simple coordinate-search optimizer with step shrinkage."""
    x = start.astype(float).copy()
    best = objective(x)

    for _ in range(n_iter):
        improved = False
        for i in range(len(x)):
            candidates = [x.copy(), x.copy()]
            candidates[0][i] += step[i]
            candidates[1][i] -= step[i]
            for cand in candidates:
                score = objective(cand)
                if score < best:
                    x = cand
                    best = score
                    improved = True

        if not improved:
            step *= 0.6
            if np.all(step < 1e-8):
                break

    return x, best


def gbm_negative_log_likelihood_transformed(params_t: Array, prices: Array, dt: float) -> float:
    """NLL for GBM under transformed parameters [mu, log_sigma].

    For log returns r_t = log(S_t/S_{t-1}):
        r_t ~ N((mu - 0.5 sigma^2) dt, sigma^2 dt)
    """
    mu = params_t[0]
    sigma = np.exp(params_t[1])

    if sigma <= 0 or dt <= 0:
        return np.inf

    if np.any(prices <= 0):
        return np.inf

    returns = np.diff(np.log(prices))
    mean = (mu - 0.5 * sigma**2) * dt
    var = sigma**2 * dt

    n = len(returns)
    if n == 0:
        return np.inf

    residual = returns - mean
    nll = 0.5 * n * np.log(2.0 * np.pi * var) + 0.5 * np.sum((residual**2) / var)
    return float(nll)


def _gbm_nll_from_returns(mu: float, sigma: float, returns: Array, dt: float) -> float:
    if sigma <= 0 or dt <= 0:
        return np.inf
    mean = (mu - 0.5 * sigma**2) * dt
    var = sigma**2 * dt
    n = len(returns)
    if n == 0:
        return np.inf
    residual = returns - mean
    return float(0.5 * n * np.log(2.0 * np.pi * var) + 0.5 * np.sum((residual**2) / var))


def estimate_gbm_mle(prices: Array, dt: float) -> GBMMLE:
    """Estimate GBM parameters via MLE with positivity transform for sigma."""
    if np.any(prices <= 0):
        raise ValueError("GBM estimation requires strictly positive prices.")

    returns = np.diff(np.log(prices))
    if len(returns) < 2:
        raise ValueError("At least 3 price points are needed for stable MLE.")

    sample_mean = np.mean(returns)
    sample_std = np.std(returns, ddof=1)
    sigma0 = max(sample_std / np.sqrt(dt), 1e-6)
    mu0 = sample_mean / dt + 0.5 * sigma0**2

    start = np.array([mu0, np.log(sigma0)], dtype=float)
    step = np.array([0.1 * max(abs(mu0), 1.0), 0.2], dtype=float)

    obj = lambda x: gbm_negative_log_likelihood_transformed(x, prices, dt)
    best_t, best_nll = _minimize_grid(obj, start=start, step=step)

    return GBMMLE(mu=float(best_t[0]), sigma=float(np.exp(best_t[1])), nll=float(best_nll))


def estimate_gbm_mle_panel(prices_paths: Array, dt: float) -> GBMMLE:
    """Estimate GBM parameters from multiple paths by pooling log-returns."""
    if prices_paths.ndim != 2:
        raise ValueError("prices_paths must be 2D with shape (n_paths, n_points).")
    if np.any(prices_paths <= 0):
        raise ValueError("GBM panel estimation requires strictly positive prices.")

    returns = np.diff(np.log(prices_paths), axis=1).ravel()
    if len(returns) < 2:
        raise ValueError("Not enough observations for panel GBM MLE.")

    sample_mean = np.mean(returns)
    sample_std = np.std(returns, ddof=1)
    sigma0 = max(sample_std / np.sqrt(dt), 1e-6)
    mu0 = sample_mean / dt + 0.5 * sigma0**2

    start = np.array([mu0, np.log(sigma0)], dtype=float)
    step = np.array([0.1 * max(abs(mu0), 1.0), 0.2], dtype=float)

    def obj(params_t: Array) -> float:
        mu = params_t[0]
        sigma = np.exp(params_t[1])
        return _gbm_nll_from_returns(mu=mu, sigma=sigma, returns=returns, dt=dt)

    best_t, best_nll = _minimize_grid(obj, start=start, step=step)
    return GBMMLE(mu=float(best_t[0]), sigma=float(np.exp(best_t[1])), nll=float(best_nll))


def ou_negative_log_likelihood_transformed(params_t: Array, series: Array, dt: float) -> float:
    """Gaussian transition density approximation for OU under EM discretization.

    X_{t+1} | X_t ~ N(X_t + theta(mu - X_t)dt, sigma^2 dt)
    with transformed parameters [log_theta, mu, log_sigma].
    """
    theta = np.exp(params_t[0])
    mu = params_t[1]
    sigma = np.exp(params_t[2])

    if theta < 0 or sigma <= 0 or dt <= 0:
        return np.inf

    x_prev = series[:-1]
    x_next = series[1:]
    if len(x_prev) == 0:
        return np.inf

    mean = x_prev + theta * (mu - x_prev) * dt
    var = sigma**2 * dt

    residual = x_next - mean
    n = len(residual)
    nll = 0.5 * n * np.log(2.0 * np.pi * var) + 0.5 * np.sum((residual**2) / var)
    return float(nll)


def _ou_em_nll(theta: float, mu: float, sigma: float, x_prev: Array, x_next: Array, dt: float) -> float:
    if theta < 0 or sigma <= 0 or dt <= 0:
        return np.inf
    mean = x_prev + theta * (mu - x_prev) * dt
    var = sigma**2 * dt
    residual = x_next - mean
    n = len(residual)
    return float(0.5 * n * np.log(2.0 * np.pi * var) + 0.5 * np.sum((residual**2) / var))


def estimate_ou_mle(series: Array, dt: float) -> OUMLE:
    """Estimate OU parameters with transformed MLE objective."""
    if len(series) < 3:
        raise ValueError("At least 3 observations are needed for OU MLE.")

    x_prev = series[:-1]
    x_next = series[1:]
    dx = x_next - x_prev

    beta = np.polyfit(x_prev, dx / dt, 1)
    theta0 = max(-beta[0], 1e-4)
    mu0 = beta[1] / theta0 if theta0 > 0 else np.mean(series)
    sigma0 = max(np.std(dx, ddof=1) / np.sqrt(dt), 1e-6)

    start = np.array([np.log(theta0), mu0, np.log(sigma0)], dtype=float)
    step = np.array([0.3, 0.2 * max(np.std(series), 1.0), 0.3], dtype=float)

    obj = lambda x: ou_negative_log_likelihood_transformed(x, series, dt)
    best_t, best_nll = _minimize_grid(obj, start=start, step=step)

    return OUMLE(
        theta=float(np.exp(best_t[0])),
        mu=float(best_t[1]),
        sigma=float(np.exp(best_t[2])),
        nll=float(best_nll),
    )


def estimate_ou_mle_panel(paths: Array, dt: float) -> OUMLE:
    """Estimate OU parameters from multiple paths by pooling transitions."""
    if paths.ndim != 2:
        raise ValueError("paths must be 2D with shape (n_paths, n_points).")
    if paths.shape[1] < 3:
        raise ValueError("Not enough observations for panel OU MLE.")

    x_prev = paths[:, :-1].ravel()
    x_next = paths[:, 1:].ravel()
    dx = x_next - x_prev

    beta = np.polyfit(x_prev, dx / dt, 1)
    theta0 = max(-beta[0], 1e-4)
    mu0 = beta[1] / theta0 if theta0 > 0 else np.mean(paths)
    sigma0 = max(np.std(dx, ddof=1) / np.sqrt(dt), 1e-6)

    start = np.array([np.log(theta0), mu0, np.log(sigma0)], dtype=float)
    step = np.array([0.3, 0.2 * max(np.std(paths), 1.0), 0.3], dtype=float)

    def obj(params_t: Array) -> float:
        theta = np.exp(params_t[0])
        mu = params_t[1]
        sigma = np.exp(params_t[2])
        return _ou_em_nll(theta=theta, mu=mu, sigma=sigma, x_prev=x_prev, x_next=x_next, dt=dt)

    best_t, best_nll = _minimize_grid(obj, start=start, step=step)

    return OUMLE(
        theta=float(np.exp(best_t[0])),
        mu=float(best_t[1]),
        sigma=float(np.exp(best_t[2])),
        nll=float(best_nll),
    )


def estimate_ou_exact_mle(series: Array, dt: float) -> OUMLE:
    """Exact-transition MLE for OU via AR(1) mapping with intercept."""
    if len(series) < 3:
        raise ValueError("At least 3 observations are needed for exact OU MLE.")
    if dt <= 0:
        raise ValueError("dt must be positive.")

    x_prev = series[:-1]
    x_next = series[1:]
    x_mean = np.mean(x_prev)
    y_mean = np.mean(x_next)
    sxx = np.sum((x_prev - x_mean) ** 2)
    if sxx <= 0:
        raise ValueError("Degenerate series for OU exact MLE.")

    b_hat = np.sum((x_prev - x_mean) * (x_next - y_mean)) / sxx
    b_hat = float(np.clip(b_hat, 1e-8, 0.999999))
    a_hat = float(y_mean - b_hat * x_mean)

    theta_hat = -np.log(b_hat) / dt
    mu_hat = a_hat / max(1.0 - b_hat, 1e-8)

    mean_hat = a_hat + b_hat * x_prev
    residual = x_next - mean_hat
    s2 = float(np.mean(residual**2))
    sigma_hat = np.sqrt(max(2.0 * theta_hat * s2 / max(1.0 - b_hat**2, 1e-12), 1e-12))

    var = (sigma_hat**2 / (2.0 * theta_hat)) * (1.0 - np.exp(-2.0 * theta_hat * dt))
    nll = float(0.5 * len(residual) * np.log(2.0 * np.pi * var) + 0.5 * np.sum((residual**2) / var))
    return OUMLE(theta=float(theta_hat), mu=float(mu_hat), sigma=float(sigma_hat), nll=nll)


def information_criteria(nll: float, n_obs: int, n_params: int) -> InformationCriteria:
    """Compute AIC and BIC from negative log-likelihood."""
    aic = 2.0 * n_params + 2.0 * nll
    bic = np.log(max(n_obs, 1)) * n_params + 2.0 * nll
    return InformationCriteria(aic=float(aic), bic=float(bic))


def _numerical_hessian(objective: Callable[[Array], float], x0: Array, eps: float = 1e-5) -> Array:
    d = len(x0)
    hessian = np.zeros((d, d), dtype=float)
    fx = objective(x0)

    for i in range(d):
        x_ip = x0.copy()
        x_im = x0.copy()
        x_ip[i] += eps
        x_im[i] -= eps
        f_ip = objective(x_ip)
        f_im = objective(x_im)
        hessian[i, i] = (f_ip - 2.0 * fx + f_im) / (eps**2)

        for j in range(i + 1, d):
            x_pp = x0.copy()
            x_pm = x0.copy()
            x_mp = x0.copy()
            x_mm = x0.copy()
            x_pp[i] += eps
            x_pp[j] += eps
            x_pm[i] += eps
            x_pm[j] -= eps
            x_mp[i] -= eps
            x_mp[j] += eps
            x_mm[i] -= eps
            x_mm[j] -= eps

            f_pp = objective(x_pp)
            f_pm = objective(x_pm)
            f_mp = objective(x_mp)
            f_mm = objective(x_mm)
            value = (f_pp - f_pm - f_mp + f_mm) / (4.0 * eps**2)
            hessian[i, j] = value
            hessian[j, i] = value

    return hessian


def _ci_from_covariance(est: Array, covariance: Array, transform: Callable[[Array], Array], alpha: float = 0.05) -> list[ParameterCI]:
    z = 1.959963984540054
    transformed = transform(est)
    ci_list: list[ParameterCI] = []
    for i in range(len(transformed)):
        std = np.sqrt(max(covariance[i, i], 1e-15))
        ci_list.append(ParameterCI(lower=float(transformed[i] - z * std), upper=float(transformed[i] + z * std)))
    return ci_list


def gbm_asymptotic_ci(prices: Array, dt: float, alpha: float = 0.05) -> dict[str, ParameterCI]:
    """Approximate asymptotic CI via observed Fisher information (numerical Hessian)."""
    est = estimate_gbm_mle(prices, dt)
    theta_hat = np.array([est.mu, np.log(est.sigma)], dtype=float)
    objective = lambda x: gbm_negative_log_likelihood_transformed(x, prices, dt)
    hessian = _numerical_hessian(objective, theta_hat)
    covariance_t = np.linalg.pinv(hessian)

    j = np.array([[1.0, 0.0], [0.0, est.sigma]])
    covariance = j @ covariance_t @ j.T
    cis = _ci_from_covariance(np.array([est.mu, est.sigma]), covariance, transform=lambda x: x, alpha=alpha)
    return {"mu": cis[0], "sigma": cis[1]}


def ou_asymptotic_ci(series: Array, dt: float, alpha: float = 0.05) -> dict[str, ParameterCI]:
    est = estimate_ou_mle(series, dt)
    theta_hat = np.array([np.log(est.theta), est.mu, np.log(est.sigma)], dtype=float)
    objective = lambda x: ou_negative_log_likelihood_transformed(x, series, dt)
    hessian = _numerical_hessian(objective, theta_hat)
    covariance_t = np.linalg.pinv(hessian)

    j = np.array(
        [
            [est.theta, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, est.sigma],
        ]
    )
    covariance = j @ covariance_t @ j.T
    cis = _ci_from_covariance(np.array([est.theta, est.mu, est.sigma]), covariance, transform=lambda x: x, alpha=alpha)
    return {"theta": cis[0], "mu": cis[1], "sigma": cis[2]}


def gbm_bootstrap_ci(prices: Array, dt: float, n_bootstrap: int = 200, alpha: float = 0.05, seed: int = 123) -> dict[str, ParameterCI]:
    """Parametric bootstrap CI for GBM parameters."""
    est = estimate_gbm_mle(prices, dt)
    n_steps = len(prices) - 1
    horizon = n_steps * dt
    rng = np.random.default_rng(seed)

    mu_samples = []
    sigma_samples = []
    for _ in range(n_bootstrap):
        boot_seed = int(rng.integers(1, 10_000_000))
        _, paths = exact_step_gbm(
            s0=float(prices[0]),
            params=GBMParams(mu=est.mu, sigma=est.sigma),
            horizon=horizon,
            n_steps=n_steps,
            n_paths=1,
            seed=boot_seed,
        )
        boot_est = estimate_gbm_mle(paths[0], dt=dt)
        mu_samples.append(boot_est.mu)
        sigma_samples.append(boot_est.sigma)

    lower_q = 100.0 * (alpha / 2.0)
    upper_q = 100.0 * (1.0 - alpha / 2.0)
    return {
        "mu": ParameterCI(lower=float(np.percentile(mu_samples, lower_q)), upper=float(np.percentile(mu_samples, upper_q))),
        "sigma": ParameterCI(lower=float(np.percentile(sigma_samples, lower_q)), upper=float(np.percentile(sigma_samples, upper_q))),
    }


def ou_bootstrap_ci(series: Array, dt: float, n_bootstrap: int = 200, alpha: float = 0.05, seed: int = 123) -> dict[str, ParameterCI]:
    """Parametric bootstrap CI for OU parameters (EM transition simulator)."""
    est = estimate_ou_mle(series, dt)
    n_steps = len(series) - 1
    horizon = n_steps * dt
    rng = np.random.default_rng(seed)

    theta_samples = []
    mu_samples = []
    sigma_samples = []

    for _ in range(n_bootstrap):
        boot_seed = int(rng.integers(1, 10_000_000))
        _, paths = euler_maruyama_ou(
            x0=float(series[0]),
            params=OUParams(theta=est.theta, mu=est.mu, sigma=est.sigma),
            horizon=horizon,
            n_steps=n_steps,
            n_paths=1,
            seed=boot_seed,
        )
        boot_est = estimate_ou_mle(paths[0], dt=dt)
        theta_samples.append(boot_est.theta)
        mu_samples.append(boot_est.mu)
        sigma_samples.append(boot_est.sigma)

    lower_q = 100.0 * (alpha / 2.0)
    upper_q = 100.0 * (1.0 - alpha / 2.0)
    return {
        "theta": ParameterCI(lower=float(np.percentile(theta_samples, lower_q)), upper=float(np.percentile(theta_samples, upper_q))),
        "mu": ParameterCI(lower=float(np.percentile(mu_samples, lower_q)), upper=float(np.percentile(mu_samples, upper_q))),
        "sigma": ParameterCI(lower=float(np.percentile(sigma_samples, lower_q)), upper=float(np.percentile(sigma_samples, upper_q))),
    }
