"""非参数最大似然核心：EM、凸优化与诊断共享同一个似然对象。

目标:  max_p  ll(p) = sum_i log(s_i),  s = A p,  p >= 0, sum p = 1
KKT:   记 d_j = sum_i A_ij / s_i（对 p_j 的偏导）。单纯形上的最优解满足
       支撑上 d_j = n、支撑外 d_j <= n；kkt_residual = max(d) - n >= 0，
       收敛时趋于 0。零质量单元中 d_j 明显小于 n 的是被钉在边界；
       d_j 仍约等于 n 的说明存在把质量放到该单元的其他最优解（似然不唯一）。

不把任何申报当作均匀分布，也不预置正态/对数正态等连续参数族；
p 是支持单元上的离散质量，单元内位置不做任何假设。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize as sp_opt


class IntervalCensoredLikelihood:
    """区间删失非参数似然。EM、SLSQP 拟合与界查询共用本类的目标与梯度。"""

    def __init__(self, A: np.ndarray):
        A = np.asarray(A, dtype=float)
        if A.ndim != 2 or A.shape[0] == 0 or A.shape[1] == 0:
            raise ValueError("A 必须是非空二维矩阵")
        if (A.sum(axis=1) <= 0).any():
            raise ValueError("存在不包含任何支持单元的申报")
        self.A = A
        self.n, self.m = A.shape

    def scores(self, p: np.ndarray) -> np.ndarray:
        """每条申报的域概率 s_i = sum_j A_ij p_j。"""
        return self.A @ np.asarray(p, dtype=float)

    def loglik(self, p: np.ndarray) -> float:
        s = self.scores(p)
        if (s <= 0).any():
            return -np.inf
        return float(np.log(s).sum())

    def gradient(self, p: np.ndarray) -> np.ndarray:
        s = self.scores(p)
        if (s <= 0).any():
            raise ValueError("梯度只在所有 s_i > 0 处有定义")
        return self.A.T @ (1.0 / s)

    def kkt_residual(self, p: np.ndarray) -> float:
        return float(self.gradient(p).max() - self.n)


@dataclass
class FitResult:
    p: np.ndarray
    loglik: float
    iterations: int
    converged: bool
    kkt_residual: float
    method: str
    gradient: np.ndarray


def _uniform(m: int) -> np.ndarray:
    return np.full(m, 1.0 / m)


def fit_em(model: IntervalCensoredLikelihood, p0=None, kkt_tol=1e-8, max_iter=2000) -> FitResult:
    """Turnbull 自洽/EM 迭代：p_j <- p_j * d_j / n。

    收敛判据只用 KKT 残差 max(d)-n <= kkt_tol；达到 max_iter 仍未满足
    则 converged=False（调用方应统计这类收敛失败，而不是当成成功）。
    """
    p = _uniform(model.m) if p0 is None else np.asarray(p0, dtype=float).copy()
    p = np.clip(p, 0.0, None)
    total = p.sum()
    if total <= 0:
        raise ValueError("初始质量必须为正")
    p /= total
    d = model.gradient(p)
    residual = float(d.max() - model.n)
    iterations = 0
    while residual > kkt_tol and iterations < max_iter:
        p = p * d / model.n
        p /= p.sum()
        d = model.gradient(p)
        residual = float(d.max() - model.n)
        iterations += 1
    return FitResult(
        p=p,
        loglik=model.loglik(p),
        iterations=iterations,
        converged=residual <= kkt_tol,
        kkt_residual=residual,
        method="em",
        gradient=d,
    )


def fit_slsqp(model: IntervalCensoredLikelihood, p0=None, kkt_tol=1e-8, max_iter=500) -> FitResult:
    """凸优化交叉校验：SLSQP 直接最大化凹目标 ll(p)（单纯形约束）。"""
    p_start = _uniform(model.m) if p0 is None else np.asarray(p0, dtype=float).copy()

    def neg_obj(p):
        ll = model.loglik(p)
        return -ll if np.isfinite(ll) else 1e100

    def neg_jac(p):
        s = np.maximum(model.scores(p), 1e-300)
        return -(model.A.T @ (1.0 / s))

    res = sp_opt.minimize(
        neg_obj,
        p_start,
        jac=neg_jac,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * model.m,
        constraints=[{
            "type": "eq",
            "fun": lambda p: float(p.sum() - 1.0),
            "jac": lambda p: np.ones(model.m),
        }],
        options={"maxiter": max_iter, "ftol": 1e-15},
    )
    p = np.clip(res.x, 0.0, None)
    p /= p.sum()
    d = model.gradient(p)
    residual = float(d.max() - model.n)
    return FitResult(
        p=p,
        loglik=model.loglik(p),
        iterations=int(res.nit),
        converged=bool(res.success) and residual <= kkt_tol,
        kkt_residual=residual,
        method="slsqp",
        gradient=d,
    )


@dataclass
class MultiFit:
    best: FitResult
    fits: list
    loglik_ref: float
    p_ref: np.ndarray
    nonunique_spread: float  # 似然相同（容差内）的拟合之间 max ||Δp||_inf
    ll_tol: float


def fit_multi(model: IntervalCensoredLikelihood, kkt_tol=1e-8, max_iter=2000,
              ll_tol=1e-8, restarts=3, seed=0) -> MultiFit:
    """多起点 EM + SLSQP 抛光。

    多个最优质量分配可能给出相同似然：用不同起点拟合，若似然在容差内
    相同而质量分配明显不同，则 nonunique_spread 显著为正——单个 EM
    停止点不代表唯一分布。
    """
    rng = np.random.default_rng(seed)
    starts = [_uniform(model.m)]
    starts += [rng.dirichlet(np.ones(model.m)) for _ in range(max(0, restarts))]
    fits = [fit_em(model, s, kkt_tol=kkt_tol, max_iter=max_iter) for s in starts]
    best_em = max(fits, key=lambda f: f.loglik)
    fits.append(fit_slsqp(model, best_em.p, kkt_tol=kkt_tol))
    best = max(fits, key=lambda f: f.loglik)
    ll_ref, p_ref = best.loglik, best.p
    near = [f for f in fits if f.loglik >= ll_ref - ll_tol]
    spread = 0.0
    for a in range(len(near)):
        for b in range(a + 1, len(near)):
            spread = max(spread, float(np.abs(near[a].p - near[b].p).max()))
    return MultiFit(best=best, fits=fits, loglik_ref=ll_ref, p_ref=p_ref,
                    nonunique_spread=spread, ll_tol=ll_tol)


def zero_mass_boundary(model: IntervalCensoredLikelihood, p, gradient=None,
                       zero_tol=1e-10, active_margin=1e-6) -> list:
    """零质量边界诊断。

    p_j≈0 的单元里：梯度明显小于 n → pinned_at_zero（进入支撑会严格降低
    似然）；梯度仍≈n → zero_but_active（其他最优解可能把质量放在这里，
    是似然不唯一的信号）。
    """
    d = model.gradient(p) if gradient is None else np.asarray(gradient, dtype=float)
    p = np.asarray(p, dtype=float)
    out = []
    for j in range(model.m):
        if p[j] <= zero_tol:
            margin = float(d[j] - model.n)
            out.append({
                "index": j,
                "gradient": float(d[j]),
                "margin": margin,
                "status": "zero_but_active" if abs(margin) <= active_margin else "pinned_at_zero",
            })
    return out
