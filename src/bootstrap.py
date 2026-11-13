"""机构层 bootstrap。

每次重抽样都重新构造支持单元与 A 矩阵、重新 EM 拟合，并统计收敛失败
次数（失败复制不进入分位数汇总）。每个复制内的 [L,U] 是部分识别范围，
不是抽样区间；这里只用分位数描述识别集合的抽样波动，不把识别范围
折叠成点估计，也不把识别不确定度与抽样不确定度混为一谈。
"""

from __future__ import annotations

import numpy as np

from .bounds import certified_bound, classify_cells, face_lp_bound
from .npmle import IntervalCensoredLikelihood, fit_em
from .support import build_support

QUANTILES = (0.05, 0.5, 0.95)


def bootstrap(declarations, prices, repeats, seed, kkt_tol, max_iter, ll_tol):
    rng = np.random.default_rng(seed)
    n = len(declarations)
    prices = [float(t) for t in prices]
    lowers = {t: [] for t in prices}
    uppers = {t: [] for t in prices}
    failures = 0
    support_sizes: list[int] = []
    for _ in range(int(repeats)):
        idx = rng.integers(0, n, size=n)  # 机构层有放回重抽样
        sample = [declarations[i] for i in idx]
        support = build_support(sample)  # 每次重建支持单元与 A
        support_sizes.append(len(support.cells))
        model = IntervalCensoredLikelihood(support.A)
        fit = fit_em(model, kkt_tol=kkt_tol, max_iter=max_iter)
        if not fit.converged:
            failures += 1  # 计收敛失败，不进入汇总
            continue
        level = fit.loglik - ll_tol
        for k, t in enumerate(prices):
            above, straddle = classify_cells(support.cells, t)
            c_lo = above.astype(float)
            c_hi = (above | straddle).astype(float)
            if ll_tol <= 0.0:
                s_star = model.scores(fit.p)
                lo = face_lp_bound(model, c_lo, s_star, maximize=False)
                hi = face_lp_bound(model, c_hi, s_star, maximize=True)
            else:
                lo = certified_bound(model, c_lo, level, fit.p, maximize=False,
                                     cut_tol=1e-8, max_cuts=60).value
                hi = certified_bound(model, c_hi, level, fit.p, maximize=True,
                                     cut_tol=1e-8, max_cuts=60).value
            if lo is not None:
                lowers[t].append(lo)
            if hi is not None:
                uppers[t].append(hi)

    def quantiles(values):
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            return None
        return {str(q): float(v) for q, v in zip(QUANTILES, np.quantile(arr, QUANTILES))}

    per_price = []
    for t in prices:
        per_price.append({
            "price": t,
            "n_used": len(lowers[t]),
            "lower_quantiles": quantiles(lowers[t]),
            "upper_quantiles": quantiles(uppers[t]),
        })
    return {
        "repeats": int(repeats),
        "converged": int(repeats) - failures,
        "convergence_failures": failures,
        "support_cells_per_replicate": {
            "min": min(support_sizes) if support_sizes else None,
            "max": max(support_sizes) if support_sizes else None,
        },
        "prices": per_price,
        "note": "每个复制内的 [L,U] 是部分识别范围而非抽样区间；分位数只描述识别集合"
                "的抽样波动，不把识别范围折叠成点估计，也不与识别不确定度相加。",
    }
