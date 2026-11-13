"""指定价位的购买概率 P(W > price) 的识别上下界。

两层未识别分开处理：
1) 似然层：最优质量分配不唯一，p 在水平集 {p 属于单纯形: ll(p) >= ll_ref - tol}
   内变动；该集合是凸集（凹函数的上水平集）。
2) 单元内位置层：价位落在某个开单元内部时，该单元质量在价位两侧的分布完全
   自由（申报只约束单元总质量）。绝不用单元中点伪造唯一尾概率。

记 c_lo = 完全位于 t 上方的单元指示，c_hi = c_lo + 跨越 t 的单元指示，则
    lower = min c_lo·p,  upper = max c_hi·p,  优化域为上述水平集。

两种求解共享 IntervalCensoredLikelihood：
- 容差为 0：log 严格凹 ⇒ 最优得分向量 s* = A p* 唯一，最优质量集是多面体
  {p >= 0, sum p = 1, A p = s*}，单次 LP 即得精确识别界（face_lp_bound）。
- 容差为正：certified_bound（Kelley 切平面法）。凹似然在任意点的支撑超平面
  都是水平集的外切半空间，任一中间 LP 值都是认证外界；迭代中每个边界二分
  点都是可行点，同步给出可行内值。外界与内值夹出真值，tight 表示已收紧。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize as sp_opt

from .npmle import IntervalCensoredLikelihood


def classify_cells(cells, t: float):
    """返回 (above, straddle)：完全在 t 上方 / 跨越 t 的单元指示向量。"""
    above = np.zeros(len(cells), dtype=bool)
    straddle = np.zeros(len(cells), dtype=bool)
    for j, cell in enumerate(cells):
        if cell.kind == "point":
            above[j] = cell.lo > t
        elif cell.kind == "gap":
            if cell.lo >= t:
                above[j] = True
            elif cell.lo < t < cell.hi:
                straddle[j] = True
        else:  # tail (lo, +inf)：无穷尾不在任何有限价位处封死
            if cell.lo >= t:
                above[j] = True
            else:
                straddle[j] = True
    return above, straddle


@dataclass
class CertifiedBound:
    value: float | None  # 认证外界（LP 松弛值）
    inner: float | None  # 可行点达到的内值
    tight: bool
    cuts: int


def face_lp_bound(model: IntervalCensoredLikelihood, c, s_star,
                  maximize: bool) -> float | None:
    """最优面 {p 属于单纯形: A p = s*} 上 min/max c·p 的精确 LP 界。

    log 严格凹 ⇒ 最优得分向量 s* = A p* 唯一，最优质量集正是多面体
    {p >= 0, sum p = 1, A p = s*}，故容差为 0 时单次 LP 即精确识别界。
    """
    c = np.asarray(c, dtype=float)
    direction = -1.0 if maximize else 1.0
    A_eq = np.vstack([model.A, np.ones(model.m)])
    b_eq = np.concatenate([np.asarray(s_star, dtype=float), [1.0]])
    res = sp_opt.linprog(direction * c, A_eq=A_eq, b_eq=b_eq,
                         bounds=[(0.0, None)] * model.m, method="highs")
    if res.status != 0:
        return None
    return float(direction * res.fun) + 0.0  # 归一化 -0.0


def _boundary_point(model, inside, outside, level, iters=80):
    """在 inside（ll>=level）与 outside（ll<level 或 -inf）之间二分出边界点。"""
    lo, hi = inside.copy(), outside.copy()
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if model.loglik(mid) >= level:
            lo = mid
        else:
            hi = mid
    return lo


def certified_bound(model: IntervalCensoredLikelihood, c, level, p_ref,
                    maximize: bool, cut_tol=1e-9, max_cuts=100) -> CertifiedBound:
    """Kelley 切平面法：水平集 {ll >= level} 上线性目标 min/max c·p 的认证括号。

    凹函数 ll 在 q 处满足 ll(p) <= ll(q) + g_q·(p - q)，因此水平集被包含在
    半空间 g_q·p >= g_q·q + level - ll(q) 内；逐步加切平面求 LP，任一阶段
    的 LP 最优值都是真值的认证外界（certified outer）。迭代中遇到的每个
    可行点（p_ref、边界二分点、最终 LP 解）给出可行内值（inner），两者
    夹出真值；tight=True 表示 LP 解已可行或括号宽度 <= 1e-6。
    """
    direction = -1.0 if maximize else 1.0
    c = np.asarray(c, dtype=float)
    obj = direction * c
    G: list = []
    h: list = []

    def add_cut(q):
        g = model.gradient(q)
        G.append(g)
        h.append(float(g @ q) + level - model.loglik(q))

    inner = None

    def consider(q, tol=0.0):
        nonlocal inner
        if model.loglik(q) >= level - tol:
            v = float(c @ q)
            if inner is None or direction * v < direction * inner:
                inner = v

    consider(p_ref)
    add_cut(p_ref)
    value, tight = None, False
    for _ in range(max_cuts):
        res = sp_opt.linprog(
            obj,
            A_ub=-np.asarray(G),
            b_ub=-np.asarray(h),
            A_eq=np.ones((1, model.m)),
            b_eq=np.ones(1),
            bounds=[(0.0, None)] * model.m,
            method="highs",
        )
        if res.status != 0:
            break
        value = float(direction * res.fun) + 0.0  # 归一化 -0.0
        if model.loglik(res.x) >= level - cut_tol:
            consider(res.x, tol=cut_tol)
            tight = True
            break
        q = _boundary_point(model, p_ref, res.x, level)
        consider(q)
        add_cut(q)
    if value is not None and inner is not None and abs(value - inner) <= 1e-6:
        tight = True
    return CertifiedBound(value=value, inner=inner, tight=tight, cuts=len(G))


@dataclass
class TailBound:
    price: float
    lower: float | None          # 认证外界（容差>0 时 Kelley，容差=0 时最优面 LP）
    upper: float | None          # 认证外界
    inner_lower: float | None    # 可行内值（真实下界 <= 该值）
    inner_upper: float | None    # 可行内值（真实上界 >= 该值）
    tight: bool
    straddle_mass_ref: float     # 参考最优点处跨越价位的单元质量
    within_cell_unidentified: bool
    ref_value: float             # 单个最优点的尾概率（不代表唯一值）


def tail_bounds(model: IntervalCensoredLikelihood, cells, p_ref, ll_ref, prices,
                ll_tol, cut_tol=1e-9, max_cuts=100) -> list:
    """对每个价位求 P(W > price) 在最优似然容差集合内的识别上下界。"""
    level = ll_ref - ll_tol
    out = []
    for k, t in enumerate(prices):
        t = float(t)
        if t < 0:
            raise ValueError("价位不能为负：潜在支付意愿非负")
        above, straddle = classify_cells(cells, t)
        c_lo = above.astype(float)
        c_hi = (above | straddle).astype(float)
        if ll_tol <= 0.0:
            # 水平集即最优面 {A p = s*}：单次 LP 精确求解
            s_star = model.scores(p_ref)
            lo_v = face_lp_bound(model, c_lo, s_star, maximize=False)
            hi_v = face_lp_bound(model, c_hi, s_star, maximize=True)
            lo = CertifiedBound(lo_v, lo_v, True, 1)
            hi = CertifiedBound(hi_v, hi_v, True, 1)
        else:
            lo = certified_bound(model, c_lo, level, p_ref, maximize=False,
                                 cut_tol=cut_tol, max_cuts=max_cuts)
            hi = certified_bound(model, c_hi, level, p_ref, maximize=True,
                                 cut_tol=cut_tol, max_cuts=max_cuts)
        out.append(TailBound(
            price=t,
            lower=lo.value,
            upper=hi.value,
            inner_lower=lo.inner,
            inner_upper=hi.inner,
            tight=lo.tight and hi.tight,
            straddle_mass_ref=float(straddle @ np.asarray(p_ref, dtype=float)),
            within_cell_unidentified=bool(straddle.any()),
            ref_value=float(c_lo @ np.asarray(p_ref, dtype=float)),
        ))
    return out
