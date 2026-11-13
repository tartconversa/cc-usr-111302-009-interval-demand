"""命令行入口：读取合成机构申报与配置，输出区间删失非参数分布分析报告(JSON)。

用法:
    python -m src.analyze [intervals.csv] [demand.json]
默认读取 fixtures/intervals.csv 与 fixtures/demand.json。输入为虚构机构的
自愿价格区间申报，本工具只做总体层面的部分识别分析，不做个人定价。
"""

from __future__ import annotations

import argparse
import json

from . import demand_reader
from .bootstrap import bootstrap
from .bounds import tail_bounds
from .declarations import parse_declarations
from .npmle import IntervalCensoredLikelihood, fit_multi, zero_mass_boundary
from .support import build_support

EXPECTED_TAIL_SCOPE = "likelihood_optimal_mass_and_within_cell_position"

CAVEATS = [
    "自愿申报存在自选择偏差：愿意申报价格区间的机构未必代表未申报机构，"
    "估计只描述申报子总体，不能当作市场总体的随机样本。",
    "申报区间是机构声明的意愿范围，不是均匀分布假设下的抽样，也不是成交记录；"
    "本工具不做个人定价。",
    "输出的 [L,U] 是部分识别范围（似然等价 + 单元内位置未识别），不是置信区间；"
    "bootstrap 只量化识别集合的抽样波动，两者不可相加或互相替代。",
    "超出最大观测端点的尾概率：无右删失申报时 NPMLE 支持不覆盖该区域，0 是支持集"
    "假设而非数据证据；有右删失申报时由无穷尾单元承载，具体价位处不可识别，不能外推。",
    "需求曲线是静态申报意愿的汇总，不能用于预测调价后的行为反应或收入最大化。",
]


def _cell_note(cells, t, has_tail_cell):
    straddle_tail = any(c.kind == "tail" and c.lo < t for c in cells)
    if straddle_tail:
        return "价位落在无穷尾单元内：该单元质量在价位两侧的分布不可识别，未封存到最大观测价。"
    if not has_tail_cell and cells and t >= max(c.hi for c in cells if c.kind != "tail"):
        return "价位达到或超过最大观测端点且无右删失申报：0 是支持集假设而非数据证据，不能外推。"
    return ""


def build_report(declarations, config) -> dict:
    prices = [float(t) for t in config["prices"]]
    ll_tol = float(config.get("likelihood_tolerance", 1e-8))
    kkt_tol = float(config.get("kkt_tolerance", 1e-8))
    max_iter = int(config.get("max_iterations", 2000))
    repeats = int(config.get("bootstrap_repeats", 200))
    seed = int(config.get("seed", 0))

    warnings = []
    if config.get("nonnegative_support") is not True:
        warnings.append("配置未声明 nonnegative_support=true；本分析假设潜在支付意愿非负。")
    if config.get("interval_convention") not in (None, "(L,R]"):
        warnings.append(f"区间约定 {config.get('interval_convention')!r} 与实现的 (L,R] 不一致。")
    if config.get("tail_scope") not in (None, EXPECTED_TAIL_SCOPE):
        warnings.append(f"tail_scope {config.get('tail_scope')!r} 未知，仍按 {EXPECTED_TAIL_SCOPE} 计算。")

    support = build_support(declarations)
    model = IntervalCensoredLikelihood(support.A)
    multi = fit_multi(model, kkt_tol=kkt_tol, max_iter=max_iter, ll_tol=ll_tol, seed=seed)
    bounds = tail_bounds(model, support.cells, multi.p_ref, multi.loglik_ref,
                         prices, ll_tol)
    boot = bootstrap(declarations, prices, repeats, seed, kkt_tol, max_iter, ll_tol)
    boundary = zero_mass_boundary(model, multi.p_ref,
                                  active_margin=max(1e-6, 100.0 * kkt_tol))
    has_tail_cell = any(c.kind == "tail" for c in support.cells)

    fits_summary = [{
        "method": f.method,
        "converged": f.converged,
        "iterations": f.iterations,
        "loglik": f.loglik,
        "kkt_residual": f.kkt_residual,
    } for f in multi.fits]

    return {
        "input": {
            "declarations": len(declarations),
            "institutions": [d.institution for d in declarations],
            "prices": prices,
            "likelihood_tolerance": ll_tol,
            "kkt_tolerance": kkt_tol,
            "max_iterations": max_iter,
            "config_warnings": warnings,
        },
        "objective": "max sum_i log(sum_j A_ij p_j), p >= 0, sum p = 1",
        "kkt_condition": "d_j = sum_i A_ij/s_i；支撑上 d_j = n，支撑外 d_j <= n；"
                         "kkt_residual = max(d) - n >= 0",
        "support": {
            "cells": [{
                "index": j,
                "label": cell.label(),
                "kind": cell.kind,
                "covered_by": [declarations[i].institution
                               for i in range(len(declarations)) if support.A[i, j] > 0],
            } for j, cell in enumerate(support.cells)],
            "dropped_uncovered": [c.label() for c in support.dropped],
            "note": "支持单元由全部端点与点质量划分，覆盖模式在单元内不变；"
                    "未被任何申报覆盖的单元质量恒为 0。",
        },
        "fit": {
            "loglik": multi.loglik_ref,
            "method": multi.best.method,
            "converged": multi.best.converged,
            "kkt_residual": multi.best.kkt_residual,
            "masses": [{"cell": support.cells[j].label(), "p": float(multi.p_ref[j])}
                       for j in range(model.m)],
            "zero_mass_boundary": [
                {**entry, "cell": support.cells[entry["index"]].label()}
                for entry in boundary
            ],
            "fits": fits_summary,
            "nonunique": {
                "spread": multi.nonunique_spread,
                "same_likelihood_distinct_fits": multi.nonunique_spread > 1e-6,
                "note": "多个最优质量分配可给出相同似然；单个 EM 停止点不代表唯一分布，"
                        "尾概率须在容差集合内求上下界。",
            },
        },
        "tail_bounds": [{
            "price": b.price,
            "lower": b.lower,
            "upper": b.upper,
            "inner_lower": b.inner_lower,
            "inner_upper": b.inner_upper,
            "certified_tight": b.tight,
            "single_fit_value": b.ref_value,
            "straddle_mass_at_ref": b.straddle_mass_ref,
            "within_cell_unidentified": b.within_cell_unidentified,
            "note": _cell_note(support.cells, b.price, has_tail_cell),
        } for b in bounds],
        "bootstrap": boot,
        "caveats": CAVEATS,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="区间删失申报的非参数分布分析（合成数据）")
    parser.add_argument("intervals", nargs="?", default="fixtures/intervals.csv")
    parser.add_argument("config", nargs="?", default="fixtures/demand.json")
    args = parser.parse_args(argv)
    declarations = parse_declarations(demand_reader.load(args.intervals))
    config = demand_reader.load(args.config)
    report = build_report(declarations, config)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
