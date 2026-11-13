"""支持单元构造。

由全部申报端点与点质量把 [0,+inf) 划分成互不重叠、覆盖模式（被哪些申报
包含）在单元内不变的单元：端点单点 {e}、相邻端点间的开区间 (a,b)、以及
最大端点之外的无穷尾 (e_max,+inf)。A_ij=1 当且仅当单元 j 完全落在申报 i
的域内。未被任何申报覆盖的单元不进入优化（其质量恒为 0），但保留清单
以便核查；无穷尾单元只要被右删失申报覆盖就保留，绝不封存到最大观测价。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .declarations import Declaration


@dataclass(frozen=True)
class Cell:
    kind: str  # "point" | "gap" | "tail"
    lo: float
    hi: float  # point: 等于 lo；tail: math.inf

    def label(self) -> str:
        if self.kind == "point":
            return f"{{{self.lo:g}}}"
        if self.kind == "gap":
            return f"({self.lo:g}, {self.hi:g})"
        return f"({self.lo:g}, +inf)"


@dataclass
class Support:
    cells: list[Cell]
    A: np.ndarray  # (n_declarations, m_cells)，元素 0/1
    dropped: list[Cell]  # 未被任何申报覆盖的单元（质量恒为 0）


def _covers(decl: Declaration, cell: Cell) -> bool:
    if cell.kind == "point":
        return decl.contains_point(cell.lo)
    return decl.contains_open_cell(cell.lo, cell.hi)


def build_support(declarations: list[Declaration]) -> Support:
    endpoints = sorted({e for d in declarations for e in d.endpoints()})
    candidates: list[Cell] = [Cell("point", e, e) for e in endpoints]
    candidates += [Cell("gap", a, b) for a, b in zip(endpoints, endpoints[1:])]
    candidates.append(Cell("tail", endpoints[-1], math.inf))
    candidates.sort(key=lambda c: (c.lo, c.hi))
    cells: list[Cell] = []
    dropped: list[Cell] = []
    for cell in candidates:
        if any(_covers(d, cell) for d in declarations):
            cells.append(cell)
        else:
            dropped.append(cell)
    A = np.zeros((len(declarations), len(cells)))
    for i, decl in enumerate(declarations):
        for j, cell in enumerate(cells):
            if _covers(decl, cell):
                A[i, j] = 1.0
    if (A.sum(axis=1) <= 0).any():
        bad = declarations[int(np.argmin(A.sum(axis=1)))].institution
        raise ValueError(f"机构 {bad} 的申报域不包含任何支持单元")
    return Support(cells=cells, A=A, dropped=dropped)
