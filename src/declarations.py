"""机构自愿申报的区间删失记录：类型化解析与校验。

demand_reader 只负责把文件读成原始行，本模块把原始行解释成声明对象，
仍不做任何统计推断。约定：interval=(L,R]，left=[0,R]，right=(L,+inf)，
point={x}；潜在支付意愿 W 非负；每个独立机构只接受一次有效申报。
空端点表示未使用，不等于 0。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

KINDS = ("interval", "left", "right", "point")


@dataclass(frozen=True)
class Declaration:
    """一次有效申报；端点开闭由 kind 决定，必须保留。"""

    institution: str
    kind: str
    left: float | None = None
    right: float | None = None
    point: float | None = None

    def endpoints(self) -> tuple[float, ...]:
        """申报域的有限端点（用于划分支持单元）。"""
        if self.kind == "interval":
            return (self.left, self.right)
        if self.kind == "left":
            return (0.0, self.right)
        if self.kind == "right":
            return (self.left,)
        return (self.point,)

    def contains_point(self, x: float) -> bool:
        """点 x 是否落在申报域内（区分开闭）。"""
        if self.kind == "interval":
            return self.left < x <= self.right
        if self.kind == "left":
            return 0.0 <= x <= self.right
        if self.kind == "right":
            return x > self.left
        return x == self.point

    def contains_open_cell(self, a: float, b: float) -> bool:
        """开单元 (a,b)（b 可为 +inf）是否完全落在申报域内。"""
        if self.kind == "interval":
            return self.left <= a and b <= self.right
        if self.kind == "left":
            return 0.0 <= a and b <= self.right
        if self.kind == "right":
            return self.left <= a
        return False


def _is_blank(value) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def _number(value, field: str, institution: str) -> float | None:
    if _is_blank(value):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"机构 {institution} 的 {field} 不是数值: {value!r}") from None
    if not math.isfinite(out):
        raise ValueError(f"机构 {institution} 的 {field} 必须有限: {value!r}")
    return out


def _require(cond: bool, message: str) -> None:
    if not cond:
        raise ValueError(message)


def parse_declarations(records) -> list[Declaration]:
    """把原始行解析为声明并校验契约。

    重复机构、非法 kind、缺端点、端点误用（空端点之外的列必须留空）、
    负值、L>=R 均报错；空字符串端点表示未使用，不等于 0。
    """
    declarations: list[Declaration] = []
    seen: set[str] = set()
    for row in records:
        institution = str(row.get("institution", "") or "").strip()
        _require(bool(institution), "存在缺少机构标识的申报行")
        _require(
            institution not in seen,
            f"机构 {institution} 重复申报：每个独立机构只接受一次有效申报",
        )
        seen.add(institution)
        kind = str(row.get("kind", "") or "").strip()
        _require(kind in KINDS, f"机构 {institution} 的 kind 非法: {kind!r}")
        left = _number(row.get("L"), "L", institution)
        right = _number(row.get("R"), "R", institution)
        point = _number(row.get("point"), "point", institution)
        if kind == "interval":
            _require(
                left is not None and right is not None and point is None,
                f"机构 {institution} 的 interval 需要 L 与 R 且 point 留空",
            )
            _require(0.0 <= left < right, f"机构 {institution} 需要 0 <= L < R")
        elif kind == "left":
            _require(
                right is not None and left is None and point is None,
                f"机构 {institution} 的 left 需要 R 且 L、point 留空",
            )
            _require(right >= 0.0, f"机构 {institution} 的 R 不能为负")
        elif kind == "right":
            _require(
                left is not None and right is None and point is None,
                f"机构 {institution} 的 right 需要 L 且 R、point 留空",
            )
            _require(left >= 0.0, f"机构 {institution} 的 L 不能为负")
        else:
            _require(
                point is not None and left is None and right is None,
                f"机构 {institution} 的 point 需要 point 且 L、R 留空",
            )
            _require(point >= 0.0, f"机构 {institution} 的 point 不能为负")
        declarations.append(Declaration(institution, kind, left, right, point))
    _require(bool(declarations), "没有任何有效申报")
    return declarations
