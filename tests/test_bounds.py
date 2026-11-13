import unittest

from src.bounds import tail_bounds
from src.declarations import parse_declarations
from src.npmle import IntervalCensoredLikelihood, fit_multi
from src.support import build_support

WORKED = [
    {"institution": "i1", "kind": "interval", "L": "0", "R": "10", "point": ""},
    {"institution": "i2", "kind": "interval", "L": "10", "R": "20", "point": ""},
]


def bounds_for(rows, prices, ll_tol=0.0):
    # ll_tol=0：水平集即最优面，界值为精确识别范围；
    # ll_tol>0：界随容差集合放宽（另由 ToleranceTests 覆盖）。
    decls = parse_declarations(rows)
    support = build_support(decls)
    model = IntervalCensoredLikelihood(support.A)
    multi = fit_multi(model, kkt_tol=1e-8, max_iter=2000, ll_tol=ll_tol, seed=3)
    return tail_bounds(model, support.cells, multi.p_ref, multi.loglik_ref,
                       prices, ll_tol)


class WorkedExampleBoundTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bounds = {b.price: b for b in bounds_for(WORKED, [5.0, 10.0, 20.0])}

    def test_price_at_shared_boundary_is_exact(self):
        b = self.bounds[10.0]
        self.assertAlmostEqual(b.lower, 0.5, places=6)
        self.assertAlmostEqual(b.upper, 0.5, places=6)

    def test_price_inside_cell_has_range_not_midpoint(self):
        b = self.bounds[5.0]
        self.assertTrue(b.within_cell_unidentified)
        self.assertAlmostEqual(b.lower, 0.5, places=6)
        self.assertAlmostEqual(b.upper, 1.0, places=6)  # 不用质量中点伪造 0.75

    def test_price_at_max_endpoint_zero_by_support_assumption(self):
        b = self.bounds[20.0]
        self.assertAlmostEqual(b.lower, 0.0, places=8)
        self.assertAlmostEqual(b.upper, 0.0, places=8)

    def test_certified_outer_brackets_feasible_inner(self):
        # 容差为正时：认证外界与可行内值夹出真值，且二者收紧到一致
        for b in bounds_for(WORKED, [5.0, 10.0, 20.0], ll_tol=1e-8):
            self.assertTrue(b.tight)
            self.assertIsNotNone(b.inner_lower)
            self.assertIsNotNone(b.inner_upper)
            self.assertLessEqual(b.lower, b.inner_lower + 1e-6)
            self.assertGreaterEqual(b.upper, b.inner_upper - 1e-6)
            self.assertAlmostEqual(b.lower, b.inner_lower, places=4)
            self.assertAlmostEqual(b.upper, b.inner_upper, places=4)


class RepeatedIntervalTests(unittest.TestCase):
    def test_identical_intervals_full_range_inside_cell(self):
        rows = [
            {"institution": "a", "kind": "interval", "L": "0", "R": "10", "point": ""},
            {"institution": "b", "kind": "interval", "L": "0", "R": "10", "point": ""},
        ]
        (b5,) = bounds_for(rows, [5.0])
        self.assertAlmostEqual(b5.lower, 0.0, places=6)
        self.assertAlmostEqual(b5.upper, 1.0, places=6)


class RightCensorTailTests(unittest.TestCase):
    def test_infinite_tail_not_clamped(self):
        rows = [
            {"institution": "a", "kind": "right", "L": "5", "R": "", "point": ""},
            {"institution": "b", "kind": "right", "L": "10", "R": "", "point": ""},
        ]
        b10, b100 = bounds_for(rows, [10.0, 100.0])
        # 价 10：无穷尾 (10,+inf) 完全在上方
        self.assertAlmostEqual(b10.lower, 1.0, places=6)
        self.assertAlmostEqual(b10.upper, 1.0, places=6)
        # 价 100：尾单元跨越价位，质量位置不可识别，不封成最大观测价
        self.assertAlmostEqual(b100.lower, 0.0, places=6)
        self.assertAlmostEqual(b100.upper, 1.0, places=6)


class BoundaryTouchingTests(unittest.TestCase):
    def test_closed_zero_endpoint_creates_point_cell(self):
        rows = [
            {"institution": "a", "kind": "left", "L": "", "R": "10", "point": ""},
            {"institution": "b", "kind": "interval", "L": "10", "R": "20", "point": ""},
        ]
        (b0,) = bounds_for(rows, [0.0])
        # {0} 单点属于 [0,10]；P(W>0) = 1 - p_{0}，p_{0} 在 [0,1/2] 内不可识别
        self.assertAlmostEqual(b0.lower, 0.5, places=6)
        self.assertAlmostEqual(b0.upper, 1.0, places=6)


class SingletonBoundTests(unittest.TestCase):
    def test_exact_points(self):
        rows = [
            {"institution": "a", "kind": "point", "L": "", "R": "", "point": "5"},
            {"institution": "b", "kind": "point", "L": "", "R": "", "point": "5"},
            {"institution": "c", "kind": "point", "L": "", "R": "", "point": "8"},
        ]
        b5, b8 = bounds_for(rows, [5.0, 8.0])
        self.assertAlmostEqual(b5.lower, 1.0 / 3.0, places=6)
        self.assertAlmostEqual(b5.upper, 1.0 / 3.0, places=6)
        self.assertAlmostEqual(b8.lower, 0.0, places=8)
        self.assertAlmostEqual(b8.upper, 0.0, places=8)


class ToleranceTests(unittest.TestCase):
    def test_wider_likelihood_tolerance_widens_bounds(self):
        (tight,) = bounds_for(WORKED, [5.0], ll_tol=1e-8)
        (loose,) = bounds_for(WORKED, [5.0], ll_tol=0.5)
        self.assertLess(loose.lower, tight.lower - 1e-3)
        self.assertGreaterEqual(loose.upper, tight.upper - 1e-9)


if __name__ == "__main__":
    unittest.main()
