import math
import unittest

import numpy as np

from src.declarations import parse_declarations
from src.npmle import (
    IntervalCensoredLikelihood,
    fit_em,
    fit_multi,
    fit_slsqp,
    zero_mass_boundary,
)
from src.support import build_support

WORKED = [
    {"institution": "i1", "kind": "interval", "L": "0", "R": "10", "point": ""},
    {"institution": "i2", "kind": "interval", "L": "10", "R": "20", "point": ""},
]


def one_row(kind, L="", R="", point="", name="d"):
    return {"institution": name, "kind": kind, "L": L, "R": R, "point": point}


class DeclarationParsingTests(unittest.TestCase):
    def test_blank_endpoint_is_not_zero(self):
        (d,) = parse_declarations([one_row("right", L="5")])
        self.assertEqual(d.left, 5.0)
        self.assertIsNone(d.right)
        self.assertFalse(d.contains_point(5.0))  # (5,+inf) 开端点
        self.assertTrue(d.contains_point(5.5))

    def test_interval_open_closed(self):
        (d,) = parse_declarations([one_row("interval", L="0", R="10")])
        self.assertFalse(d.contains_point(0.0))  # (0,10] 开左端
        self.assertTrue(d.contains_point(10.0))  # 闭右端

    def test_left_censored_closed_at_zero(self):
        (d,) = parse_declarations([one_row("left", R="10")])
        self.assertTrue(d.contains_point(0.0))  # [0,10] 闭左端
        self.assertTrue(d.contains_point(10.0))
        self.assertFalse(d.contains_point(10.5))

    def test_duplicate_institution_rejected(self):
        with self.assertRaises(ValueError):
            parse_declarations([WORKED[0], dict(WORKED[0])])

    def test_malformed_rows_rejected(self):
        bad_rows = [
            one_row("interval", L="-1", R="5"),      # 负端点
            one_row("interval", L="5", R="5"),       # L 不小于 R
            one_row("interval", L="1", R="5", point="3"),  # point 未留空
            one_row("point"),                        # 缺 point
            one_row("left", L="0", R="5"),           # left 的 L 必须留空
            one_row("unknown", L="1", R="2"),        # 非法 kind
        ]
        for row in bad_rows:
            with self.subTest(row=row):
                with self.assertRaises(ValueError):
                    parse_declarations([row])


class SupportTests(unittest.TestCase):
    def test_worked_example_cells(self):
        support = build_support(parse_declarations(WORKED))
        self.assertEqual([c.label() for c in support.cells],
                         ["(0, 10)", "{10}", "(10, 20)", "{20}"])
        # {0} 与无穷尾未被任何申报覆盖，被丢弃且质量恒为 0
        self.assertEqual([c.label() for c in support.dropped], ["{0}", "(20, +inf)"])
        np.testing.assert_array_equal(
            support.A, np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]))

    def test_open_closed_preserved_in_coverage(self):
        support = build_support(parse_declarations(WORKED))
        self.assertEqual(support.A[0, 1], 1.0)  # {10} 属于 (0,10]
        self.assertEqual(support.A[1, 1], 0.0)  # {10} 不属于 (10,20]

    def test_left_censored_covers_zero_point(self):
        support = build_support(parse_declarations([
            one_row("left", R="10", name="a"),
            one_row("interval", L="10", R="20", name="b"),
        ]))
        self.assertIn("{0}", [c.label() for c in support.cells])


class EmFitTests(unittest.TestCase):
    def setUp(self):
        self.support = build_support(parse_declarations(WORKED))
        self.model = IntervalCensoredLikelihood(self.support.A)

    def test_em_reaches_kkt_and_expected_likelihood(self):
        fit = fit_em(self.model, kkt_tol=1e-8, max_iter=2000)
        self.assertTrue(fit.converged)
        self.assertLessEqual(fit.kkt_residual, 1e-8)
        self.assertAlmostEqual(fit.loglik, -2.0 * math.log(2.0), places=10)
        # 两个申报域各得 1/2 质量（域内拆分不唯一）
        self.assertAlmostEqual(fit.p[0] + fit.p[1], 0.5, places=8)
        self.assertAlmostEqual(fit.p[2] + fit.p[3], 0.5, places=8)

    def test_slsqp_agrees_with_em(self):
        em = fit_em(self.model)
        slsqp = fit_slsqp(self.model, em.p)
        self.assertTrue(slsqp.converged)
        self.assertAlmostEqual(slsqp.loglik, em.loglik, places=8)

    def test_likelihood_nonunique_across_starts(self):
        multi = fit_multi(self.model, kkt_tol=1e-8, max_iter=2000,
                          ll_tol=1e-8, restarts=4, seed=7)
        # 不同起点收敛到不同质量分配，但似然相同：单个 EM 停止点不代表唯一分布
        self.assertGreater(multi.nonunique_spread, 1e-3)
        for f in multi.fits:
            self.assertAlmostEqual(f.loglik, multi.loglik_ref, delta=1e-6)

    def test_zero_mass_boundary_pinned(self):
        decls = parse_declarations([
            one_row("interval", L="0", R="10", name="a"),
            one_row("point", point="10", name="b"),
        ])
        support = build_support(decls)
        model = IntervalCensoredLikelihood(support.A)
        fit = fit_slsqp(model)
        self.assertTrue(fit.converged)
        self.assertAlmostEqual(fit.p[1], 1.0, places=6)  # 全部质量在 {10}
        boundary = zero_mass_boundary(model, fit.p)
        self.assertEqual(len(boundary), 1)
        self.assertEqual(boundary[0]["status"], "pinned_at_zero")
        self.assertLess(boundary[0]["gradient"], model.n - 1e-3)

    def test_singletons_exact_mass(self):
        decls = parse_declarations([
            one_row("point", point="5", name="a"),
            one_row("point", point="5", name="b"),
            one_row("point", point="8", name="c"),
        ])
        support = build_support(decls)
        self.assertEqual([c.label() for c in support.cells], ["{5}", "{8}"])
        model = IntervalCensoredLikelihood(support.A)
        fit = fit_em(model, kkt_tol=1e-8, max_iter=2000)
        self.assertTrue(fit.converged)
        self.assertAlmostEqual(fit.p[0], 2.0 / 3.0, places=6)
        self.assertAlmostEqual(fit.p[1], 1.0 / 3.0, places=6)

    def test_all_right_censored_mass_in_infinite_tail(self):
        decls = parse_declarations([
            one_row("right", L="5", name="a"),
            one_row("right", L="10", name="b"),
        ])
        support = build_support(decls)
        self.assertEqual([c.label() for c in support.cells],
                         ["(5, 10)", "{10}", "(10, +inf)"])
        model = IntervalCensoredLikelihood(support.A)
        fit = fit_em(model, kkt_tol=1e-8, max_iter=2000)
        self.assertTrue(fit.converged)
        self.assertAlmostEqual(fit.p[2], 1.0, places=6)  # 全部质量在无穷尾


if __name__ == "__main__":
    unittest.main()
