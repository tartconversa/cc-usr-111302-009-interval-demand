import unittest
from pathlib import Path

from src import demand_reader
from src.analyze import build_report
from src.bootstrap import bootstrap
from src.declarations import parse_declarations

WORKED = [
    {"institution": "i1", "kind": "interval", "L": "0", "R": "10", "point": ""},
    {"institution": "i2", "kind": "interval", "L": "10", "R": "20", "point": ""},
]

OVERLAP = [
    {"institution": "i1", "kind": "interval", "L": "0", "R": "10", "point": ""},
    {"institution": "i2", "kind": "interval", "L": "5", "R": "15", "point": ""},
]


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.decls = parse_declarations(WORKED)

    def test_structure_and_quantiles(self):
        out = bootstrap(self.decls, [5.0, 10.0], repeats=40, seed=1,
                        kkt_tol=1e-8, max_iter=2000, ll_tol=1e-8)
        self.assertEqual(out["repeats"], 40)
        self.assertEqual(out["convergence_failures"], 0)
        self.assertEqual(out["converged"], 40)
        # 每次重建支持：复制 (i1,i1)/(i2,i2) 有 2 个单元，混合复制有 4 个
        sizes = out["support_cells_per_replicate"]
        self.assertEqual(sizes["min"], 2)
        self.assertEqual(sizes["max"], 4)
        for entry in out["prices"]:
            self.assertEqual(entry["n_used"], 40)
            self.assertLessEqual(entry["lower_quantiles"]["0.5"],
                                 entry["upper_quantiles"]["0.5"])

    def test_identification_range_not_collapsed_to_point(self):
        out = bootstrap(self.decls, [5.0], repeats=60, seed=5,
                        kkt_tol=1e-8, max_iter=2000, ll_tol=1e-8)
        entry = out["prices"][0]
        # 识别范围 [L,U] 不是抽样区间：下界与上界分别汇总，不得折叠成点
        self.assertGreater(entry["upper_quantiles"]["0.5"],
                           entry["lower_quantiles"]["0.5"])

    def test_convergence_failures_counted(self):
        decls = parse_declarations(OVERLAP)
        out = bootstrap(decls, [5.0], repeats=24, seed=2,
                        kkt_tol=1e-12, max_iter=1, ll_tol=1e-8)
        # max_iter=1 时混合复制（均匀起点非最优）无法收敛，计入失败
        self.assertGreater(out["convergence_failures"], 0)
        self.assertEqual(out["converged"] + out["convergence_failures"], 24)
        self.assertEqual(out["prices"][0]["n_used"], out["converged"])


class FixturePipelineTests(unittest.TestCase):
    def test_fixture_report(self):
        root = Path(__file__).resolve().parents[1]
        decls = parse_declarations(demand_reader.load(root / "fixtures" / "intervals.csv"))
        config = dict(demand_reader.load(root / "fixtures" / "demand.json"))
        config["bootstrap_repeats"] = 25
        report = build_report(decls, config)
        bounds = {b["price"]: b for b in report["tail_bounds"]}
        # 配置 likelihood_tolerance=1e-8：界是最优似然容差集合内的范围，
        # 相对最优面值 0.5 有约 5e-5 的容差放宽
        self.assertAlmostEqual(bounds[5.0]["lower"], 0.5, delta=2e-4)
        self.assertAlmostEqual(bounds[5.0]["upper"], 1.0, delta=2e-4)
        self.assertAlmostEqual(bounds[10.0]["lower"], 0.5, delta=2e-4)
        self.assertAlmostEqual(bounds[10.0]["upper"], 0.5, delta=2e-4)
        self.assertAlmostEqual(bounds[20.0]["upper"], 0.0, places=8)
        self.assertEqual(report["bootstrap"]["repeats"], 25)
        self.assertEqual(report["input"]["config_warnings"], [])
        self.assertTrue(report["fit"]["nonunique"]["same_likelihood_distinct_fits"])
        self.assertTrue(report["caveats"])


if __name__ == "__main__":
    unittest.main()
