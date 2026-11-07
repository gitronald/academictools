import polars as pl
import pytest

from tools.stats.nonparametric import (
    get_modal_category,
    kruskal_test,
    kruskal_test_multi,
    mannw_test,
    mcnemar_test,
    nonparametric_sig_test,
    p_value_sig,
    spearmanr,
)
from tools.stats.parametric import (
    paired_ttest,
    pearsonr_ci,
    z_test,
)


class TestKruskalTest:
    def test_basic(self):
        df = pl.DataFrame(
            {
                "group": ["a"] * 20 + ["b"] * 20,
                "metric": list(range(20)) + list(range(10, 30)),
            }
        )
        result = kruskal_test(df, "group", "metric")
        assert "chi" in result
        assert "P" in result

    def test_polars_input(self):
        df = pl.DataFrame(
            {
                "group": ["a"] * 20 + ["b"] * 20,
                "metric": list(range(20)) + list(range(10, 30)),
            }
        )
        result = kruskal_test(df, "group", "metric")
        assert "chi" in result
        assert "P" in result


class TestMannwTest:
    def test_basic(self):
        df = pl.DataFrame(
            {
                "group": ["a"] * 20 + ["b"] * 20,
                "metric": list(range(20)) + list(range(10, 30)),
            }
        )
        result = mannw_test(df, "group", "metric")
        assert "U" in result
        assert "P" in result


class TestSpearmanr:
    def test_basic(self):
        df = pl.DataFrame({"x": range(20), "y": range(20)})
        result = spearmanr(df, "x", "y")
        assert result["rho"] == pytest.approx(1.0)
        assert result["p"] == pytest.approx(0.0, abs=1e-10)

    def test_polars_input(self):
        df = pl.DataFrame({"x": range(20), "y": range(20)})
        result = spearmanr(df, "x", "y")
        assert result["rho"] == pytest.approx(1.0)


class TestNonparametricSigTest:
    def test_two_groups(self):
        df = pl.DataFrame(
            {
                "group": ["a"] * 20 + ["b"] * 20,
                "metric": list(range(20)) + list(range(10, 30)),
            }
        )
        result = nonparametric_sig_test(df, "group", "metric")
        assert "U" in result
        assert "p" in result

    def test_three_groups(self):
        df = pl.DataFrame(
            {
                "group": ["a"] * 15 + ["b"] * 15 + ["c"] * 15,
                "metric": list(range(15)) + list(range(5, 20)) + list(range(10, 25)),
            }
        )
        result = nonparametric_sig_test(df, "group", "metric")
        assert "chi" in result
        assert "p" in result


class TestPearsonrCi:
    def test_filters_null_and_nan_pairs(self):
        df = pl.DataFrame(
            {
                "x": [1.0, 2.0, float("nan"), 4.0, 5.0, None],
                "y": [1.0, 3.0, 8.0, 5.0, 7.0, 9.0],
            }
        )
        result = pearsonr_ci(df, "x", "y")
        expected = (
            __import__("scipy.stats", fromlist=["pearsonr"])
            .pearsonr([1.0, 2.0, 4.0, 5.0], [1.0, 3.0, 5.0, 7.0])
            .statistic
        )
        assert result["r"] == pytest.approx(expected)

    def test_basic(self):
        import numpy as np

        np.random.seed(42)
        x = np.arange(50, dtype=float)
        y = x + np.random.normal(0, 5, 50)
        df = pl.DataFrame({"x": x, "y": y})
        result = pearsonr_ci(df, "x", "y")
        assert 0 < result["r"] < 1
        assert "ci_min" in result
        assert "ci_max" in result
        assert result["ci_min"] < result["ci_max"]


class TestMcnemarTest:
    def test_one_category_table_has_only_observed_category(self):
        df = pl.DataFrame({"x": [0, 0], "y": [0, 0]})
        table, _ = mcnemar_test(df, "x", "y")
        assert table.columns == ["x", "0"]
        assert table.to_dict(as_series=False) == {"x": [0], "0": [2]}

    def test_numeric_order_and_constant_category(self):
        df = pl.DataFrame({"x": [2] * 9 + [10], "y": [2] * 9 + [10]})
        table, result = mcnemar_test(df, "x", "y")
        assert table.columns == ["x", "2", "10"]
        assert table.drop("x").to_numpy().tolist() == [[9, 0], [0, 1]]
        assert result["pvalue"] == pytest.approx(1.0)

    def test_constant_category_and_too_many_categories(self):
        df = pl.DataFrame({"x": [0, 0, 0], "y": [1, 1, 1]})
        table, _ = mcnemar_test(df, "x", "y")
        assert table.drop("x").shape == (2, 2)
        with pytest.raises(ValueError, match="no more than two categories"):
            mcnemar_test(pl.DataFrame({"x": [0, 1, 2], "y": [0, 1, 2]}), "x", "y")

    def test_basic(self):
        df = pl.DataFrame(
            {
                "test1": [1, 1, 0, 0, 1, 0, 1, 0] * 10,
                "test2": [1, 0, 1, 0, 1, 1, 0, 0] * 10,
            }
        )
        xtab, output = mcnemar_test(df, "test1", "test2")
        assert isinstance(xtab, pl.DataFrame)
        # First column holds the group1 (rows); group2 values are count columns
        assert xtab.columns[0] == "test1"
        assert "pvalue" in output
        assert "chi2" in output

    def test_contingency_matches_counts(self):
        # 2x2 contingency from explicit pairs: (1,1)x2, (1,0)x1, (0,1)x1, (0,0)x1
        df = pl.DataFrame(
            {
                "test1": [1, 1, 1, 0, 0],
                "test2": [1, 1, 0, 1, 0],
            }
        )
        xtab, _ = mcnemar_test(df, "test1", "test2")
        # rows sorted by test1 (0, 1); count columns sorted by test2 (0, 1)
        assert xtab["test1"].to_list() == [0, 1]
        assert xtab.drop("test1").to_numpy().tolist() == [[1, 1], [1, 2]]


class TestGetModalCategory:
    def test_basic(self):
        df = pl.DataFrame(
            {
                "group": ["a", "a", "a", "b", "b", "b"],
                "cat": ["x", "x", "y", "y", "y", "z"],
            }
        )
        result = get_modal_category(df, "group", "cat")
        assert isinstance(result, pl.DataFrame)
        assert result.columns == ["group", "category"]
        # Group "a" most frequent is "x", group "b" is "y"
        assert result.to_dicts() == [
            {"group": "a", "category": "x"},
            {"group": "b", "category": "y"},
        ]


class TestPValueSig:
    def test_three_stars(self):
        assert p_value_sig(0.0001) == "***"

    def test_two_stars(self):
        assert p_value_sig(0.005) == "**"

    def test_one_star(self):
        assert p_value_sig(0.03) == "*"

    def test_not_significant(self):
        assert p_value_sig(0.1) == ""


class TestZTest:
    def test_basic(self):
        result = z_test([10, 20], [100, 100])
        assert "z" in result
        assert "p" in result

    def test_equal_proportions(self):
        result = z_test([50, 50], [100, 100])
        assert result["z"] == pytest.approx(0.0)
        assert result["p"] == pytest.approx(1.0)


class TestKruskalTestMulti:
    def test_basic(self, capsys):
        df = pl.DataFrame(
            {
                "group": ["a"] * 20 + ["b"] * 20,
                "metric1": list(range(20)) + list(range(10, 30)),
                "metric2": list(range(0, 40, 2)) + list(range(1, 41, 2)),
            }
        )
        kruskal_test_multi(df, metrics=["metric1"], groups=["group"])
        captured = capsys.readouterr()
        assert "Mean metric1" in captured.out
        assert "Kruskal-Wallis" in captured.out

    def test_multiple_metrics_groups(self, capsys):
        df = pl.DataFrame(
            {
                "g1": ["a"] * 15 + ["b"] * 15,
                "g2": (["x"] * 10 + ["y"] * 5) * 2,
                "m1": list(range(30)),
                "m2": list(range(0, 60, 2)),
            }
        )
        kruskal_test_multi(df, metrics=["m1", "m2"], groups=["g1", "g2"])
        captured = capsys.readouterr()
        assert "Mean m1" in captured.out
        assert "Mean m2" in captured.out

    def test_no_details(self, capsys):
        df = pl.DataFrame(
            {
                "group": ["a"] * 20 + ["b"] * 20,
                "metric": list(range(40)),
            }
        )
        kruskal_test_multi(df, metrics=["metric"], groups=["group"], details=False)
        captured = capsys.readouterr()
        assert "Kruskal-Wallis" in captured.out
        # describe table should NOT be printed
        assert "count" not in captured.out

    def test_polars_input(self, capsys):
        df = pl.DataFrame(
            {
                "group": ["a"] * 20 + ["b"] * 20,
                "metric": list(range(20)) + list(range(10, 30)),
            }
        )
        kruskal_test_multi(df, metrics=["metric"], groups=["group"])
        captured = capsys.readouterr()
        assert "Kruskal-Wallis" in captured.out


class TestPairedTtest:
    def test_means_use_same_complete_pairs_as_test(self):
        df = pl.DataFrame(
            {
                "var1": [1.0, 100.0, 3.0, 5.0, float("nan")],
                "var2": [2.0, None, 4.0, 8.0, 9.0],
            }
        )
        result = paired_ttest(df, "var1", "var2")
        assert result["var1"] == pytest.approx(3.0)
        assert result["var2"] == pytest.approx(14 / 3)
        assert result["effect_size"] == pytest.approx(5 / 3)

    def test_basic(self):
        import numpy as np

        rng = np.random.default_rng(42)
        var1 = list(range(20))
        var2 = [x + 2 + rng.normal(0, 0.5) for x in var1]
        df = pl.DataFrame(
            {
                "root": ["all"] * 20,
                "var1": var1,
                "var2": var2,
            }
        )
        result = paired_ttest(df, "var1", "var2")
        assert "root" in result
        assert "t" in result
        assert "P" in result
        assert result["effect_size"] == pytest.approx(2.0, abs=0.5)

    def test_with_root_filter(self):
        import numpy as np

        rng = np.random.default_rng(42)
        var1 = list(range(20))
        var2 = [x + 1 + rng.normal(0, 0.5) for x in var1]
        df = pl.DataFrame(
            {
                "root": ["a"] * 10 + ["b"] * 10,
                "var1": var1,
                "var2": var2,
            }
        )
        result = paired_ttest(df, "var1", "var2", root="a")
        assert result["root"] == "a"

    def test_with_date_filter(self):
        import numpy as np

        rng = np.random.default_rng(42)
        var1 = list(range(20))
        var2 = [x + 1 + rng.normal(0, 0.5) for x in var1]
        df = pl.DataFrame(
            {
                "root": ["all"] * 20,
                "date": ["2023-01-01"] * 10 + ["2023-01-02"] * 10,
                "var1": var1,
                "var2": var2,
            }
        )
        result = paired_ttest(df, "var1", "var2", date="2023-01-01")
        assert "t" in result
        assert "P" in result

    def test_empty_raises(self):
        # all-null columns -> Series.mean() is None; expect a clear ValueError,
        # not a TypeError from None arithmetic
        df = pl.DataFrame({"var1": [None, None], "var2": [None, None]})
        with pytest.raises(ValueError, match="no non-null values"):
            paired_ttest(df, "var1", "var2")
