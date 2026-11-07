import os
import tempfile

import polars as pl
import pytest

from tools.dataframes.dates import decompose_dates, process_timestamps
from tools.dataframes.describe import describe
from tools.dataframes.dft import (
    binned_col,
    dft,
    get_binned_dft,
    neat_n,
    neat_p,
    presentation_table,
)
from tools.dataframes.display import (
    display_tables,
    display_tables_html,
    display_tables_text,
    set_css,
)
from tools.dataframes.inspection import (
    dfshape,
    get_closest_row,
    get_dataframe_details,
    get_nonzero_min,
    print_row,
    unlist,
)
from tools.dataframes.reshaping import (
    crosstab,
    crosstab_np,
    groupby_value_counts,
    join_df_list,
)
from tools.dataframes.sessions import add_sessions
from tools.io.data import load_xsv, save_xsv


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def test_dataframe():
    """Create test DataFrame for describe tests"""
    return pl.DataFrame(
        {"category": ["A", "A", "A", "B", "B", "C"], "value": [10, 20, 30, 15, 25, 35]}
    )


# --- inspection ---


class TestDfshape:
    def test_dataframe_shape(self, sample_dataframe):
        result = dfshape(sample_dataframe)
        assert result == "(3, 3)"

    def test_series_shape(self, sample_dataframe):
        series = sample_dataframe["name"]
        result = dfshape(series)
        assert result == "(3, 1)"

    def test_invalid_input(self):
        with pytest.raises(
            ValueError, match="input must be a polars DataFrame or Series"
        ):
            # intentionally invalid input to exercise the ValueError path
            dfshape([1, 2, 3])  # type: ignore[bad-argument-type]

    def test_polars_support(self):
        df = pl.DataFrame(
            {
                "name": ["Alice", "Bob", "Charlie"],
                "age": [25, 30, 35],
                "city": ["New York", "London", "Tokyo"],
            }
        )
        assert dfshape(df) == "(3, 3)"
        assert dfshape(df["name"]) == "(3, 1)"


class TestUnlist:
    def test_basic(self):
        assert unlist([[1, 2], [3, 4]]) == [1, 2, 3, 4]

    def test_empty(self):
        assert unlist([]) == []

    def test_single_list(self):
        assert unlist([[1, 2, 3]]) == [1, 2, 3]


class TestGetNonzeroMin:
    def test_polars(self):
        df = pl.DataFrame({"x": [0, 1, 2, 3]})
        assert get_nonzero_min(df, "x") == 1

    def test_all_positive(self):
        df = pl.DataFrame({"x": [3, 1, 4, 1, 5]})
        assert get_nonzero_min(df, "x") == 1


# --- dft ---


class TestNeatN:
    def test_integer(self):
        assert neat_n(1000) == "1,000"

    def test_float(self):
        assert neat_n(1000.0) == "1,000"

    def test_null(self):
        assert neat_n(None) == "-"


class TestNeatP:
    def test_proportion(self):
        assert neat_p(0.5) == "50.0%"

    def test_small(self):
        assert neat_p(0.001) == "0.1%"

    def test_null(self):
        assert neat_p(None) == "-"


class TestPresentationTable:
    def test_n_columns(self):
        df = pl.DataFrame({"n": [1000, 2000], "p": [0.5, 0.25]})
        result = presentation_table(df)
        assert result["n"][0] == "1,000"

    def test_p_columns(self):
        df = pl.DataFrame({"n": [100], "p": [0.1234]})
        result = presentation_table(df)
        assert result["p"][0] == "12.3%"

    def test_explicit_ncols_pcols(self):
        df = pl.DataFrame({"count": [5000], "rate": [0.75]})
        result = presentation_table(df, ncols=["count"], pcols=["rate"])
        assert result["count"][0] == "5,000"
        assert result["rate"][0] == "75.0%"

    def test_mean_columns(self):
        df = pl.DataFrame({"mean_score": [3.14159]})
        result = presentation_table(df)
        assert result["mean_score"][0] == "3.142"

    def test_prefixed_columns(self):
        df = pl.DataFrame({"n_total": [2500], "p_rate": [0.333]})
        result = presentation_table(df)
        assert result["n_total"][0] == "2,500"
        assert result["p_rate"][0] == "33.3%"

    def test_null_renders_as_dash(self):
        # Count/percentage formatters map nulls to '-' (skip_nulls=False).
        df = pl.DataFrame({"n": [None, 5], "p": [None, 0.5]})
        result = presentation_table(df)
        assert result["n"].to_list() == ["-", "5"]
        assert result["p"].to_list() == ["-", "50.0%"]


class TestDFTIntegration:
    def test_dft_with_polars_data(self, temp_dir):
        filepath = os.path.join(temp_dir, "test_dft.csv")
        test_data = pl.DataFrame({"category": ["A", "B", "A", "C", "B", "A"]})
        save_xsv(test_data, filepath, verbose=False)

        df = load_xsv(filepath, verbose=False)
        assert isinstance(df, pl.DataFrame)

        result = dft(df["category"], neat=False)
        assert isinstance(result, pl.DataFrame)
        assert "n" in result.columns
        assert "p" in result.columns
        assert result.shape[0] == 3
        assert result["n"].sum() == 6
        assert abs(float(result["p"].sum()) - 1.0) < 1e-6

        result_neat = dft(df["category"])
        assert isinstance(result_neat, pl.DataFrame)
        assert "n" in result_neat.columns
        assert "p" in result_neat.columns
        assert result_neat.shape[0] == 3
        n_values = sorted(result_neat["n"].to_list())
        assert n_values == ["1", "2", "3"]


# --- describe ---


class TestPolarsDescribe:
    def test_describe_float_column_uses_is_not_nan(self):
        # Float columns count non-NaN values (is_not_nan), excluding NaN entries
        df = pl.DataFrame({"value": [1.5, 2.5, 3.5, float("nan")]})
        result = describe(df, "value")
        assert result["notnull"][0] == 3

    def test_describe_without_grouping(self, test_dataframe):
        result = describe(test_dataframe, "value")

        assert isinstance(result, pl.DataFrame)
        assert result.shape[0] == 1
        assert "column" in result.columns
        assert "count" in result.columns
        assert "mean" in result.columns
        assert "std" in result.columns

        assert result["column"][0] == "value"
        assert result["count"][0] == 6
        assert result["mean"][0] == 22.5

        assert "prop" not in result.columns

    def test_describe_with_grouping(self, test_dataframe):
        result = describe(test_dataframe, "value", group="category")

        assert isinstance(result, pl.DataFrame)
        assert result.shape[0] == 4
        assert "column" in result.columns
        assert "category" in result.columns
        assert "count" in result.columns
        assert "prop" in result.columns
        assert "nn_prop" in result.columns

        columns = result.columns
        count_idx = columns.index("count")
        prop_idx = columns.index("prop")
        assert prop_idx == count_idx + 1

        grouped_rows = result.filter(pl.col("category") != "overall")
        prop_sum = float(grouped_rows["prop"].sum())
        assert abs(prop_sum - 1.0) < 1e-10

        overall_row = result.filter(pl.col("category") == "overall")
        assert overall_row["prop"][0] == 1.0

        category_a = result.filter(pl.col("category") == "A")
        assert category_a["count"][0] == 3
        assert category_a["prop"][0] == 0.5

        category_b = result.filter(pl.col("category") == "B")
        assert category_b["count"][0] == 2
        assert abs(category_b["prop"][0] - 0.333333) < 1e-5

        category_c = result.filter(pl.col("category") == "C")
        assert category_c["count"][0] == 1
        assert abs(category_c["prop"][0] - 0.166667) < 1e-5

    def test_describe_confidence_intervals(self, test_dataframe):
        result = describe(test_dataframe, "value")

        assert "95ci_low" in result.columns
        assert "95ci_high" in result.columns

        result_no_ci = describe(test_dataframe, "value", confidence_intervals=False)
        assert "95ci_low" not in result_no_ci.columns
        assert "95ci_high" not in result_no_ci.columns


# --- dates ---


class TestDftInput:
    def test_returns_polars(self):
        s = pl.Series("cat", ["A", "B", "A", "C", "B", "A"])
        result = dft(s, neat=False)
        assert isinstance(result, pl.DataFrame)
        assert result["n"].sum() == 6
        assert abs(float(result["p"].sum()) - 1.0) < 1e-6

    def test_invalid_input(self):
        with pytest.raises(ValueError, match="input must be a polars Series"):
            # intentionally invalid input to exercise the ValueError path
            dft([1, 2, 3])  # type: ignore[bad-argument-type]


class TestDftPolarsSeries:
    @pytest.mark.parametrize("name", ["proportion", "count"])
    def test_internal_columns_do_not_replace_category_values(self, name):
        result = dft(pl.Series(name, ["a", "a", "b"]), neat=False)
        assert result[name].to_list() == ["a", "b"]

    @pytest.mark.parametrize("name", ["n", "p"])
    def test_output_name_collision_is_rejected(self, name):
        with pytest.raises(ValueError, match="input series name cannot be"):
            dft(pl.Series(name, [1, 1, 2]))

    def test_default_sort_by_frequency(self):
        s = pl.Series("val", [3, 1, 2, 1, 3, 3])
        result = dft(s, neat=False)
        counts = result["n"].to_list()
        assert counts == sorted(counts, reverse=True)

    def test_sort_index(self):
        s = pl.Series("val", [3, 1, 2, 1, 3, 3])
        result = dft(s, sort_index=True, neat=False)
        values = result["val"].to_list()
        assert values == sorted(values)

    def test_dropna_false(self):
        s = pl.Series("val", [1, 2, None, 1])
        result = dft(s, dropna=False, neat=False)
        assert result.shape[0] == 3  # 1, 2, null

    def test_default_column_order(self):
        s = pl.Series("cat", ["a", "b", "a"])
        assert dft(s, neat=False).columns == ["cat", "n", "p"]
        assert dft(s).columns == ["cat", "n", "p"]

    def test_neat_thousands_separator(self):
        s = pl.Series("v", ["x"] * 1234 + ["y"] * 5)
        result = dft(s)  # neat=True by default, sorted by count desc
        assert result["n"][0] == "1,234"

    def test_neat_percentage_formatting(self):
        s = pl.Series("v", ["a", "a", "b", "b"])
        result = dft(s)  # 50% each
        assert result["p"].to_list() == ["50.0%", "50.0%"]

    def test_proportion_values_raw(self):
        s = pl.Series("v", [1, 1, 1, 2])
        result = dft(s, neat=False, sort_index=True)
        assert result["n"].to_list() == [3, 1]
        assert result["p"].to_list() == [0.75, 0.25]


class TestBinnedCol:
    LABELS = ["0", "1", "2-10", "11-100", "101-1000", "1000+"]

    def test_basic(self):
        col = pl.Series([0, 1, 5, 50, 500, 5000])
        result = binned_col(col)
        # Ordered Enum so bins sort in declared order, not lexically
        assert result.dtype == pl.Enum(self.LABELS)
        assert result.to_list() == self.LABELS

    def test_above_top_break_bins_to_last(self):
        # Values beyond the largest break fall into "1000+" (polars uses +inf as
        # the upper bound) rather than silently becoming null, as pd.cut did with
        # a fixed top edge.
        col = pl.Series([1500, 20000, 999999])
        result = binned_col(col)
        assert result.to_list() == ["1000+", "1000+", "1000+"]
        assert result.null_count() == 0


class TestGetBinnedDft:
    def test_basic(self):
        col = pl.Series([0, 1, 5, 50, 500, 5000])
        result = get_binned_dft(col)
        assert isinstance(result, pl.DataFrame)
        assert "n" in result.columns
        assert "p" in result.columns
        # Bins must be in numeric (declared) order, not lexical
        assert result[result.columns[0]].to_list() == [
            "0",
            "1",
            "2-10",
            "11-100",
            "101-1000",
            "1000+",
        ]

    def test_bin_boundaries(self):
        # Right-closed bins (a, b]: each edge value falls into the lower bin.
        col = pl.Series([0, 1, 10, 11, 100, 101, 1000, 1001])
        result = get_binned_dft(col)
        assert result[result.columns[0]].to_list() == [
            "0",
            "1",
            "2-10",
            "11-100",
            "101-1000",
            "1000+",
        ]
        assert result["n"].to_list() == ["1", "1", "1", "2", "2", "1"]


# --- inspection (additional) ---


class TestGetDataframeDetails:
    def test_basic(self, capsys):
        df = pl.DataFrame({"a": [1, 2, None], "b": ["x", "y", "z"]})
        get_dataframe_details(df)
        captured = capsys.readouterr()
        assert "checking dataframe details" in captured.out
        # The details table is printed to stdout (column names + nunique header)
        assert "nunique" in captured.out
        assert "a" in captured.out
        assert "b" in captured.out


class TestPrintRow:
    def test_basic(self, capsys):
        row = {"name": "Alice", "age": 30}
        print_row(row)
        captured = capsys.readouterr()
        assert "Alice" in captured.out
        assert "30" in captured.out

    def test_dataframe_input(self, capsys):
        # a single-row DataFrame is converted to a dict before printing
        df = pl.DataFrame({"name": ["Alice"], "age": [30]})
        print_row(df)
        captured = capsys.readouterr()
        assert "Alice" in captured.out


class TestGetClosestRow:
    def test_unsigned_difference_does_not_wrap(self):
        df = pl.DataFrame({"val": pl.Series([0, 10], dtype=pl.UInt64)})
        assert get_closest_row(df, "val", 1)["val"] == 0

    def test_basic(self):
        df = pl.DataFrame({"val": [10, 20, 30, 40]})
        result = get_closest_row(df, "val", 25)
        assert result["val"] == 20 or result["val"] == 30

    def test_exact_match(self):
        df = pl.DataFrame({"val": [1.0, 2.0, 3.0]})
        result = get_closest_row(df, "val", 2.0)
        assert result["val"] == 2.0


# --- reshaping ---


class TestGroupbyValueCounts:
    def test_null_group_does_not_merge_with_zero(self):
        df = pl.DataFrame({"group": [None, 0, None, 0], "type": ["x", "y", "x", "x"]})
        result = groupby_value_counts(df, "group", "type")
        assert result["group"].to_list() == [None, 0]
        assert result["n_x"].to_list() == [2, 1]
        assert result["n_y"].to_list() == [0, 1]

    def test_basic(self):
        df = pl.DataFrame(
            {
                "group": ["a", "a", "a", "b", "b"],
                "type": ["x", "x", "y", "x", "y"],
            }
        )
        result = groupby_value_counts(df, "group", "type")
        assert isinstance(result, pl.DataFrame)
        # default prefix 'n_' -> single-underscore value-count columns
        assert result.columns == ["group", "n_x", "n_y"]
        assert result["group"].to_list() == ["a", "b"]
        assert result["n_x"].to_list() == [2, 1]
        assert result["n_y"].to_list() == [1, 1]

    def test_no_prefix(self):
        df = pl.DataFrame(
            {
                "group": ["a", "a", "b"],
                "type": ["x", "y", "x"],
            }
        )
        result = groupby_value_counts(df, "group", "type", prefix="")
        assert "x" in result.columns
        assert "y" in result.columns

    def test_multiple_group_keys(self):
        df = pl.DataFrame(
            {
                "g1": ["a", "a", "b"],
                "g2": ["p", "p", "q"],
                "type": ["x", "y", "x"],
            }
        )
        result = groupby_value_counts(df, ["g1", "g2"], "type")
        assert result.columns == ["g1", "g2", "n_x", "n_y"]
        assert result.row(0) == ("a", "p", 1, 1)


class TestCrosstab:
    def test_total_label_collision_is_rejected(self):
        df = pl.DataFrame({"row": ["a"], "col": ["total"]})
        with pytest.raises(ValueError, match="margin label 'total'"):
            crosstab(df, columns="col", index="row")

    @pytest.mark.parametrize("margins", [True, False])
    def test_empty_schema_with_default_format(self, margins):
        df = pl.DataFrame(schema={"row": pl.String, "col": pl.String})
        result = crosstab(df, columns="col", index="row", margins=margins)
        if margins:
            assert result.columns == ["row", "total"]
            assert result["row"].to_list() == ["total"]
            assert result["total"].to_list() == ["0.0%"]
        else:
            assert result.columns == ["row"]
            assert result.height == 0

    def test_basic(self):
        df = pl.DataFrame(
            {
                "row": ["a", "a", "b", "b"],
                "col": ["x", "y", "x", "y"],
            }
        )
        result = crosstab(df, columns="col", index="row")
        assert isinstance(result, pl.DataFrame)
        # index becomes the first column; margins add a "total" row and column
        assert result.columns == ["row", "x", "y", "total"]
        assert result["row"].to_list() == ["a", "b", "total"]
        # default normalize=True, fmt=".1%"
        assert result["x"][0] == "25.0%"
        assert result["total"].to_list()[-1] == "100.0%"

    def test_no_format(self):
        df = pl.DataFrame(
            {
                "row": ["a", "a", "b", "b"],
                "col": ["x", "y", "x", "y"],
            }
        )
        result = crosstab(df, columns="col", index="row", fmt="")
        # Raw values should be floats, not strings
        assert isinstance(result["x"][0], float)
        assert result["x"][0] == 0.25

    def test_requires_columns_and_index(self):
        df = pl.DataFrame({"row": ["a"], "col": ["x"]})
        with pytest.raises(ValueError, match="both `columns` and `index` are required"):
            crosstab(df)

    def test_counts_no_margins(self):
        df = pl.DataFrame(
            {
                "row": ["a", "a", "b", "b"],
                "col": ["x", "y", "x", "y"],
            }
        )
        result = crosstab(
            df, columns="col", index="row", normalize=False, margins=False, fmt=""
        )
        assert result.columns == ["row", "x", "y"]
        assert result["row"].to_list() == ["a", "b"]  # no total row
        assert result.drop("row").to_numpy().tolist() == [[1, 1], [1, 1]]


class TestCrosstabNp:
    def test_with_separator(self):
        df = pl.DataFrame(
            {
                "a": ["x", "x", "y", "y", "x"],
                "b": ["1", "2", "1", "2", "1"],
            }
        )
        result = crosstab_np(df, "a", "b", add_seperator=True)
        assert isinstance(result, pl.DataFrame)
        # Flat N_*/P_* columns with an empty separator column between the blocks
        assert "" in result.columns
        assert "N_x" in result.columns
        assert "P_x" in result.columns
        # separator sits between the N and P blocks
        sep = result.columns.index("")
        assert result.columns.index("N_x") < sep < result.columns.index("P_x")

    def test_without_separator(self):
        df = pl.DataFrame(
            {
                "a": ["x", "x", "y", "y"],
                "b": ["1", "2", "1", "2"],
            }
        )
        result = crosstab_np(df, "a", "b", add_seperator=False)
        assert isinstance(result, pl.DataFrame)
        assert "" not in result.columns
        assert "N_x" in result.columns
        assert "P_x" in result.columns


class TestJoinDfList:
    def test_basic(self):
        df1 = pl.DataFrame({"a": [1, 2]})
        df2 = pl.DataFrame({"b": [3, 4]})
        df3 = pl.DataFrame({"c": [5, 6]})
        result = join_df_list([df1, df2, df3])
        assert result.columns == ["a", "b", "c"]
        assert result.shape == (2, 3)
        assert result["b"].to_list() == [3, 4]

    def test_two_dfs(self):
        df1 = pl.DataFrame({"x": [10]})
        df2 = pl.DataFrame({"y": [20]})
        result = join_df_list([df1, df2])
        assert result["x"][0] == 10
        assert result["y"][0] == 20

    def test_join_on_key(self):
        # on= reduce-joins on a shared key, tolerating differing row order
        df1 = pl.DataFrame({"id": [0, 1], "a": [1, 2]})
        df2 = pl.DataFrame({"id": [1, 0], "b": [4, 3]})
        result = join_df_list([df1, df2], on="id")
        assert result.columns == ["id", "a", "b"]
        assert result.sort("id")["b"].to_list() == [3, 4]


# --- sessions ---


class TestAddSessions:
    @staticmethod
    def _df(times: list[str], col: str = "_pulled", **extra) -> pl.DataFrame:
        return pl.DataFrame({col: times, **extra}).with_columns(
            pl.col(col).str.to_datetime()
        )

    def test_basic(self):
        df = self._df(
            [
                "2024-01-01 10:00:00",
                "2024-01-01 10:00:30",
                "2024-01-01 10:05:00",  # >60s gap = new session
                "2024-01-01 10:05:15",
            ],
            val=[1, 2, 3, 4],
        )
        result = add_sessions(df)
        assert "sesh_id" in result.columns
        assert "sesh_time_diff" in result.columns
        assert "sesh_time_elapsed" in result.columns
        # Should have 2 sessions (gap at 10:05:00 > 60s delimiter)
        assert result["sesh_id"].n_unique() == 2

    def test_fractional_gap_crosses_threshold(self):
        df = self._df(["2024-01-01 10:00:00.000", "2024-01-01 10:01:00.900"])
        result = add_sessions(df, session_delimiter=60)
        assert result["sesh_id"].to_list() == [0, 1]

    def test_single_session(self):
        df = self._df(
            [
                "2024-01-01 10:00:00",
                "2024-01-01 10:00:10",
                "2024-01-01 10:00:20",
            ]
        )
        result = add_sessions(df)
        assert result["sesh_id"].n_unique() == 1

    def test_custom_delimiter(self):
        df = self._df(
            [
                "2024-01-01 10:00:00",
                "2024-01-01 10:00:30",
                "2024-01-01 10:01:30",  # 60s gap
            ]
        )
        # With 30s delimiter, the 60s gap creates a new session
        result = add_sessions(df, session_delimiter=30)
        assert result["sesh_id"].n_unique() == 2

    def test_custom_tvar(self):
        df = self._df(["2024-01-01 10:00:00", "2024-01-01 12:00:00"], col="timestamp")
        result = add_sessions(df, tvar="timestamp")
        assert "sesh_id" in result.columns


# --- dates ---


class TestDecomposeDates:
    def test_basic(self):
        dates = pl.Series(["20230101", "20230615"])
        result = decompose_dates(dates)
        assert "date" in result.columns
        assert "week" in result.columns
        assert "month" in result.columns
        assert "year" in result.columns
        assert result["month"][0] == 1
        assert result["month"][1] == 6


class TestProcessTimestamps:
    def test_drop_tz_false_still_parses_and_converts(self):
        col = pl.Series(["2023-01-01T12:00:00"])
        result = process_timestamps(col, tz_from="UTC", tz_to="UTC", drop_tz=False)
        assert result.dtype == pl.Datetime("us", "UTC")
        assert result.dt.hour()[0] == 12

    def test_utc_to_eastern_from_string(self):
        col = pl.Series(["2023-01-01T12:00:00"])
        result = process_timestamps(col, tz_from="UTC", tz_to="US/Eastern")
        # Noon UTC in January is 07:00 US/Eastern (EST, UTC-5); tz info dropped
        assert result.dtype == pl.Datetime
        assert result[0].hour == 7
        assert result[0].tzinfo is None

    def test_already_datetime_input(self):
        col = pl.Series(["2023-06-01T12:00:00"]).str.to_datetime()
        result = process_timestamps(col, tz_from="UTC", tz_to="US/Eastern")
        # June is EDT (UTC-4), so noon UTC becomes 08:00
        assert result[0].hour == 8

    def test_aware_string_uses_its_own_timezone(self):
        col = pl.Series(["2023-01-01T12:00:00-05:00"])
        result = process_timestamps(col, tz_from="UTC", tz_to="UTC", drop_tz=False)
        assert result.dt.hour()[0] == 17


class TestDisplayTables:
    def test_text_with_titles(self, capsys):
        t1 = pl.DataFrame({"a": [1, 2]})
        t2 = pl.DataFrame({"b": [3, 4]})
        display_tables_text([t1, t2], titles=["First", "Second"])
        out = capsys.readouterr().out
        assert "First" in out
        assert "Second" in out

    def test_text_without_titles(self, capsys):
        display_tables_text([pl.DataFrame({"colname": [1]})])
        assert "colname" in capsys.readouterr().out

    def test_dispatch_uses_text_outside_notebook(self, capsys):
        # not in a notebook -> display_tables routes to the text path
        display_tables([pl.DataFrame({"xcol": [1, 2]})])
        assert "xcol" in capsys.readouterr().out

    def test_dispatch_uses_html_in_notebook(self, monkeypatch):
        # force the notebook branch -> display_tables routes to the HTML path
        monkeypatch.setattr(
            "tools.dataframes.display.in_jupyter_notebook", lambda: True
        )
        display_tables([pl.DataFrame({"xcol": [1, 2]})])

    def test_html_with_titles_runs(self):
        # smoke: builds the side-by-side HTML and displays it without raising
        display_tables_html(
            [pl.DataFrame({"a": [1]}), pl.DataFrame({"b": [2]})],
            titles=["A", "B"],
            ncols=2,
        )

    def test_html_without_titles_runs(self):
        display_tables_html([pl.DataFrame({"a": [1, 2]})])

    def test_set_css_runs(self):
        set_css()
