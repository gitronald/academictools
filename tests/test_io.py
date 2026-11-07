import bz2
import os
import tempfile

import pandas as pd
import polars as pl
import pytest
from matplotlib.figure import Figure

from tools.io.data import (
    FilePath,
    load_and_concat,
    load_and_merge,
    load_dataframe,
    load_feather,
    load_jsonl,
    load_jsonl_chunked,
    load_parquet,
    load_text,
    load_xsv,
    merge_df_list,
    save_feather,
    save_fig,
    save_jsonl,
    save_parquet,
    save_text,
    save_xsv,
)
from tools.io.files import generate_temp_dir, get_size_in_gb, remove_temp_dir


@pytest.fixture
def sample_dataframe():
    """Create a sample polars DataFrame for testing"""
    return pl.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
            "city": ["New York", "London", "Tokyo"],
        }
    )


@pytest.fixture
def sample_polars_dataframe():
    """Create a sample polars DataFrame for testing"""
    return pl.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
            "city": ["New York", "London", "Tokyo"],
        }
    )


@pytest.fixture
def sample_pandas_dataframe():
    """Create a sample pandas DataFrame for testing"""
    return pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
            "city": ["New York", "London", "Tokyo"],
        }
    )


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestFileUtils:
    def test_generate_and_remove_temp_dir(self, temp_dir):
        test_path = os.path.join(temp_dir, "subdir")
        generate_temp_dir(test_path)
        assert os.path.exists(test_path)
        remove_temp_dir(test_path)
        assert not os.path.exists(test_path)

    def test_remove_nonexistent_dir(self, temp_dir):
        # Should not raise
        remove_temp_dir(os.path.join(temp_dir, "nonexistent"))

    def test_get_size_in_gb(self):
        result = get_size_in_gb("hello")
        assert isinstance(result, float)
        assert result > 0


class TestTextIO:
    def test_save_and_load_text(self, temp_dir):
        test_text = "Hello, world!\nThis is a test."
        filepath = os.path.join(temp_dir, "test.txt")

        save_text(test_text, filepath, verbose=False)
        assert os.path.exists(filepath)

        loaded_text = load_text(filepath, verbose=False)
        assert loaded_text == test_text

    def test_save_text_no_overwrite(self, temp_dir):
        filepath = os.path.join(temp_dir, "test.txt")
        save_text("first", filepath, verbose=False)
        with pytest.raises(FileExistsError):
            save_text("second", filepath, overwrite=False, verbose=False)

        loaded_text = load_text(filepath, verbose=False)
        assert loaded_text == "first"

    def test_load_nonexistent_file(self, temp_dir):
        filepath = os.path.join(temp_dir, "nonexistent.txt")
        with pytest.raises(FileNotFoundError):
            load_text(filepath, verbose=False)


class TestCSVIO:
    @pytest.mark.parametrize("suffix", ["csv.bz2", "tsv.bz2"])
    def test_save_bz2_is_compressed(self, sample_dataframe, temp_dir, suffix):
        filepath = os.path.join(temp_dir, f"data.{suffix}")
        save_xsv(sample_dataframe, filepath, verbose=False)
        assert bz2.decompress(open(filepath, "rb").read())

    def test_save_and_load_csv(self, sample_dataframe, temp_dir):
        filepath = os.path.join(temp_dir, "test.csv")

        save_xsv(sample_dataframe, filepath, verbose=False)
        assert os.path.exists(filepath)

        loaded_df = load_xsv(filepath, verbose=False)
        assert isinstance(loaded_df, pl.DataFrame)

        assert loaded_df.shape == sample_dataframe.shape
        assert loaded_df.columns == sample_dataframe.columns

    def test_save_and_load_tsv(self, sample_dataframe, temp_dir):
        filepath = os.path.join(temp_dir, "test.tsv")

        save_xsv(sample_dataframe, filepath, verbose=False)
        assert os.path.exists(filepath)

        loaded_df = load_xsv(filepath, verbose=False)
        assert isinstance(loaded_df, pl.DataFrame)

        assert loaded_df.shape == sample_dataframe.shape
        assert loaded_df.columns == sample_dataframe.columns

    def test_invalid_extension(self, sample_dataframe, temp_dir):
        filepath = os.path.join(temp_dir, "test.txt")

        with pytest.raises(ValueError, match="file must be"):
            save_xsv(sample_dataframe, filepath)


class TestLoadAndConcat:
    def test_concat_two_files(self, sample_dataframe, temp_dir):
        paths: list[FilePath] = []
        for i in range(2):
            filepath = os.path.join(temp_dir, f"part{i}.csv")
            save_xsv(sample_dataframe, filepath, verbose=False)
            paths.append(filepath)

        df = load_and_concat(paths, verbose=False)
        assert isinstance(df, pl.DataFrame)
        assert df.shape == (2 * sample_dataframe.height, sample_dataframe.width)

    def test_concat_with_dtypes(self, sample_dataframe, temp_dir):
        filepath = os.path.join(temp_dir, "part.csv")
        save_xsv(sample_dataframe, filepath, verbose=False)

        # dtypes are forwarded as schema_overrides to pl.read_csv
        df = load_and_concat([filepath], dtypes={"age": pl.Int64}, verbose=False)
        assert df.schema["age"] == pl.Int64


class TestFeatherIO:
    def test_save_and_load_feather(self, sample_dataframe, temp_dir):
        filepath = os.path.join(temp_dir, "test.feather")

        save_feather(sample_dataframe, filepath, verbose=False)
        assert os.path.exists(filepath)

        loaded_df = load_feather(filepath, verbose=False)
        assert isinstance(loaded_df, pl.DataFrame)

        assert loaded_df.shape == sample_dataframe.shape
        assert loaded_df.columns == sample_dataframe.columns


class TestParquetIO:
    def test_save_and_load_parquet(self, sample_dataframe, temp_dir):
        filepath = os.path.join(temp_dir, "test.parquet")

        save_parquet(sample_dataframe, filepath, verbose=False)
        assert os.path.exists(filepath)

        loaded_df = load_parquet(filepath, verbose=False)
        assert isinstance(loaded_df, pl.DataFrame)

        assert loaded_df.shape == sample_dataframe.shape
        assert loaded_df.columns == sample_dataframe.columns


class TestPolarsDataFrames:
    def test_save_polars_dataframe(self, sample_polars_dataframe, temp_dir):
        filepath = os.path.join(temp_dir, "test_polars.csv")

        save_xsv(sample_polars_dataframe, filepath, verbose=False)
        assert os.path.exists(filepath)

        loaded_df = load_xsv(filepath, verbose=False)
        assert isinstance(loaded_df, pl.DataFrame)

        assert loaded_df.shape == sample_polars_dataframe.shape
        assert loaded_df.columns == sample_polars_dataframe.columns


class TestOverwriteBehavior:
    def test_csv_no_overwrite(self, sample_dataframe, temp_dir):
        filepath = os.path.join(temp_dir, "test.csv")

        save_xsv(sample_dataframe, filepath, verbose=False)

        new_df = pl.DataFrame({"col": [1, 2, 3]})
        with pytest.raises(FileExistsError):
            save_xsv(new_df, filepath, overwrite=False, verbose=False)

        loaded_df = load_xsv(filepath, verbose=False)
        assert loaded_df.shape == sample_dataframe.shape
        assert loaded_df.columns == sample_dataframe.columns

    def test_csv_with_overwrite(self, sample_dataframe, temp_dir):
        filepath = os.path.join(temp_dir, "test.csv")

        save_xsv(sample_dataframe, filepath, verbose=False)

        new_df = pl.DataFrame({"col": [1, 2, 3]})
        save_xsv(new_df, filepath, overwrite=True, verbose=False)

        loaded_df = load_xsv(filepath, verbose=False)
        assert loaded_df.shape == new_df.shape
        assert loaded_df.columns == new_df.columns


class TestPolarsOnlySaves:
    def test_save_xsv_polars(self, sample_dataframe, temp_dir):
        filepath = os.path.join(temp_dir, "test.csv")
        save_xsv(sample_dataframe, filepath, verbose=False)
        loaded_df = load_xsv(filepath, verbose=False)
        assert isinstance(loaded_df, pl.DataFrame)
        assert loaded_df.shape == sample_dataframe.shape
        assert loaded_df.columns == sample_dataframe.columns


class TestLoadDataframe:
    def test_dispatch_csv(self, sample_dataframe, temp_dir):
        filepath = os.path.join(temp_dir, "test.csv")
        save_xsv(sample_dataframe, filepath, verbose=False)

        loaded_df = load_dataframe(filepath)
        assert isinstance(loaded_df, pl.DataFrame)
        assert loaded_df.shape == sample_dataframe.shape
        assert loaded_df.columns == sample_dataframe.columns

    def test_dispatch_tsv(self, sample_dataframe, temp_dir):
        filepath = os.path.join(temp_dir, "test.tsv")
        save_xsv(sample_dataframe, filepath, verbose=False)

        loaded_df = load_dataframe(filepath)
        assert loaded_df.shape == sample_dataframe.shape
        assert loaded_df.columns == sample_dataframe.columns

    def test_dispatch_csv_bz2(self, sample_dataframe, temp_dir):
        filepath = os.path.join(temp_dir, "test.csv.bz2")
        save_xsv(sample_dataframe, filepath, verbose=False)

        loaded_df = load_dataframe(filepath)
        assert loaded_df.shape == sample_dataframe.shape
        assert loaded_df.columns == sample_dataframe.columns

    def test_dispatch_tsv_bz2(self, sample_dataframe, temp_dir):
        filepath = os.path.join(temp_dir, "test.tsv.bz2")
        save_xsv(sample_dataframe, filepath, verbose=False)

        loaded_df = load_dataframe(filepath)
        assert loaded_df.shape == sample_dataframe.shape
        assert loaded_df.columns == sample_dataframe.columns

    def test_dispatch_feather(self, sample_dataframe, temp_dir):
        filepath = os.path.join(temp_dir, "test.feather")
        save_feather(sample_dataframe, filepath, verbose=False)

        loaded_df = load_dataframe(filepath)
        assert loaded_df.shape == sample_dataframe.shape
        assert loaded_df.columns == sample_dataframe.columns

    def test_dispatch_parquet(self, sample_dataframe, temp_dir):
        filepath = os.path.join(temp_dir, "test.parquet")
        save_parquet(sample_dataframe, filepath, verbose=False)

        loaded_df = load_dataframe(filepath)
        assert loaded_df.shape == sample_dataframe.shape
        assert loaded_df.columns == sample_dataframe.columns

    def test_no_loader_for_extension(self, temp_dir):
        filepath = os.path.join(temp_dir, "test.txt")
        with pytest.raises(ValueError, match="no loader found"):
            load_dataframe(filepath)


class TestBz2XsvRoundTrip:
    def test_csv_bz2_round_trip(self, sample_dataframe, temp_dir):
        filepath = os.path.join(temp_dir, "test.csv.bz2")
        save_xsv(sample_dataframe, filepath, verbose=False)
        assert os.path.exists(filepath)

        loaded_df = load_xsv(filepath, verbose=False)
        assert isinstance(loaded_df, pl.DataFrame)
        assert loaded_df.equals(sample_dataframe)

    def test_load_legacy_plain_text_bz2(self, temp_dir):
        filepath = os.path.join(temp_dir, "legacy.csv.bz2")
        with open(filepath, "w") as outfile:
            outfile.write("name,value\nalice,1\nbob,2\n")

        loaded_df = load_xsv(filepath, verbose=False)
        assert loaded_df.to_dict(as_series=False) == {
            "name": ["alice", "bob"],
            "value": [1, 2],
        }

    def test_tsv_bz2_round_trip(self, sample_dataframe, temp_dir):
        filepath = os.path.join(temp_dir, "test.tsv.bz2")
        save_xsv(sample_dataframe, filepath, verbose=False)
        assert os.path.exists(filepath)

        loaded_df = load_xsv(filepath, verbose=False)
        assert loaded_df.shape == sample_dataframe.shape
        assert loaded_df.columns == sample_dataframe.columns


class TestXsvEdgeCases:
    def test_load_xsv_missing_file(self, temp_dir):
        filepath = os.path.join(temp_dir, "nonexistent.csv")
        with pytest.raises(FileNotFoundError, match="File not found"):
            load_xsv(filepath, verbose=False)

    def test_load_xsv_display_head(self, sample_dataframe, temp_dir):
        filepath = os.path.join(temp_dir, "test.csv")
        save_xsv(sample_dataframe, filepath, verbose=False)

        loaded_df = load_xsv(filepath, display_head=2, verbose=False)
        assert loaded_df.shape == sample_dataframe.shape

    def test_save_xsv_empty_path(self, sample_dataframe):
        with pytest.raises(ValueError, match="file path must be provided"):
            save_xsv(sample_dataframe, "", verbose=False)


class TestDisplayHead:
    def test_load_feather_display_head(self, sample_dataframe, temp_dir):
        filepath = os.path.join(temp_dir, "test.feather")
        save_feather(sample_dataframe, filepath, verbose=False)

        loaded_df = load_feather(filepath, display_head=2, verbose=False)
        assert loaded_df.shape == sample_dataframe.shape

    def test_load_parquet_display_head(self, sample_dataframe, temp_dir):
        filepath = os.path.join(temp_dir, "test.parquet")
        save_parquet(sample_dataframe, filepath, verbose=False)

        loaded_df = load_parquet(filepath, display_head=2, verbose=False)
        assert loaded_df.shape == sample_dataframe.shape


class TestJSONLIO:
    def test_append_compressed_jsonl_preserves_existing_rows(self, temp_dir):
        filepath = os.path.join(temp_dir, "append.json.bz2")
        save_jsonl([{"a": 1}], filepath, verbose=False)
        save_jsonl([{"a": 2}], filepath, mode="a", overwrite=True, verbose=False)
        assert load_jsonl(filepath, verbose=False) == [{"a": 1}, {"a": 2}]

    def test_save_and_load_jsonl(self, temp_dir):
        data = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        filepath = os.path.join(temp_dir, "test.json")

        save_jsonl(data, filepath, verbose=False)
        assert os.path.exists(filepath)

        loaded = load_jsonl(filepath, verbose=False)
        assert loaded == data

    def test_save_and_load_jsonl_bz2(self, temp_dir):
        data = [{"k": "v"}, {"k": "w"}]
        filepath = os.path.join(temp_dir, "test.json.bz2")

        save_jsonl(data, filepath, verbose=False)
        assert os.path.exists(filepath)

        loaded = load_jsonl(filepath, verbose=False)
        assert loaded == data

    def test_load_jsonl_as_df(self, temp_dir):
        data = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        filepath = os.path.join(temp_dir, "test.json")
        save_jsonl(data, filepath, verbose=False)

        df = load_jsonl(filepath, as_df=True, display_head=1, verbose=False)
        assert isinstance(df, pl.DataFrame)
        assert df.shape == (2, 2)
        assert df.columns == ["a", "b"]

    def test_load_jsonl_nrows(self, temp_dir):
        data = [{"a": 1}, {"a": 2}, {"a": 3}]
        filepath = os.path.join(temp_dir, "test.json")
        save_jsonl(data, filepath, verbose=False)

        loaded = load_jsonl(filepath, nrows=2, verbose=False)
        assert loaded == data[:2]

    def test_load_jsonl_return_errors(self, temp_dir):
        data = [{"a": 1}, {"a": 2}]
        filepath = os.path.join(temp_dir, "test.json")
        save_jsonl(data, filepath, verbose=False)
        # Append a malformed line to trigger the error-collection path.
        with open(filepath, "a") as f:
            f.write("not valid json\n")

        result = load_jsonl(filepath, return_errors=True, verbose=False)
        assert isinstance(result, tuple)
        loaded, errors = result
        assert isinstance(errors, list)
        assert loaded == data
        assert len(errors) == 1
        first_error: dict[str, object] = errors[0]
        assert first_error["idx"] == 2
        assert "line" in first_error
        assert "error" in first_error

    def test_save_jsonl_no_overwrite(self, temp_dir):
        filepath = os.path.join(temp_dir, "test.json")
        save_jsonl([{"a": 1}], filepath, verbose=False)

        with pytest.raises(FileExistsError):
            save_jsonl([{"a": 2}], filepath, overwrite=False, verbose=False)

    def test_jsonl_invalid_extension(self, temp_dir):
        filepath = os.path.join(temp_dir, "test.txt")
        with pytest.raises(ValueError, match="file must be .json"):
            save_jsonl([{"a": 1}], filepath, verbose=False)

    def test_save_and_load_jsonl_verbose(self, temp_dir):
        # Exercise the verbose print branches in save_jsonl/load_jsonl.
        data = [{"a": 1}, {"a": 2}]
        filepath = os.path.join(temp_dir, "test.json")

        save_jsonl(data, filepath, verbose=True)
        loaded = load_jsonl(filepath, verbose=True)
        assert loaded == data


class TestLoadJSONLChunked:
    def test_chunked_even_split(self, temp_dir):
        data = [{"i": i} for i in range(4)]
        filepath = os.path.join(temp_dir, "test.json")
        save_jsonl(data, filepath, verbose=False)

        chunks = list(load_jsonl_chunked(filepath, chunk_size=2))
        assert [len(c) for c in chunks] == [2, 2]
        assert [row for chunk in chunks for row in chunk] == data

    def test_chunked_with_remainder(self, temp_dir):
        data = [{"i": i} for i in range(5)]
        filepath = os.path.join(temp_dir, "test.json")
        save_jsonl(data, filepath, verbose=False)

        chunks = list(load_jsonl_chunked(filepath, chunk_size=2))
        assert [len(c) for c in chunks] == [2, 2, 1]

    def test_chunked_skips_bad_lines(self, temp_dir):
        data = [{"i": 0}, {"i": 1}]
        filepath = os.path.join(temp_dir, "test.json")
        save_jsonl(data, filepath, verbose=False)
        with open(filepath, "a") as f:
            f.write("garbage\n")

        chunks = list(load_jsonl_chunked(filepath, chunk_size=10, verbose=True))
        assert [row for chunk in chunks for row in chunk] == data


class TestLoadAndMerge:
    def test_merge_single_file(self, temp_dir):
        base = pl.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
        extra = pl.DataFrame({"id": [1, 2, 3], "score": [10, 20, 30]})
        filepath = os.path.join(temp_dir, "extra.csv")
        save_xsv(extra, filepath, verbose=False)

        merged = load_and_merge(base, [filepath], how="left", on="id", verbose=False)
        assert merged.shape == (3, 3)
        assert set(merged.columns) == {"id", "name", "score"}
        assert merged.sort("id")["score"].to_list() == [10, 20, 30]

    def test_merge_multiple_files(self, temp_dir):
        base = pl.DataFrame({"id": [1, 2], "name": ["a", "b"]})
        extra1 = pl.DataFrame({"id": [1, 2], "score": [10, 20]})
        extra2 = pl.DataFrame({"id": [1, 2], "rank": [100, 200]})
        fp1 = os.path.join(temp_dir, "extra1.csv")
        fp2 = os.path.join(temp_dir, "extra2.csv")
        save_xsv(extra1, fp1, verbose=False)
        save_xsv(extra2, fp2, verbose=False)

        merged = load_and_merge(base, [fp1, fp2], on="id", verbose=False)
        assert merged.shape == (2, 4)
        assert set(merged.columns) == {"id", "name", "score", "rank"}


class TestMergeDfList:
    def test_cross_join_without_on(self):
        left = pl.DataFrame({"a": [1, 2]})
        right = pl.DataFrame({"b": ["x", "y"]})
        result = merge_df_list([left, right], how="cross")
        assert result.shape == (4, 2)

    def test_merge_df_list_inner(self):
        df1 = pl.DataFrame({"id": [1, 2, 3], "a": [1, 2, 3]})
        df2 = pl.DataFrame({"id": [2, 3, 4], "b": [20, 30, 40]})

        merged = merge_df_list([df1, df2], how="inner", on="id")
        assert merged.shape == (2, 3)
        assert merged.sort("id")["id"].to_list() == [2, 3]

    def test_merge_df_list_left(self):
        df1 = pl.DataFrame({"id": [1, 2], "a": [1, 2]})
        df2 = pl.DataFrame({"id": [1, 2], "b": [10, 20]})
        df3 = pl.DataFrame({"id": [1, 2], "c": [100, 200]})

        merged = merge_df_list([df1, df2, df3], how="left", on="id")
        assert merged.shape == (2, 4)
        assert set(merged.columns) == {"id", "a", "b", "c"}


class TestSaveFig:
    def test_no_overwrite_checks_every_destination(self, temp_dir):
        fig = Figure()
        fig.add_subplot(111).plot([1, 2], [3, 4])
        filepath = os.path.join(temp_dir, "fig.png")
        pdf_path = os.path.join(temp_dir, "fig.pdf")
        with open(pdf_path, "wb") as output:
            output.write(b"existing")
        with pytest.raises(FileExistsError):
            save_fig(fig, filepath, overwrite=False, png=True, pdf=True)
        assert open(pdf_path, "rb").read() == b"existing"

    def test_save_png(self, temp_dir):
        fig = Figure()
        ax = fig.add_subplot(111)
        ax.plot([1, 2, 3], [4, 5, 6])
        filepath = os.path.join(temp_dir, "fig.png")

        save_fig(fig, filepath, png=True)
        assert os.path.exists(filepath)

    def test_save_all_formats(self, temp_dir):
        fig = Figure()
        ax = fig.add_subplot(111)
        ax.plot([1, 2, 3], [4, 5, 6])
        filepath = os.path.join(temp_dir, "fig.png")

        save_fig(fig, filepath, png=True, tiff=True, pdf=True)
        assert os.path.exists(os.path.join(temp_dir, "fig.png"))
        assert os.path.exists(os.path.join(temp_dir, "fig.tiff"))
        assert os.path.exists(os.path.join(temp_dir, "fig.pdf"))

    def test_save_pdf_from_pdf_path(self, temp_dir):
        fig = Figure()
        fig.add_subplot(111).plot([1, 2], [3, 4])
        filepath = os.path.join(temp_dir, "fig.pdf")

        save_fig(fig, filepath, png=False, pdf=True)
        assert os.path.exists(filepath)

    def test_invalid_extension(self, temp_dir):
        fig = Figure()
        filepath = os.path.join(temp_dir, "fig.svg")

        with pytest.raises(ValueError, match="can only produce"):
            save_fig(fig, filepath)

    def test_no_overwrite(self, temp_dir):
        fig = Figure()
        fig.add_subplot(111).plot([1, 2], [3, 4])
        filepath = os.path.join(temp_dir, "fig.png")
        save_fig(fig, filepath, png=True)

        with pytest.raises(FileExistsError):
            save_fig(fig, filepath, overwrite=False, png=True)
