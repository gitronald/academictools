import os
import tempfile

import duckdb
import polars as pl
import pytest

from tools.io import sql


@pytest.fixture
def sample_dataframe():
    """Create a sample polars DataFrame for testing"""
    return pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "age": [25, 30, 35, 40, 45],
            "city": ["New York", "London", "Tokyo", "Paris", "Berlin"],
        }
    )


@pytest.fixture
def sample_serp_dataframe():
    """Create a sample SERP metadata DataFrame for testing"""
    return pl.DataFrame(
        {
            "qry": ["test query", "another query"],
            "loc": ["US", "UK"],
            "url": ["http://example.com/1", "http://example.com/2"],
            "timestamp": ["2024-01-01 12:00:00", "2024-01-02 13:00:00"],
            "serp_id": ["serp_001", "serp_002"],
            "crawl_id": ["crawl_001", "crawl_002"],
            "response_code": [200, 200],
        }
    )


@pytest.fixture
def sample_serp_full_dataframe():
    """Create a sample full SERP DataFrame with HTML for testing"""
    return pl.DataFrame(
        {
            "qry": ["test query"],
            "loc": ["US"],
            "url": ["http://example.com/1"],
            "html": ["<html><body>Test HTML</body></html>"],
            "headers": ['{"Content-Type": "text/html"}'],
            "timestamp": ["2024-01-01 12:00:00"],
            "serp_id": ["serp_001"],
            "crawl_id": ["crawl_001"],
            "response_code": [200],
        }
    )


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def temp_parquet_file(sample_dataframe, temp_dir):
    """Create a temporary parquet file for testing"""
    filepath = os.path.join(temp_dir, "test_data.parquet")
    sample_dataframe.write_parquet(filepath)
    return filepath


@pytest.fixture
def temp_parquet_files(sample_dataframe, temp_dir):
    """Create multiple temporary parquet files for testing"""
    filepaths = []
    for i in range(3):
        filepath = os.path.join(temp_dir, f"test_data_{i}.parquet")
        # Create slightly different data for each file
        df = sample_dataframe.with_columns(pl.col("id") + (i * 10))
        df.write_parquet(filepath)
        filepaths.append(filepath)
    return filepaths


@pytest.fixture
def temp_duckdb(temp_dir):
    """Create a temporary DuckDB database file"""
    filepath = os.path.join(temp_dir, "test.duckdb")
    return filepath


class TestCheckDuckDBSettings:
    """Test check_duckdb_settings function"""

    def test_check_settings_runs(self, capsys):
        """Test that check_duckdb_settings runs without error"""
        # This test just verifies the function runs
        # We can't easily test the display output, but we can ensure no errors
        try:
            sql.check_duckdb_settings()
        except Exception as e:
            pytest.fail(f"check_duckdb_settings raised {e}")


class TestCreateParquetView:
    """Test create_parquet_view function"""

    def test_create_view_with_file_path(self, temp_parquet_file, temp_duckdb):
        """Test creating a view using a database file path"""
        sql.create_parquet_view(
            fp_parquet=temp_parquet_file,
            fp_duckdb=temp_duckdb,
            view_name="test_view",
            verbose=False,
        )

        # Verify the view was created
        con = duckdb.connect(temp_duckdb)
        result = con.sql("SELECT COUNT(*) FROM test_view").fetchone()
        assert result is not None
        assert result[0] == 5
        con.close()

    def test_create_view_with_connection(self, temp_parquet_file, temp_duckdb):
        """Test creating a view using an existing connection"""
        con = duckdb.connect(temp_duckdb)

        sql.create_parquet_view(
            fp_parquet=temp_parquet_file,
            duckdb_con=con,
            view_name="test_view",
            verbose=False,
        )

        # Verify the view was created
        result = con.sql("SELECT COUNT(*) FROM test_view").fetchone()
        assert result is not None
        assert result[0] == 5
        con.close()

    def test_create_view_custom_name(self, temp_parquet_file, temp_duckdb):
        """Test creating a view with a custom name"""
        sql.create_parquet_view(
            fp_parquet=temp_parquet_file,
            fp_duckdb=temp_duckdb,
            view_name="custom_view_name",
            verbose=False,
        )

        # Verify the view exists with custom name
        con = duckdb.connect(temp_duckdb)
        result = con.sql("SELECT COUNT(*) FROM custom_view_name").fetchone()
        assert result is not None
        assert result[0] == 5
        con.close()

    def test_create_view_replaces_existing(self, temp_parquet_file, temp_duckdb):
        """Test that creating a view with same name replaces the old one"""
        # Create first view
        sql.create_parquet_view(
            fp_parquet=temp_parquet_file,
            fp_duckdb=temp_duckdb,
            view_name="test_view",
            verbose=False,
        )

        # Create second view with same name (should replace)
        sql.create_parquet_view(
            fp_parquet=temp_parquet_file,
            fp_duckdb=temp_duckdb,
            view_name="test_view",
            verbose=False,
        )

        # Verify view still works
        con = duckdb.connect(temp_duckdb)
        result = con.sql("SELECT COUNT(*) FROM test_view").fetchone()
        assert result is not None
        assert result[0] == 5
        con.close()

    def test_verbose_output(self, temp_parquet_file, temp_duckdb, capsys):
        """Verbose mode prints the CREATE VIEW query"""
        sql.create_parquet_view(
            fp_parquet=temp_parquet_file,
            fp_duckdb=temp_duckdb,
            view_name="test_view",
            verbose=True,
        )
        assert "CREATE OR REPLACE VIEW" in capsys.readouterr().out

    def test_create_view_invalid_args(self, temp_parquet_file):
        """Test that providing both or neither arguments raises ValueError"""
        # Test with both arguments
        with pytest.raises(
            ValueError, match="Must provide either duckdb_con or fp_duckdb"
        ):
            sql.create_parquet_view(
                fp_parquet=temp_parquet_file,
                fp_duckdb="test.duckdb",
                duckdb_con=duckdb.connect(":memory:"),
            )

        # Test with neither argument
        with pytest.raises(
            ValueError, match="Must provide either duckdb_con or fp_duckdb"
        ):
            sql.create_parquet_view(fp_parquet=temp_parquet_file)


class TestCreateParquetsView:
    """Test create_parquets_view function"""

    def test_create_view_from_multiple_files(self, temp_parquet_files, temp_duckdb):
        """Test creating a view from multiple parquet files"""
        sql.create_parquets_view(
            fps_parquets=temp_parquet_files,
            fp_duckdb=temp_duckdb,
            view_name="combined_view",
            verbose=False,
        )

        # Verify the view contains data from all files
        con = duckdb.connect(temp_duckdb)
        result = con.sql("SELECT COUNT(*) FROM combined_view").fetchone()
        assert result is not None
        assert result[0] == 15  # 3 files * 5 rows each
        con.close()

    def test_create_view_with_connection(self, temp_parquet_files, temp_duckdb):
        """Test creating a view using an existing connection"""
        con = duckdb.connect(temp_duckdb)

        sql.create_parquets_view(
            fps_parquets=temp_parquet_files,
            duckdb_con=con,
            view_name="combined_view",
            verbose=False,
        )

        # Verify the view was created
        result = con.sql("SELECT COUNT(*) FROM combined_view").fetchone()
        assert result is not None
        assert result[0] == 15
        con.close()

    def test_create_view_single_file(self, temp_parquet_file, temp_duckdb):
        """Test creating a view from a single-element list"""
        sql.create_parquets_view(
            fps_parquets=[temp_parquet_file],
            fp_duckdb=temp_duckdb,
            view_name="single_view",
            verbose=False,
        )

        # Verify the view was created
        con = duckdb.connect(temp_duckdb)
        result = con.sql("SELECT COUNT(*) FROM single_view").fetchone()
        assert result is not None
        assert result[0] == 5
        con.close()

    def test_verbose_output(self, temp_parquet_files, temp_duckdb, capsys):
        """Verbose mode prints the CREATE VIEW query"""
        sql.create_parquets_view(
            fps_parquets=temp_parquet_files,
            fp_duckdb=temp_duckdb,
            view_name="combined_view",
            verbose=True,
        )
        assert "CREATE OR REPLACE VIEW" in capsys.readouterr().out

    def test_create_view_invalid_args(self, temp_parquet_files):
        """Test that providing both or neither arguments raises ValueError"""
        # Test with both arguments
        with pytest.raises(
            ValueError, match="Must provide either duckdb_con or fp_duckdb"
        ):
            sql.create_parquets_view(
                fps_parquets=temp_parquet_files,
                fp_duckdb="test.duckdb",
                duckdb_con=duckdb.connect(":memory:"),
            )

        # Test with neither argument
        with pytest.raises(
            ValueError, match="Must provide either duckdb_con or fp_duckdb"
        ):
            sql.create_parquets_view(fps_parquets=temp_parquet_files)


class TestDropView:
    """Test drop_view function"""

    def test_drop_existing_view(self, temp_parquet_file, temp_duckdb, capsys):
        """Test dropping an existing view"""
        # Create a view first
        sql.create_parquet_view(
            fp_parquet=temp_parquet_file, fp_duckdb=temp_duckdb, view_name="test_view"
        )

        # Drop the view
        con = duckdb.connect(temp_duckdb)
        sql.drop_view(con, "test_view")
        con.close()

    def test_drop_nonexistent_view(self, temp_duckdb):
        """Test dropping a view that doesn't exist (should do nothing)"""
        con = duckdb.connect(temp_duckdb)
        sql.drop_view(con, "nonexistent_view")
        con.close()


class TestGetColumns:
    """Test get_columns function"""

    def test_get_columns_from_parquet(self, temp_parquet_file):
        """Test getting columns from a parquet file"""
        cols = sql.get_columns(fp_parquet=temp_parquet_file)

        assert isinstance(cols, pl.DataFrame)
        assert list(cols.columns) == ["id", "name", "age", "city"]
        assert len(cols) == 0  # Should be empty (LIMIT 0)

    def test_get_columns_from_view(self, temp_parquet_file, temp_duckdb):
        """Test getting columns from a DuckDB view"""
        # Create a view first
        sql.create_parquet_view(
            fp_parquet=temp_parquet_file, fp_duckdb=temp_duckdb, view_name="test_view"
        )

        # Get columns
        con = duckdb.connect(temp_duckdb)
        cols = sql.get_columns(duckdb_con=con, view_name="test_view")
        con.close()

        assert isinstance(cols, pl.DataFrame)
        assert "column_name" in cols.columns
        assert len(cols) == 4  # 4 columns in sample data

    def test_get_columns_invalid_args(self):
        """Test that providing invalid arguments raises ValueError"""
        with pytest.raises(
            ValueError, match="Must provide either duckdb_con or fp_parquet"
        ):
            sql.get_columns()


class TestRowCount:
    """Test row_count function"""

    def test_row_count_all_rows(self, temp_parquet_file, temp_duckdb):
        """Test counting all rows using *"""
        # Create a view first
        sql.create_parquet_view(
            fp_parquet=temp_parquet_file, fp_duckdb=temp_duckdb, view_name="test_view"
        )

        # Count rows
        con = duckdb.connect(temp_duckdb)
        count = sql.row_count(con, "*", "test_view")
        con.close()

        assert count == 5

    def test_row_count_specific_column(self, temp_parquet_file, temp_duckdb):
        """Test counting a specific column"""
        # Create a view first
        sql.create_parquet_view(
            fp_parquet=temp_parquet_file, fp_duckdb=temp_duckdb, view_name="test_view"
        )

        # Count rows
        con = duckdb.connect(temp_duckdb)
        count = sql.row_count(con, "name", "test_view")
        con.close()

        assert count == 5


class TestLoadSearchMetadata:
    """Test load_search_metadata function"""

    def test_load_metadata_basic(self, sample_serp_dataframe, temp_dir, capsys):
        """Test loading search metadata"""
        # Create a parquet file with SERP metadata
        parquet_path = os.path.join(temp_dir, "serp_metadata.parquet")
        sample_serp_dataframe.write_parquet(parquet_path)

        # Create a view
        db_path = os.path.join(temp_dir, "test.duckdb")
        sql.create_parquet_view(
            fp_parquet=parquet_path, fp_duckdb=db_path, view_name="serp_metadata"
        )

        # Load metadata
        con = duckdb.connect(db_path)
        metadata = sql.load_search_metadata(con, "serp_metadata")
        con.close()

        # Verify results
        assert isinstance(metadata, pl.DataFrame)
        assert metadata.shape[0] == 2  # 2 rows
        assert "qry" in metadata.columns
        assert "serp_id" in metadata.columns

        # Verify output
        captured = capsys.readouterr()
        assert "SELECT" in captured.out
        assert "loaded: serp_metadata" in captured.out

    def test_load_metadata_with_display(
        self, sample_serp_dataframe, temp_dir, monkeypatch
    ):
        """Test loading metadata with display_head parameter"""
        # Create a parquet file with SERP metadata
        parquet_path = os.path.join(temp_dir, "serp_metadata.parquet")
        sample_serp_dataframe.write_parquet(parquet_path)

        # Create a view
        db_path = os.path.join(temp_dir, "test.duckdb")
        sql.create_parquet_view(
            fp_parquet=parquet_path, fp_duckdb=db_path, view_name="serp_metadata"
        )

        displayed = []
        monkeypatch.setattr(sql, "display", displayed.append)

        # Load metadata with display
        con = duckdb.connect(db_path)
        try:
            metadata = sql.load_search_metadata(con, "serp_metadata", display_head=1)
        finally:
            con.close()
        assert metadata.shape[0] == 2
        assert metadata["qry"].to_list() == ["test query", "another query"]
        assert len(displayed) == 1
        assert displayed[0].height == 1


class TestLoadSerp:
    """Test load_serp function"""

    def test_load_serp_basic(self, sample_serp_full_dataframe, temp_dir):
        """Test loading a single SERP"""
        # Create a parquet file with full SERP data
        parquet_path = os.path.join(temp_dir, "serp_full.parquet")
        sample_serp_full_dataframe.write_parquet(parquet_path)

        # Create a view
        db_path = os.path.join(temp_dir, "test.duckdb")
        sql.create_parquet_view(
            fp_parquet=parquet_path, fp_duckdb=db_path, view_name="serps"
        )

        # Load SERP
        con = duckdb.connect(db_path)
        serp = sql.load_serp(con, view_name="serps", serp_id="serp_001")
        con.close()

        # Verify results
        assert serp is not None
        assert isinstance(serp, dict)
        assert serp["qry"] == "test query"
        assert serp["serp_id"] == "serp_001"

    def test_load_serp_specific_columns(self, sample_serp_full_dataframe, temp_dir):
        """Test loading a SERP with specific columns"""
        # Create a parquet file with full SERP data
        parquet_path = os.path.join(temp_dir, "serp_full.parquet")
        sample_serp_full_dataframe.write_parquet(parquet_path)

        # Create a view
        db_path = os.path.join(temp_dir, "test.duckdb")
        sql.create_parquet_view(
            fp_parquet=parquet_path, fp_duckdb=db_path, view_name="serps"
        )

        # Load SERP with specific columns
        con = duckdb.connect(db_path)
        serp = sql.load_serp(
            con, view_name="serps", serp_id="serp_001", cols=["qry", "html"]
        )
        con.close()

        # Verify results
        assert serp is not None
        assert "qry" in serp
        assert "html" in serp

    def test_load_serp_nonexistent(self, sample_serp_full_dataframe, temp_dir):
        """Test loading a SERP that doesn't exist"""
        # Create a parquet file with full SERP data
        parquet_path = os.path.join(temp_dir, "serp_full.parquet")
        sample_serp_full_dataframe.write_parquet(parquet_path)

        # Create a view
        db_path = os.path.join(temp_dir, "test.duckdb")
        sql.create_parquet_view(
            fp_parquet=parquet_path, fp_duckdb=db_path, view_name="serps"
        )

        # Load nonexistent SERP
        con = duckdb.connect(db_path)
        serp = sql.load_serp(con, view_name="serps", serp_id="nonexistent")
        con.close()

        # Should return None
        assert serp is None

    def test_load_serp_no_id_raises(self, temp_duckdb):
        """Test that not providing serp_id raises ValueError"""
        con = duckdb.connect(temp_duckdb)

        with pytest.raises(ValueError, match="Must provide serp_id"):
            sql.load_serp(con, view_name="serps")

        con.close()

    def test_load_serp_verbose(self, sample_serp_full_dataframe, temp_dir, capsys):
        """Test loading a SERP with verbose output"""
        # Create a parquet file with full SERP data
        parquet_path = os.path.join(temp_dir, "serp_full.parquet")
        sample_serp_full_dataframe.write_parquet(parquet_path)

        # Create a view
        db_path = os.path.join(temp_dir, "test.duckdb")
        sql.create_parquet_view(
            fp_parquet=parquet_path, fp_duckdb=db_path, view_name="serps"
        )

        # Load SERP with verbose
        con = duckdb.connect(db_path)
        sql.load_serp(con, view_name="serps", serp_id="serp_001", verbose=True)
        con.close()

        # Verify query was printed
        captured = capsys.readouterr()
        assert "SELECT" in captured.out
        assert 'WHERE "serp_id" = ?' in captured.out


class TestBackwardsCompatibility:
    """Test backwards compatibility with existing code"""

    def test_create_parquet_view_old_style(self, temp_parquet_file, temp_duckdb):
        """Test that old-style function calls still work"""
        # Old style: positional arguments
        sql.create_parquet_view(temp_parquet_file, temp_duckdb, "test_view")

        # Verify it worked
        con = duckdb.connect(temp_duckdb)
        result = con.sql("SELECT COUNT(*) FROM test_view").fetchone()
        assert result is not None
        assert result[0] == 5
        con.close()

    def test_get_columns_old_style(self, temp_parquet_file):
        """Test that old-style get_columns calls still work"""
        # Old style: positional argument
        cols = sql.get_columns(None, None, temp_parquet_file)
        assert isinstance(cols, pl.DataFrame)
