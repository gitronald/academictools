import sys
import time
from datetime import timedelta
from pathlib import Path

import pytest

from tools.utils.jupyter import in_jupyter_notebook, stop_notebook
from tools.utils.printing import (
    get_script_header,
    get_script_path,
    print_green,
    print_line,
    print_red,
    wraprint,
)
from tools.utils.text import clean_text, remove_mentions, remove_urls
from tools.utils.timing import (
    get_printable_timedelta,
    print_time,
    print_time_elapsed,
    seconds_as_clock,
    stopwatch,
)


class TestSecondsAsClock:
    def test_zero(self):
        assert seconds_as_clock(0) == "00:00:00.0"

    def test_one_hour(self):
        assert seconds_as_clock(3600) == "01:00:00.0"

    def test_fractional(self):
        result = seconds_as_clock(61.5)
        assert result == "00:01:01.5"

    def test_precision_rounds_before_splitting(self):
        assert seconds_as_clock(61.25, precision=2) == "00:01:01.25"
        assert seconds_as_clock(59.99) == "00:01:00.0"
        assert seconds_as_clock(61.9, precision=0) == "00:01:02"
        assert seconds_as_clock(1.23456, precision=3) == "00:00:01.235"


class TestGetPrintableTimedelta:
    def test_basic(self):
        td = timedelta(seconds=90.5)
        result = get_printable_timedelta(td)
        assert "00:01:30" in result


class TestPrintTime:
    def test_basic(self, capsys):
        start = time.time()
        print_time("test", start)
        captured = capsys.readouterr()
        assert "test:" in captured.out
        assert "seconds" in captured.out


class TestPrintTimeElapsed:
    def test_basic(self, capsys):
        start = time.time()
        print_time_elapsed(start)
        captured = capsys.readouterr()
        assert "time elapsed:" in captured.out

    def test_newline(self, capsys):
        start = time.time()
        print_time_elapsed(start, newline=True)
        captured = capsys.readouterr()
        assert captured.out.endswith("\n\n")

    def test_custom_prefix(self, capsys):
        start = time.time()
        print_time_elapsed(start, prefix="done")
        captured = capsys.readouterr()
        assert "done:" in captured.out


class TestStopwatch:
    def test_basic(self, capsys):
        @stopwatch("timer")
        def quick_fn():
            return 42

        result = quick_fn()
        assert result == 42
        captured = capsys.readouterr()
        assert "timer:" in captured.out


class TestRemoveMentions:
    def test_single_mention(self):
        assert remove_mentions("hello @user world") == "hello world"

    def test_multiple_mentions(self):
        result = remove_mentions("@foo @bar text")
        assert "@" not in result
        assert "text" in result

    def test_no_mentions(self):
        assert remove_mentions("no mentions here") == "no mentions here"

    def test_mention_with_colon(self):
        result = remove_mentions("@user: hello")
        assert "@" not in result
        assert "hello" in result


class TestRemoveUrls:
    def test_http_url(self):
        result = remove_urls("visit http://example.com today")
        assert "http" not in result
        assert "today" in result

    def test_https_url(self):
        result = remove_urls("see https://example.com/path")
        assert "https" not in result

    def test_no_urls(self):
        assert remove_urls("no urls here") == "no urls here"


class TestCleanText:
    def test_removes_mentions_and_urls(self):
        import polars as pl

        text = pl.Series(["@user check http://example.com out"])
        result = clean_text(text)
        assert "@" not in result[0]
        assert "http" not in result[0]
        assert "check" in result[0]


class TestWraprint:
    def test_return_str(self):
        result = wraprint("hello world", width=5, return_str=True)
        assert isinstance(result, str)
        assert "\n" in result

    def test_print(self, capsys):
        wraprint("hello world", width=5)
        captured = capsys.readouterr()
        assert "hello" in captured.out


class TestPrintLine:
    def test_basic(self, capsys):
        print_line()
        captured = capsys.readouterr()
        assert len(captured.out.strip()) == 80

    def test_with_word(self, capsys):
        print_line("test")
        captured = capsys.readouterr()
        assert captured.out.startswith("test")

    def test_newline(self, capsys):
        print_line("test", newline=True)
        captured = capsys.readouterr()
        assert captured.out.endswith("\n\n")


class TestPrintGreen:
    def test_basic(self, capsys):
        print_green("success")
        captured = capsys.readouterr()
        assert "success" in captured.out


class TestPrintRed:
    def test_basic(self, capsys):
        print_red("error")
        captured = capsys.readouterr()
        assert "error" in captured.out


class TestGetScriptPath:
    def test_returns_string(self):
        result = get_script_path()
        assert isinstance(result, str)


class TestGetScriptHeader:
    def test_returns_string_with_path(self):
        header = get_script_header()
        assert isinstance(header, str)
        assert get_script_path() in header

    def test_lines_wrap_the_path(self):
        header = get_script_header(lines=True, line_char="=")
        # lines=True frames the path with separator lines above and below
        assert header.count("\n") == 2
        assert "=" in header

    def test_no_lines_is_single_line(self):
        header = get_script_header(lines=False)
        assert "\n" not in header

    def test_arrow_style_end_appends_trailing_arrow(self):
        with_end = get_script_header(arrow_style_end=True)
        without_end = get_script_header(arrow_style_end=False)
        assert len(with_end) > len(without_end)


class TestInJupyterNotebook:
    def test_false_outside_notebook(self):
        # pytest runs in a plain interpreter, not an IPython kernel
        assert in_jupyter_notebook() is False


class TestStopNotebook:
    def test_raises_system_exit(self):
        with pytest.raises(SystemExit, match="stop here"):
            stop_notebook()

    def test_custom_message(self):
        with pytest.raises(SystemExit, match="done"):
            stop_notebook("done")


class TestQrMain:
    def test_cli_writes_png(self, temp_dir, monkeypatch, capsys):
        pytest.importorskip("qrcode")
        from tools.utils.qr import main

        out = str(Path(temp_dir) / "cli.png")
        monkeypatch.setattr(sys, "argv", ["make_qr", "https://example.com", out])
        main()

        assert Path(out).exists()
        assert "Wrote" in capsys.readouterr().out


class TestMakeQr:
    URL = "https://ronalderobertson.com"

    def test_creates_valid_png(self, temp_dir):
        pytest.importorskip("qrcode")
        from PIL import Image

        from tools.utils import make_qr

        out = make_qr(self.URL, Path(temp_dir) / "qr.png")
        assert out.exists()
        with Image.open(out) as img:
            assert img.format == "PNG"
            assert img.width > 0 and img.height > 0

    def test_creates_parent_dirs(self, temp_dir):
        pytest.importorskip("qrcode")
        from tools.utils import make_qr

        out = make_qr(self.URL, Path(temp_dir) / "nested" / "deep" / "qr.png")
        assert out.exists()

    def test_deterministic(self, temp_dir):
        # Same URL renders byte-identically within an environment
        pytest.importorskip("qrcode")
        from tools.utils import make_qr

        a = make_qr(self.URL, Path(temp_dir) / "a.png")
        b = make_qr(self.URL, Path(temp_dir) / "b.png")
        assert a.read_bytes() == b.read_bytes()

    def test_different_url_differs(self, temp_dir):
        # Distinct content must produce distinct output
        pytest.importorskip("qrcode")
        from tools.utils import make_qr

        a = make_qr(self.URL, Path(temp_dir) / "a.png")
        b = make_qr("https://example.com", Path(temp_dir) / "b.png")
        assert a.read_bytes() != b.read_bytes()
