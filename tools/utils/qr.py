"""Generate a PNG QR code from a URL."""

import argparse
from pathlib import Path
from typing import Any

__all__ = ["make_qr"]


# drawer: a qrcode module-drawer instance (e.g. VerticalBarsDrawer) from the
# optional, untyped `qrcode` extra; typed as Any since the dependency may be absent.
def make_qr(url: str, output: str | Path, drawer: Any = None) -> Path:
    """Generate a styled PNG QR code from a URL and save it to disk.

    Args:
        url: URL (or text) to encode in the QR code
        output: Output PNG file path; parent directories are created as needed
        drawer: Optional qrcode module drawer instance (defaults to
            VerticalBarsDrawer)

    Returns:
        Path to the written PNG file

    Note:
        Requires the optional 'qr' extra: ``uv sync --extra qr``
    """
    try:
        import qrcode
        from qrcode.image.styledpil import StyledPilImage
        from qrcode.image.styles.moduledrawers.pil import VerticalBarsDrawer
    except ImportError as e:
        raise ImportError(
            "make_qr requires the 'qr' extra; install with: uv sync --extra qr"
        ) from e

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    qr = qrcode.QRCode()
    qr.add_data(url)
    img = qr.make_image(
        image_factory=StyledPilImage,
        module_drawer=drawer or VerticalBarsDrawer(),
    )
    img.save(output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="URL to encode in the QR code")
    parser.add_argument("output", help="Output PNG file path")
    args = parser.parse_args()

    output_path = make_qr(args.url, args.output)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
