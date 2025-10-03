#!/usr/bin/env python3
"""make_ohlc_images.py
=================================

Entry-point script for generating deterministic, embedding-ready OHLC candlestick
images from Yahoo Finance data. The heavy lifting lives in the
``ohlc_image_module`` package which keeps the codebase modular and easier to
maintain.

Example usage
-------------

* Full range chart for a daily equity series::

    python make_ohlc_images.py --ticker AAPL --start 2023-01-01 --end 2023-12-31 --interval 1d --out_dir ./out

* Rolling windows for embedding pipelines::

    python make_ohlc_images.py --ticker MSFT --start 2022-01-01 --end 2023-12-31 --interval 1h --window 128 --stride 0 --normalize zscore --img_size 256 --no_volume --out_dir ./out

The script requires the following packages: ``yfinance``, ``pandas``, ``numpy``,
``matplotlib``, ``mplfinance``, ``pillow`` and ``python-dateutil``.
"""
from __future__ import annotations

from ohlc_image_module.cli import main


if __name__ == "__main__":
    main()
