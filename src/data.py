"""Data loading and basic transforms.

Conventions used across the project:
  * DataFrames are indexed by date (rows) with one column per ticker.
  * Returns are daily log returns, in decimals (0.01 = 1%).
  * Variances are daily unless a name says otherwise (e.g. var_h = h-day variance).
"""
from pathlib import Path
import numpy as np
import pandas as pd

CACHE = Path(__file__).resolve().parents[1] / "data"


def download_prices(tickers: list[str], start: str, end: str | None = None
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download adjusted close prices and volumes from Yahoo Finance.

    Returns (close, volume), indexed by date with one column per ticker. Dates with no price
    for any ticker are dropped, price gaps of up to 2 days are forward-filled, and missing
    volume is set to 0.
    """
    import yfinance as yf
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    close, volume = raw["Close"][tickers], raw["Volume"][tickers]
    keep = close.notna().any(axis=1)
    return close[keep].ffill(limit=2), volume[keep].fillna(0)
    


def load_or_download(tickers: list[str], start: str, refresh: bool = False
                     ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (close, volume) from the pickles in CACHE, downloading them if needed.

    Data is downloaded and saved if either file is missing or `refresh` is True. The cache
    isn't keyed on `tickers` or `start`, so pass refresh=True after changing either.
    """
    close_path, volume_path = CACHE / "close.pkl", CACHE / "volume.pkl"
    if close_path.exists() and volume_path.exists() and not refresh:
        return pd.read_pickle(close_path), pd.read_pickle(volume_path)
    close, volume = download_prices(tickers, start)
    close.to_pickle(close_path)
    volume.to_pickle(volume_path)
    return close, volume


def log_returns(close: pd.DataFrame) -> pd.DataFrame:
    """Return daily log returns, ln(P_t / P_{t-1}), without the empty first row."""
    r_t = (close / close.shift(1)).apply(np.log).iloc[1:]
    return r_t


def realised_vol(returns: pd.DataFrame, window: int, annualise: bool = True) -> pd.DataFrame:
    """Return trailing realised volatility: sqrt of the mean of r^2 over `window` days.

    The window includes day t. Multiplied by sqrt(252) if `annualise` is True.
    """
    realised_vol = (returns**2).rolling(window=window).mean().apply(np.sqrt)
    return realised_vol*np.sqrt(252) if annualise else realised_vol

def forward_realised_var(returns: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Return realised variance over the next h days: sum_{k=1..h} r_{t+k}^2 at date t.

    This uses future returns, so it is a forecasting target and never a feature. The last
    `horizon` rows are NaN.
    """
    forward_realised_var = (returns**2).rolling(window=horizon).sum().shift(-horizon)
    return forward_realised_var

def average_dollar_volume(close: pd.DataFrame, volume: pd.DataFrame, window: int = 20
                          ) -> pd.DataFrame:
    """Return average dollar volume: the rolling mean of close * volume over `window` days."""
    adv = (close*volume).rolling(window).mean()
    return adv

