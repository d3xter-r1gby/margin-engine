"""All tunable settings in one place. Import with: from config import CFG"""
from dataclasses import dataclass, field


@dataclass
class Config:
    # --- data ---
    tickers: list = field(default_factory=lambda: [
        "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AMD",
        "NFLX", "JPM", "XOM", "JNJ", "PG", "KO", "BA", "GME", "SPY"])
    market: str = "SPY"
    start: str = "2012-06-01"
    train_end: str = "2020-12-31"   # ML + GARCH fitted on data up to here
    valid_end: str = "2021-12-31"   # ML early-stopping window; test starts after this

    # --- risk settings ---
    horizon: int = 10               # margin period of risk, trading days
    conf: float = 0.99              # ES / VaR confidence level
    ewma_lambda: float = 0.94
    fhs_window: int = 750           # days of history used to build scenarios (~3 years)

    # --- margin add-ons ---
    participation: float = 0.20     # max share of daily volume we could trade when liquidating
    conc_k: float = 2.0             # concentration slope
    hhi_threshold: float = 0.15     # HHI above which the concentration add-on starts
    margin_floor: float = 0.05      # minimum margin as fraction of gross
    flat_margin: float = 0.15       # benchmark: flat margin as fraction of gross


CFG = Config()

# Example books (US dollars notional; positive = long, negative = short)
CONCENTRATED_BOOK = {"NVDA": 400e6, "TSLA": 200e6, "META": 150e6,
                     "SPY": -300e6, "GME": -150e6}
DIVERSIFIED_BOOK = {"AAPL": 60e6, "MSFT": 60e6, "AMZN": 60e6, "GOOGL": 60e6,
                    "JPM": 60e6, "XOM": 60e6, "JNJ": 60e6, "PG": 60e6,
                    "KO": -80e6, "NFLX": -80e6, "BA": -80e6, "SPY": -100e6}
