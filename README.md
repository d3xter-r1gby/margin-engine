# Dynamic Margin Engine

Prototype prime-brokerage margin model: ML / GARCH / EWMA volatility forecasts feed a
filtered-historical-simulation Expected Shortfall, with concentration and liquidity add-ons,
backtested against a flat-margin benchmark.

## Run
    pip install -r requirements.txt
    python -m scripts.run_models
    streamlit run app.py

## Results
Data runs to 2026-10-30. Forecasts are scored from 2022-01-03.

| Model | RMSE (10d vol) | QLIKE |
|---|---|---|
| EWMA | 0.0410 | 0.3990 |
| GARCH | 0.0428 | 0.3718 |
| LightGBM | 0.0396 | 0.3608 |

The backtest uses EWMA from 2016-01-04, with 2,713 overlapping 10-day windows. Margin percentages
are of gross, and Kupiec p tests 99% VaR breaches on 272 non-overlapping windows.

| Book | Flat margin avg % | Dynamic avg % | Flat breaches | Dynamic breaches | Kupiec p |
|---|---|---|---|---|---|
| Concentrated | 15.0 | 63.3 (median 13.8) | 25 | 13 | 0.229 |
| Diversified | 15.0 | 5.9 (median 5.0) | 0 | 0 | 0.645 |

LightGBM won on QLIKE overall (0.361, against 0.372 for GARCH and 0.399 for EWMA), and its lead was
widest in the high-vol regime, the 20% of test days with the highest SPY realised vol (0.333, against
0.385 and 0.401).
On the diversified book, dynamic margin averaged 5.9% of gross against 15% flat, with no breaches
for either; on the concentrated book it cut breaches from 25 to 13 at a median of 13.8%, but its
63.3% average is inflated by the days after the GME squeeze, when it peaked at 11,483% of gross.
The biggest limitation is that margin only reacts after a move and the backtest holds the book
fixed, so neither approach covered the January 2021 GME squeeze: the $150M short lost $2.46bn in 10
days on a $1.2bn book, leaving a shortfall of about $2.3bn under both flat and dynamic margin.

## Limitations
Historical correlations, daily data, no earnings gap modelling, synthetic books, simple liquidity model.
