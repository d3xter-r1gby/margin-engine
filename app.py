"""Streamlit dashboard for the margin engine. Run with: streamlit run app.py"""
from dataclasses import replace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import CFG, CONCENTRATED_BOOK, DIVERSIFIED_BOOK
from src import margin

st.set_page_config(page_title="Dynamic Margin Engine", layout="wide")

# Colours follow the entity, so a model keeps its colour on every chart
DARK = st.context.theme.type == "dark"
SERIES = {"EWMA": "#3987e5", "GARCH": "#d95926", "LightGBM": "#199e70"} if DARK else \
         {"EWMA": "#2a78d6", "GARCH": "#eb6834", "LightGBM": "#1baf7a"}
INK = "#c3c2b7" if DARK else "#52514e"
GRID = "#2c2c2a" if DARK else "#e1e0d9"
CRITICAL = "#d03b3b"


@st.cache_data
def load():
    return pd.read_pickle("data/artifacts.pkl")


def dollars_m(x: float) -> str:
    """Dollar amount in $m, with the sign in front: -$50.6m."""
    return f"{'-' if x < 0 else ''}${abs(x) / 1e6:,.1f}m"


def style(fig: go.Figure, y_title: str) -> go.Figure:
    """Shared chart look: recessive grid, legend on top, unified hover."""
    fig.update_layout(margin=dict(l=0, r=0, t=32, b=0), hovermode="x unified",
                      legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(title=y_title, gridcolor=GRID, zerolinecolor=GRID)
    return fig


A = load()

# ---- Sidebar: inputs ----
st.sidebar.header("Portfolio")
preset = st.sidebar.selectbox("Preset book", ["Concentrated", "Diversified"])
book = CONCENTRATED_BOOK if preset == "Concentrated" else DIVERSIFIED_BOOK
edited = st.sidebar.data_editor(
    pd.DataFrame({"ticker": list(book), "notional_$m": [v / 1e6 for v in book.values()]}),
    num_rows="dynamic")
size = st.sidebar.slider("Scale book size (x)", 0.25, 10.0, 1.0, 0.25)
model = st.sidebar.selectbox("Vol model", ["EWMA", "GARCH", "LightGBM"])
with st.sidebar.expander("Risk settings"):
    cfg = replace(
        CFG,
        conf=st.slider("ES confidence", 0.95, 0.995, CFG.conf, 0.005, format="%.3f"),
        conc_k=st.slider("Concentration slope k", 0.0, 5.0, CFG.conc_k, 0.5),
        hhi_threshold=st.slider("HHI threshold", 0.05, 0.50, CFG.hhi_threshold, 0.01),
        participation=st.slider("Participation rate", 0.05, 0.50, CFG.participation, 0.05),
        flat_margin=st.slider("Flat margin", 0.05, 0.50, CFG.flat_margin, 0.01),
    )

port = edited.dropna().set_index("ticker")["notional_$m"] * 1e6 * size
port = port[port.index.isin(A["rets"].columns) & (port != 0)]

st.title("Dynamic Margin Engine")
if port.empty:
    st.warning("Add at least one position with a ticker from the data set.")
    st.stop()

sigma_today = margin.daily_sigma_from_var_h(A["var_h"][model], cfg.horizon).iloc[-1]
res = margin.compute_margin(port, A["z"], sigma_today, A["adv"].iloc[-1], cfg)
as_of = A["rets"].index[-1].date()
st.caption(f"Margin as of {as_of}, using {model} volatility and {cfg.horizon}-day scenarios.")

# ---- 1. KPI row ----
flat_today = cfg.flat_margin * res.gross
c1, c2, c3, c4 = st.columns(4)
c1.metric("Dynamic margin", dollars_m(res.total),
          f"{res.total / res.gross:.1%} of gross", delta_color="off", delta_arrow="off")
c2.metric("Flat margin", dollars_m(flat_today),
          f"{cfg.flat_margin:.0%} of gross", delta_color="off", delta_arrow="off")
c3.metric("HHI", f"{res.hhi:.3f}")
c4.metric("Worst days to liquidate", f"{res.liq_days.max():.1f}", str(res.liq_days.idxmax()),
          delta_color="off", delta_arrow="off")

# ---- 2. Margin waterfall ----
st.header("Margin waterfall")
steps = np.array([res.es_base, res.liquidity_addon, res.concentration_addon,
                  res.floor_addon, res.total]) / 1e6
fig = go.Figure(go.Waterfall(
    measure=["absolute", "relative", "relative", "relative", "total"],
    x=["Base ES", "Liquidity", "Concentration", "Floor", "Total margin"],
    y=[*steps[:-1], 0],
    text=[f"${v:,.1f}m" for v in steps], textposition="outside",
    totals=dict(marker=dict(color=SERIES["EWMA"])),
    increasing=dict(marker=dict(color=SERIES["GARCH"])),
    decreasing=dict(marker=dict(color=SERIES["LightGBM"])),
    connector=dict(line=dict(color=GRID)), cliponaxis=False))
fig = style(fig, "$m").update_layout(hovermode="closest", showlegend=False)
st.plotly_chart(fig, width="stretch")

# ---- 3. Volatility forecasts ----
st.header("Volatility forecasts")
tickers = list(A["rets"].columns)
ticker = st.selectbox("Ticker", tickers, index=tickers.index(CFG.market))
test = A["target"].index > CFG.valid_end
dates = A["target"].index[test]
fig = go.Figure()
fig.add_scatter(x=dates, y=A["target"].loc[test, ticker] ** 0.5, name="Realised",
                line=dict(color=INK, width=1))
for name, var_h in A["var_h"].items():
    fig.add_scatter(x=dates, y=var_h.loc[test, ticker] ** 0.5, name=name,
                    line=dict(color=SERIES[name], width=2))
fig = style(fig, f"{cfg.horizon}-day volatility").update_yaxes(tickformat=".0%")
st.plotly_chart(fig, width="stretch")

# ---- 4. Model scorecard ----
st.header("Model scorecard")
st.dataframe(A["metrics"].round(4))
st.caption(f"Test period after {CFG.valid_end}. Lower is better. RMSE is on {cfg.horizon}-day "
           "volatility; QLIKE_highvol covers only the days when SPY's realised vol was in its "
           "top 20%.")

# ---- 5. Backtest ----
st.header(f"Backtest: {preset} book")
bt = A["backtests"][preset]
fig = go.Figure()
fig.add_scatter(x=bt.index, y=bt.realised_loss / 1e6, name=f"Realised {cfg.horizon}-day loss",
                line=dict(color=INK, width=1))
fig.add_scatter(x=bt.index, y=bt.dyn_margin / 1e6, name="Dynamic margin",
                line=dict(color=SERIES["EWMA"], width=2))
fig.add_scatter(x=bt.index, y=bt.flat_margin / 1e6, name=f"Flat {CFG.flat_margin:.0%}",
                line=dict(color=SERIES["GARCH"], width=2, dash="dash"))
b = bt[bt.breach_flat]
fig.add_scatter(x=b.index, y=b.realised_loss / 1e6, mode="markers", name="Flat breaches",
                marker=dict(color=CRITICAL, size=8, symbol="x"))
# A few extreme days would flatten the rest, so the axis stops short of them
top = 1.2 * bt[["dyn_margin", "realised_loss"]].max(axis=1).quantile(0.95) / 1e6
fig = style(fig, "$m").update_yaxes(range=[bt.realised_loss.min() / 1e6 * 1.05, top])
st.plotly_chart(fig, width="stretch")

summary = A["bt_summary"][preset].copy()
summary["worst_shortfall"] /= 1e6
st.dataframe(summary.rename(columns={"worst_shortfall": "worst_shortfall_$m"}).round(4))
preset_gross = sum(abs(v) for v in book.values())
peak = bt.dyn_margin.max()
st.caption(f"The preset book at size x1 with EWMA and config settings; the sidebar doesn't "
           f"change this section. Median dynamic margin: {bt.dyn_margin.median() / preset_gross:.1%} "
           f"of gross. Peak: {peak / preset_gross:,.0%} on {bt.dyn_margin.idxmax().date()}. The "
           f"chart's axis stops at ${top:,.0f}m to keep normal days readable; use the autoscale "
           "button on the chart to see everything.")

# ---- 6. Stress test ----
st.header("Stress test")
recent = A["rets"].iloc[-252:]
betas = recent.cov()[CFG.market] / recent[CFG.market].var()
names = list(port.index)
c1, c2, c3 = st.columns(3)
shock = c1.slider("Market shock", -30, 10, -10, 1, format="%d%%") / 100
gap_name = c2.selectbox("Gap ticker", names, index=names.index("GME") if "GME" in names else 0)
gap = c3.slider("Single-name gap", -50, 100, 100, 5, format="%d%%") / 100

R = betas[port.index] * np.log(1 + shock)
R[gap_name] += np.log(1 + gap)
stress_loss = -(port * (R.apply(np.exp) - 1)).sum()

c1, c2, c3 = st.columns(3)
c1.metric("Stress loss", dollars_m(stress_loss))
c2.metric("Margin", dollars_m(res.total))
if stress_loss > 0:
    coverage = res.total / stress_loss
    c3.metric("Coverage", f"{coverage:.0%}")
    st.write(f"If the market moves {shock:+.0%} and {gap_name} gaps {gap:+.0%}, "
             f"the margin covers {coverage:.0%} of the loss.")
else:
    c3.metric("Coverage", "n/a")
    st.write("This scenario makes money for the book, so there's no loss to cover.")
