from __future__ import annotations

import math
from pathlib import Path
from datetime import timedelta
from typing import Dict, Tuple, cast

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


# ---------------------------
# Paths and constants
# ---------------------------
PACKAGE_DIR = Path(__file__).resolve().parents[0]
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"

BTC_DAILY_FILE = DATA_DIR / "binance_BTC-USDT_1d_last3.0y.csv"
CLIENT_FILES = {
    "Client A (6.5 BTC start)": DATA_DIR / "6.5.csv",
    "Client B (10 BTC start)": DATA_DIR / "10.csv",
    "Client C (100 BTC start)": DATA_DIR / "100.csv",
}

ANNUALIZATION = 365  # crypto trades daily


# ---------------------------
# Utility functions
# ---------------------------
@st.cache_data(show_spinner=False)
def load_btc_daily(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Expect columns: timestamp, open, high, low, close, volume, datetime
    if "datetime" in df.columns:
        parsed = pd.to_datetime(df["datetime"], utc=True, format="ISO8601", errors="coerce")
        if parsed.notna().any():
            df = df[parsed.notna()].copy()
            df["datetime"] = parsed[parsed.notna()]
            df = df.set_index("datetime").sort_index()
        elif "timestamp" in df.columns:
            # Fallback when datetime strings are unparseable
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df = df.set_index("timestamp").sort_index()
        else:
            raise ValueError("BTC daily CSV has a 'datetime' column but values are unparseable and no 'timestamp' present")
    elif "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp").sort_index()
    else:
        raise ValueError("BTC daily CSV must include 'datetime' or 'timestamp' column")

    # Ensure numeric types
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Indicators
    df["sma20"] = df["close"].rolling(20).mean()
    df["sma50"] = df["close"].rolling(50).mean()
    df["sma200"] = df["close"].rolling(200).mean()
    df["ret"] = df["close"].pct_change()
    df["vol30"] = df["ret"].rolling(30).std() * math.sqrt(ANNUALIZATION)
    df["rsi14"] = rsi(pd.Series(df["close"]), window=14)

    # Bollinger Bands (20, 2)
    roll20 = df["close"].rolling(20)
    bb_std = roll20.std()
    df["bb_mid"] = df["sma20"]
    df["bb_up"] = df["sma20"] + 2 * bb_std
    df["bb_dn"] = df["sma20"] - 2 * bb_std

    # MACD (12, 26, 9)
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    return df


@st.cache_data(show_spinner=False)
def load_client(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Expect columns: timestamp, datetime, portfolio_btc
    if "datetime" in df.columns:
        parsed = pd.to_datetime(df["datetime"], utc=True, format="ISO8601", errors="coerce")
        if parsed.notna().any():
            df = df[parsed.notna()].copy()
            df["datetime"] = parsed[parsed.notna()]
            df = df.set_index("datetime").sort_index()
        elif "timestamp" in df.columns:
            # Fallback when datetime strings are unparseable
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df = df.set_index("timestamp").sort_index()
        else:
            raise ValueError("Client CSV has a 'datetime' column but values are unparseable and no 'timestamp' present")
    elif "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp").sort_index()
    else:
        raise ValueError("Client CSV must include 'datetime' or 'timestamp' column")

    if "portfolio_btc" not in df.columns:
        raise ValueError("Client CSV must include 'portfolio_btc' column")

    df["portfolio_btc"] = pd.to_numeric(df["portfolio_btc"], errors="coerce")

    # Convert to daily frequency using last observation of each day
    daily = df[["portfolio_btc"]].resample("1D").last().dropna()
    return daily


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)

    roll_up = pd.Series(up, index=series.index).rolling(window).mean()
    roll_down = pd.Series(down, index=series.index).rolling(window).mean()

    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def compute_portfolio_views(client_daily: pd.DataFrame, btc_daily: pd.DataFrame) -> pd.DataFrame:
    # Join on daily index intersection
    df = client_daily.join(
        btc_daily[["open", "high", "low", "close", "sma20", "sma50", "sma200", "rsi14", "ret"]],
        how="inner",
    )

    # Compute USD equity and indicators
    df["portfolio_usd"] = df["portfolio_btc"] * df["close"]
    df["usd_ret"] = df["portfolio_usd"].pct_change().fillna(0.0)

    # Rolling Sharpe (30d)
    df["sharpe30"] = df["usd_ret"].rolling(30).mean() / df["usd_ret"].rolling(30).std()
    df["sharpe30"] *= math.sqrt(ANNUALIZATION)

    # Rolling volatility (30d, annualized)
    df["vol30p"] = df["usd_ret"].rolling(30).std() * math.sqrt(ANNUALIZATION)

    # Rolling correlation and beta to BTC (using BTC daily returns from join)
    if "ret" not in df.columns:
        # Ensure btc returns are present for corr/beta
        pass
    else:
        df["corr30"] = df["usd_ret"].rolling(30).corr(df["ret"])  # portfolio vs BTC
        cov30 = df["usd_ret"].rolling(30).cov(df["ret"])
        var30 = df["ret"].rolling(30).var()
        df["beta30"] = cov30 / var30

    # Drawdown on USD equity
    roll_max = df["portfolio_usd"].cummax()
    df["drawdown"] = df["portfolio_usd"] / roll_max - 1.0

    # Moving averages on equity
    df["usd_ma30"] = df["portfolio_usd"].rolling(30).mean()
    df["usd_ma90"] = df["portfolio_usd"].rolling(90).mean()

    return df.dropna(subset=["close"])  # ensure valid rows


def perf_summary(df: pd.DataFrame) -> Dict[str, float]:
    first_btc = df["portfolio_btc"].iloc[0]
    last_btc = df["portfolio_btc"].iloc[-1]
    first_usd = df["portfolio_usd"].iloc[0]
    last_usd = df["portfolio_usd"].iloc[-1]

    total_ret_btc = (last_btc / first_btc) - 1.0 if first_btc and first_btc != 0 else np.nan
    total_ret_usd = (last_usd / first_usd) - 1.0 if first_usd and first_usd != 0 else np.nan
    max_dd = df["drawdown"].min()
    sharpe30 = df["sharpe30"].iloc[-1]

    return {
        "btc_position": last_btc,
        "equity_usd": last_usd,
        "total_ret_btc": total_ret_btc,
        "total_ret_usd": total_ret_usd,
        "max_drawdown": max_dd,
        "sharpe30": sharpe30,
    }


def make_combined_nav_chart(df: pd.DataFrame, title: str) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Left Y: USD — BTC close and portfolio USD
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["close"],
            name="BTC Close (USD)",
            mode="lines",
            line=dict(color="#1f77b4"),
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["portfolio_usd"],
            name="Portfolio (USD)",
            mode="lines",
            line=dict(color="#2ca02c"),
        ),
        secondary_y=False,
    )

    # Right Y: BTC holdings
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["portfolio_btc"],
            name="Portfolio (BTC)",
            mode="lines",
            line=dict(color="#ff7f0e"),
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title=title,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=40, b=10),
        height=450,
        template="plotly_white",
    )

    fig.update_yaxes(title_text="USD", secondary_y=False)
    fig.update_yaxes(title_text="BTC", secondary_y=True)
    return fig


def make_btc_candles(df_btc: pd.DataFrame, date_range: Tuple[pd.Timestamp, pd.Timestamp] | None = None) -> go.Figure:
    data = df_btc
    if date_range is not None:
        start, end = date_range
        data = data.loc[(data.index >= start) & (data.index <= end)]

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["open"],
            high=data["high"],
            low=data["low"],
            close=data["close"],
            name="BTC",
            increasing_line_color="#2ca02c",
            decreasing_line_color="#d62728",
        )
    )

    fig.add_trace(go.Scatter(x=data.index, y=data["sma20"], name="SMA 20", line=dict(color="#9467bd")))
    fig.add_trace(go.Scatter(x=data.index, y=data["sma50"], name="SMA 50", line=dict(color="#8c564b")))
    fig.add_trace(go.Scatter(x=data.index, y=data["sma200"], name="SMA 200", line=dict(color="#7f7f7f")))

    fig.update_layout(
        title="BTC Daily Candles + SMAs",
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=40, b=10),
        height=450,
        template="plotly_white",
    )
    fig.update_yaxes(title_text="Price (USD)")
    return fig


def make_btc_market_panel(
    df_btc: pd.DataFrame, date_range: Tuple[pd.Timestamp, pd.Timestamp] | None = None
) -> go.Figure:
    data = df_btc
    if date_range is not None:
        start, end = date_range
        data = data.loc[(data.index >= start) & (data.index <= end)]

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.55, 0.15, 0.15, 0.15],
    )

    # Row 1: Candles + SMAs + Bollinger
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["open"],
            high=data["high"],
            low=data["low"],
            close=data["close"],
            name="BTC",
            increasing_line_color="#2ca02c",
            decreasing_line_color="#d62728",
        ),
        row=1, col=1,
    )
    fig.add_trace(go.Scatter(x=data.index, y=data["sma20"], name="SMA 20", line=dict(color="#9467bd")), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data["sma50"], name="SMA 50", line=dict(color="#8c564b")), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data["sma200"], name="SMA 200", line=dict(color="#7f7f7f")), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data["bb_up"], name="BB Up", line=dict(color="#9edae5", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data["bb_mid"], name="BB Mid", line=dict(color="#c5b0d5", width=1, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data["bb_dn"], name="BB Dn", line=dict(color="#9edae5", width=1)), row=1, col=1)

    # Row 2: Volume
    fig.add_trace(
        go.Bar(x=data.index, y=data["volume"], name="Volume", marker_color="#7f7f7f"),
        row=2, col=1,
    )

    # Row 3: MACD
    fig.add_trace(go.Scatter(x=data.index, y=data["macd"], name="MACD", line=dict(color="#1f77b4")), row=3, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data["macd_signal"], name="Signal", line=dict(color="#ff7f0e")), row=3, col=1)
    fig.add_trace(
        go.Bar(x=data.index, y=data["macd_hist"], name="Hist", marker_color="#2ca02c"),
        row=3, col=1,
    )

    # Row 4: RSI with guide lines
    fig.add_trace(go.Scatter(x=data.index, y=data["rsi14"], name="RSI 14", line=dict(color="#d62728")), row=4, col=1)
    if len(data.index) > 0:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=[70] * len(data.index),
                name="RSI 70",
                line=dict(color="#d62728", width=1, dash="dash"),
                showlegend=False,
            ),
            row=4,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=[30] * len(data.index),
                name="RSI 30",
                line=dict(color="#2ca02c", width=1, dash="dash"),
                showlegend=False,
            ),
            row=4,
            col=1,
        )

    fig.update_layout(
        title="BTC Market Panel",
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=40, b=10),
        height=900,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.update_yaxes(title_text="RSI", row=4, col=1, range=[0, 100])
    return fig


def make_corr_beta_vol_chart(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)

    fig.add_trace(go.Scatter(x=df.index, y=df.get("corr30"), name="Rolling Corr (30d)", line=dict(color="#1f77b4")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df.get("beta30"), name="Rolling Beta (30d)", line=dict(color="#ff7f0e")), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df.get("vol30p"), name="Volatility (30d, ann.)", line=dict(color="#2ca02c")), row=2, col=1)

    fig.update_yaxes(title_text="Corr/Beta", row=1, col=1)
    fig.update_yaxes(title_text="Vol (ann.)", row=2, col=1)
    fig.update_layout(title="Risk Metrics: Corr, Beta, Volatility", height=500, template="plotly_white", margin=dict(l=10, r=10, t=40, b=10))
    return fig


def make_returns_hist(df: pd.DataFrame) -> go.Figure:
    r = df["usd_ret"].dropna() * 100.0
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=r, nbinsx=60, marker_color="#7f7f7f", name="Daily %"))
    fig.update_layout(title="Distribution of Daily Returns (USD)", template="plotly_white", margin=dict(l=10, r=10, t=40, b=10))
    fig.update_xaxes(title_text="Daily Return (%)")
    fig.update_yaxes(title_text="Count")
    return fig


def make_equity_and_drawdown(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])

    # Equity with MAs
    fig.add_trace(
        go.Scatter(x=df.index, y=df["portfolio_usd"], name="Equity (USD)", line=dict(color="#2ca02c")),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["usd_ma30"], name="MA 30d", line=dict(color="#1f77b4", dash="dash")),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["usd_ma90"], name="MA 90d", line=dict(color="#ff7f0e", dash="dot")),
        row=1, col=1,
    )

    # Drawdown area
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["drawdown"] * 100.0,
            name="Drawdown (%)",
            line=dict(color="#d62728"),
            fill="tozeroy",
        ),
        row=2, col=1,
    )

    fig.update_yaxes(title_text="USD", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)

    fig.update_layout(
        title="Equity Curve + Drawdown",
        margin=dict(l=10, r=10, t=40, b=10),
        height=520,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def make_perf_vs_btc(df: pd.DataFrame) -> go.Figure:
    # Cumulative returns normalized to 100
    base_port = df["portfolio_usd"].iloc[0]
    base_btc = df["close"].iloc[0]
    port_index = df["portfolio_usd"] / base_port * 100.0
    btc_index = df["close"] / base_btc * 100.0

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=port_index, name="Portfolio USD (Indexed)", line=dict(color="#2ca02c")))
    fig.add_trace(go.Scatter(x=df.index, y=btc_index, name="BTC Close (Indexed)", line=dict(color="#1f77b4")))

    fig.update_layout(
        title="Performance vs BTC (Indexed = 100)",
        margin=dict(l=10, r=10, t=40, b=10),
        height=380,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_yaxes(title_text="Index (100 = start)")
    return fig


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(
    page_title="QuantShow – Crypto Clients Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("Crypto Clients Performance Dashboard")
st.caption("BTC perpetual trading performance – net value in BTC and USD, plus market context and rolling metrics")

# Load data
btc_daily = load_btc_daily(BTC_DAILY_FILE)

# Sidebar controls
st.sidebar.header("Controls")
if btc_daily.empty:
    st.error("BTC daily data is empty. Please check the data file.")
    st.stop()

# Compute min/max dates safely with explicit casts for the type checker
min_idx = cast(pd.Timestamp, btc_daily.index.min())
max_idx = cast(pd.Timestamp, btc_daily.index.max())
min_date = min_idx.date()
max_date = max_idx.date()

# Default to last 1 year
default_start = max_date - timedelta(days=365)
default_start = max(default_start, min_date)

date_range = st.sidebar.date_input(
    "Date range",
    value=(default_start, max_date),
    min_value=min_date,
    max_value=max_date,
)

start_date: pd.Timestamp
end_date: pd.Timestamp
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date = cast(pd.Timestamp, pd.Timestamp(date_range[0]).tz_localize("UTC"))
    end_date = cast(pd.Timestamp, pd.Timestamp(date_range[1]).tz_localize("UTC"))
else:
    start_date = cast(pd.Timestamp, pd.Timestamp(min_date).tz_localize("UTC"))
    end_date = cast(pd.Timestamp, pd.Timestamp(max_date).tz_localize("UTC"))

dr: Tuple[pd.Timestamp, pd.Timestamp] = (start_date, end_date)

# BTC Market panel at the top for context
st.subheader("Market Context")
st.plotly_chart(
    make_btc_market_panel(btc_daily, dr),
    use_container_width=True,
    key="btc_market_panel",
)


# ---------------------------
# Cross-client analytics helpers
# ---------------------------
def make_correlation_heatmap(ret_df: pd.DataFrame, title: str) -> go.Figure:
    corr = ret_df.corr(min_periods=5)
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=list(corr.columns),
            y=list(corr.index),
            colorscale="RdBu",
            zmid=0,
            colorbar=dict(title="Corr"),
        )
    )
    fig.update_layout(title=title, height=420, template="plotly_white", margin=dict(l=10, r=10, t=40, b=10))
    return fig


def make_monthly_returns_heatmap(ret: pd.Series, title: str) -> go.Figure:
    r = ret.dropna()
    if r.empty:
        return go.Figure()
    monthly = (1.0 + r).resample("M").apply(lambda s: (1.0 + s).prod() - 1.0)
    if monthly.empty:
        return go.Figure()
    dfm = pd.DataFrame({
        "Year": monthly.index.year,
        "Month": monthly.index.month,
        "Ret": monthly.values * 100.0,
    })
    pivot = dfm.pivot(index="Year", columns="Month", values="Ret").reindex(columns=list(range(1, 13)))
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=month_labels,
            y=pivot.index,
            colorscale="RdYlGn",
            zmid=0,
            colorbar=dict(title="%"),
        )
    )
    fig.update_layout(title=title, height=420, template="plotly_white", margin=dict(l=10, r=10, t=40, b=10))
    return fig


def make_risk_return_scatter(ret_map: Dict[str, pd.Series], title: str) -> go.Figure:
    names: list[str] = []
    ann_ret: list[float] = []
    ann_vol: list[float] = []
    for name, s in ret_map.items():
        x = s.dropna()
        n = len(x)
        if n < 5:
            continue
        r = float(np.prod(1.0 + x) ** (ANNUALIZATION / n) - 1.0)
        v = float(x.std() * math.sqrt(ANNUALIZATION))
        names.append(name)
        ann_ret.append(r * 100.0)
        ann_vol.append(v * 100.0)
    fig = go.Figure(
        data=go.Scatter(
            x=ann_vol,
            y=ann_ret,
            mode="markers+text",
            text=names,
            textposition="top center",
            marker=dict(size=12, color="#1f77b4", opacity=0.8),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Volatility (ann. %)",
        yaxis_title="Return (ann. %)",
        height=420,
        template="plotly_white",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def make_drawdown_heatmap(client_dfs: Dict[str, pd.DataFrame]) -> go.Figure:
    if not client_dfs:
        return go.Figure()
    # Union of dates
    all_dates = sorted(set().union(*[df.index for df in client_dfs.values()]))
    names = list(client_dfs.keys())
    Z = []
    for name in names:
        s = client_dfs[name]["drawdown"].reindex(all_dates) * 100.0
        Z.append(s.values)
    fig = go.Figure(
        data=go.Heatmap(
            z=Z,
            x=all_dates,
            y=names,
            colorscale="Blues",
            reversescale=True,
            zmin=-100,
            zmax=0,
            colorbar=dict(title="Drawdown %"),
        )
    )
    fig.update_layout(title="Underwater Heatmap (Clients)", height=420, template="plotly_white", margin=dict(l=10, r=10, t=40, b=10))
    return fig


# ---------------------------
# Overview + Client tabs
# ---------------------------
btc_slice_full = btc_daily.loc[(btc_daily.index >= start_date) & (btc_daily.index <= end_date)]

# Pre-compute client data in selected range
clients_data: Dict[str, pd.DataFrame] = {}
ret_map: Dict[str, pd.Series] = {}
for client_name, file_path in CLIENT_FILES.items():
    client_daily = load_client(file_path)
    client_slice = client_daily.loc[(client_daily.index >= start_date) & (client_daily.index <= end_date)]
    if client_slice.empty or btc_slice_full.empty:
        continue
    dfv = compute_portfolio_views(client_slice, btc_slice_full)
    if dfv.empty:
        continue
    clients_data[client_name] = dfv
    ret_map[client_name] = cast(pd.Series, dfv["usd_ret"])

tabs = st.tabs(["Overview"] + list(CLIENT_FILES.keys()))

# Overview tab
with tabs[0]:
    st.subheader("Cross-Client Analytics")
    if clients_data:
        # Correlation heatmap incl. BTC
        ret_df = pd.DataFrame({name: s for name, s in ret_map.items()})
        ret_df["BTC"] = btc_slice_full["ret"]
        st.plotly_chart(
            make_correlation_heatmap(ret_df, "Daily Returns Correlation"),
            use_container_width=True,
            key="overview_corr_heatmap",
        )

        # Risk vs return scatter
        st.plotly_chart(
            make_risk_return_scatter(ret_map, "Risk vs Return (Annualized)"),
            use_container_width=True,
            key="overview_risk_return_scatter",
        )

        # Underwater heatmap across clients
        st.plotly_chart(
            make_drawdown_heatmap(clients_data),
            use_container_width=True,
            key="overview_underwater_heatmap",
        )

        # Monthly returns heatmap for selected client
        sel = st.selectbox("Monthly returns heatmap – select client", options=list(clients_data.keys()))
        st.plotly_chart(
            make_monthly_returns_heatmap(cast(pd.Series, clients_data[sel]["usd_ret"]), f"{sel} – Monthly Returns"),
            use_container_width=True,
            key=f"overview_monthly_heatmap_{sel}",
        )
    else:
        st.info("No client data available in selected date range.")

# Individual client tabs
for idx, (client_name, _file_path) in enumerate(CLIENT_FILES.items(), start=1):
    with tabs[idx]:
        st.markdown(f"### {client_name}")
        df = clients_data.get(client_name)
        if df is None or df.empty:
            st.info("No data in selected range.")
            continue

        # Metrics row
        metrics = perf_summary(df)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("BTC Position", f"{metrics['btc_position']:.6f} BTC")
        c2.metric("Equity (USD)", f"${metrics['equity_usd']:,.0f}")
        c3.metric("Total Return (USD)", f"{metrics['total_ret_usd']*100:.2f}%")
        c4.metric("Total Return (BTC)", f"{metrics['total_ret_btc']*100:.2f}%")
        c5.metric("Sharpe (30d)", f"{metrics['sharpe30']:.2f}")

        # Charts
        st.plotly_chart(
            make_combined_nav_chart(df, title="Combined Net Value"),
            use_container_width=True,
            key=f"{client_name}_combined_nav",
        )
        st.plotly_chart(
            make_equity_and_drawdown(df),
            use_container_width=True,
            key=f"{client_name}_equity_drawdown",
        )
        st.plotly_chart(
            make_perf_vs_btc(df),
            use_container_width=True,
            key=f"{client_name}_perf_vs_btc",
        )
        st.plotly_chart(
            make_corr_beta_vol_chart(df),
            use_container_width=True,
            key=f"{client_name}_corr_beta_vol",
        )
        with st.expander("Daily Returns Distribution"):
            st.plotly_chart(
                make_returns_hist(df),
                use_container_width=True,
                key=f"{client_name}_returns_hist",
            )
        with st.expander("Monthly Returns Heatmap"):
            st.plotly_chart(
                make_monthly_returns_heatmap(cast(pd.Series, df["usd_ret"]), f"{client_name} – Monthly Returns"),
                use_container_width=True,
                key=f"{client_name}_monthly_returns_heatmap",
            )

st.caption("Data sources: Client portfolio CSVs and Binance BTC/USDT daily OHLCV. All times UTC.")
