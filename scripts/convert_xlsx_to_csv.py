#!/usr/bin/env python3
"""
Convert one or more .xlsx files to .csv while normalizing the time column
so it matches the format in data/binance_BTC-USDT_1d_last3.0y.csv:

- Add/ensure a timezone-aware UTC `datetime` column like "2022-08-22 00:00:00+00:00"
- Add/ensure an integer `timestamp` column in milliseconds since epoch
- Place `timestamp` and `datetime` as the first two columns in the output CSV

Heuristics to find the time column (in this order):
1) Explicit --time-col if provided
2) A numeric column named 'timestamp' (auto-detect seconds vs ms)
3) A column named one of: 'datetime', 'time', 'date', 'Date', 'Time', 'DateTime'
4) The first column with datetime dtype
5) Attempt to parse the first column to datetime

Usage examples:
  uv run scripts/convert_xlsx_to_csv.py data/6.5.xlsx data/10.xlsx data/100.xlsx
  uv run scripts/convert_xlsx_to_csv.py --time-col Date --sheet Sheet1 data/6.5.xlsx
  uv run scripts/convert_xlsx_to_csv.py --outdir data/converted data/*.xlsx
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable, Optional, Any, cast

import pandas as pd  # type: ignore
from pandas.api.types import is_scalar as pd_is_scalar, is_datetime64tz_dtype


CANDIDATE_TIME_NAMES = [
    "datetime", "time", "date", "Date", "Time", "DateTime",
]


def _looks_like_datetime_header(name: object) -> bool:
    """Heuristic: return True if a column *name* looks like data, not a header.

    This catches cases where the first row (a timestamp or number) was used as the
    header by pd.read_excel with the default header=0.
    """
    # Non-string headers like Timestamp, numbers, or NaN are suspicious
    if not isinstance(name, str):
        # pandas.Timestamp or datetime-like
        try:
            # Only check NA for scalars to avoid ambiguous truth values
            if pd_is_scalar(name) and bool(pd.isna(name)):
                return True
        except Exception:
            pass
        # Timestamp instance
        if hasattr(name, "tz") or isinstance(name, pd.Timestamp):
            return True
        # Numeric types
        if isinstance(name, (int, float)):
            return True
        # Fallback: try to coerce to datetime
        try:
            parsed = pd.to_datetime(cast(Any, name), utc=True, errors="coerce")
            if parsed is not pd.NaT and not pd.isna(parsed):
                return True
        except Exception:
            pass
        return False
    name_str = cast(str, name) if isinstance(name, str) else None
    if name_str is not None and name_str.startswith("Unnamed"):
        return True
    # Datetime-like strings
    try:
        parsed = pd.to_datetime(name_str if name_str is not None else name, utc=True, errors="coerce")
        if parsed is not pd.NaT and not pd.isna(parsed):
            return True
    except Exception:
        pass
    # Purely numeric-like (e.g., "10")
    try:
        float(name_str if name_str is not None else name)
        return True
    except Exception:
        return False


def _as_utc_datetime(series: pd.Series) -> pd.Series:
    """Return a tz-aware (UTC) pandas datetime Series.

    - If series is datetime64 but tz-naive, localize to UTC.
    - If tz-aware, convert to UTC.
    - If numeric, treat as epoch seconds or milliseconds depending on magnitude.
    - Otherwise, attempt to parse with pd.to_datetime and localize to UTC if naive.
    """
    s = series.copy()

    # Numeric epoch
    if pd.api.types.is_numeric_dtype(s):
        # Heuristic: epoch seconds ~1e9 range, ms ~1e12
        # If values are very large (>= 10^11), assume ms. Else seconds.
        # Use median of non-na values to be robust.
        sample = cast(pd.Series, pd.to_numeric(s.dropna().astype("float"), errors="coerce"))
        if sample.empty:
            raise ValueError("Time column has no valid numeric values to infer epoch.")
        med = float(cast(pd.Series, sample).median())
        unit = "ms" if med >= 1e11 else "s"
        dt = pd.to_datetime(s, unit=unit, utc=True, errors="coerce")
        if dt.isna().all():
            raise ValueError("Failed to parse numeric epoch to datetime.")
        return dt

    # Already datetime
    if pd.api.types.is_datetime64_any_dtype(s) or is_datetime64tz_dtype(s):
        # If tz-aware, convert to UTC; if naive, localize to UTC
        if getattr(s.dt, "tz", None) is not None:
            return s.dt.tz_convert("UTC")
        return s.dt.tz_localize("UTC")

    # Fallback: parse strings
    dt = pd.to_datetime(s, utc=True, errors="coerce")
    if dt.isna().all():
        raise ValueError("Failed to parse time column to datetime.")
    return dt


def _choose_time_column(df: pd.DataFrame, explicit: Optional[object]) -> object:
    if explicit is not None:
        # Only accept explicit if it is a string-like column label present in df
        if isinstance(explicit, str) and explicit in df.columns:
            return explicit
        raise KeyError(f"--time-col '{explicit}' not found in columns: {list(df.columns)}")

    # 1) numeric 'timestamp'
    if "timestamp" in df.columns and pd.api.types.is_numeric_dtype(df["timestamp"]):
        return "timestamp"

    # 2) named candidates
    for name in CANDIDATE_TIME_NAMES:
        if name in df.columns:
            return name

    # 3) first datetime dtype column
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col

    # 4) fallback: first column
    return df.columns[0]


def _ensure_time_columns(
    df: pd.DataFrame,
    time_col: Optional[object],
    *,
    portfolio_override_col: Optional[object] = None,
    portfolio_source_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    col = _choose_time_column(df, time_col)
    dt = _as_utc_datetime(cast(pd.Series, df[col]))

    # Milliseconds since epoch (int)
    # Use astype instead of view to avoid deprecation warnings.
    ts_ms = (dt.astype("int64") // 1_000_000).astype("Int64")

    # Output: timestamp, datetime (space-separated ISO8601), and ONLY the last non-time column
    # Keep colon in timezone offset and use space instead of 'T' to match Binance CSVs
    dt_iso = dt.map(lambda x: x.isoformat().replace("T", " ") if pd.notna(x) else "")
    out = pd.DataFrame({
        "timestamp": ts_ms,
        "datetime": dt_iso,
    })

    # Decide portfolio column: choose the rightmost non-empty numeric-like column from the cleaned df
    candidates = [
        c for c in df.columns
        if c not in ("timestamp", "datetime") and c != col
    ]

    def _score_series(s: pd.Series) -> tuple[int, float, float]:
        # returns (position score placeholder, non_null_fraction, numeric_fraction)
        try:
            s2 = s.replace(r"^\s*$", pd.NA, regex=True)
        except Exception:
            s2 = s
        try:
            non_null = s2.notna().sum()
            total = len(s2)
            non_null_frac = float(non_null) / float(total) if total else 0.0
        except Exception:
            non_null_frac = 0.0
        try:
            numeric = cast(pd.Series, pd.to_numeric(s2, errors="coerce"))
            numeric_frac = float(numeric.notna().sum()) / float(len(numeric)) if len(numeric) else 0.0
        except Exception:
            numeric_frac = 0.0
        return (0, non_null_frac, numeric_frac)

    chosen: Optional[object] = None
    # Prefer rightmost with numeric_frac >= 0.5; fallback to rightmost with non_null_frac > 0
    for c in reversed(candidates):
        s = cast(pd.Series, df[c])
        _, nnf, nf = _score_series(s)
        if nf >= 0.5:
            chosen = c
            break
    if chosen is None:
        for c in reversed(candidates):
            s = cast(pd.Series, df[c])
            _, nnf, _ = _score_series(s)
            if nnf > 0.0:
                chosen = c
                break
    if chosen is not None:
        out["portfolio_btc"] = df[chosen]

    return out


def convert_one(path: str, sheet: Optional[str], time_col: Optional[str], outdir: Optional[str]) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    read_kwargs = {"sheet_name": sheet} if sheet else {}
    # First attempt with default header=0
    df = pd.read_excel(path, **read_kwargs)  # type: ignore
    df_raw = df.copy()

    # If headers look like data (e.g., datetime strings or 'Unnamed'), re-read with no header
    if any(_looks_like_datetime_header(c) for c in df.columns):
        df = pd.read_excel(path, header=None, **read_kwargs)  # type: ignore
        df_raw = df.copy()

    # Treat empty/whitespace strings as NA, then drop columns entirely empty
    try:
        df = df.replace(r"^\s*$", pd.NA, regex=True)
        df = df.dropna(axis=1, how="all")
    except Exception:
        pass

    # Determine original last column for portfolio_btc override
    portfolio_override_col = None
    try:
        if len(df_raw.columns) > 0:
            portfolio_override_col = df_raw.columns[-1]
    except Exception:
        portfolio_override_col = None

    df_out = _ensure_time_columns(
        df,
        time_col,
        portfolio_override_col=portfolio_override_col,
        portfolio_source_df=df_raw,
    )

    # Default output path: alongside input with .csv
    base = os.path.splitext(os.path.basename(path))[0] + ".csv"
    out_dir = outdir if outdir else os.path.dirname(path)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, base)

    # Ensure pandas writes tz-aware datetime in the desired format like '...+00:00'
    df_out.to_csv(out_path, index=False)

    return out_path


def main(argv: Optional[Iterable[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Convert .xlsx to .csv with normalized UTC time columns")
    p.add_argument("inputs", nargs="+", help="Input .xlsx files (glob expanded by shell)")
    p.add_argument("--sheet", default=None, help="Sheet name to read (default: first sheet)")
    p.add_argument("--time-col", default=None, help="Which column contains time (default: auto-detect)")
    p.add_argument("--outdir", default=None, help="Output directory (default: same as input)")
    args = p.parse_args(list(argv) if argv is not None else None)

    ok = 0
    for inp in args.inputs:
        try:
            out_path = convert_one(inp, args.sheet, args.time_col, args.outdir)
            print(f"Converted: {inp} -> {out_path}")
            ok += 1
        except Exception as e:
            print(f"[ERROR] {inp}: {e}", file=sys.stderr)

    if ok == 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
