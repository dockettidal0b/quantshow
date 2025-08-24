#!/usr/bin/env python3
import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import ccxt  # type: ignore
import pandas as pd  # type: ignore


def fetch_ohlcv_range(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    since_ms: int,
    until_ms: Optional[int] = None,
    limit: int = 1000,
    params: Optional[dict] = None,
) -> List[list]:
    """
    Paginate forward using since+limit until we reach until_ms (if provided) or no more data.

    Returns a list of [timestamp, open, high, low, close, volume].
    """
    params = params or {}
    all_rows: List[list] = []
    tf_ms = exchange.parse_timeframe(timeframe) * 1000

    while True:
        # Respect rate limits
        time.sleep(getattr(exchange, "rateLimit", 0) / 1000)

        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since_ms, limit, params)
        if not ohlcv:
            break

        all_rows.extend(ohlcv)

        last_ts = ohlcv[-1][0]
        next_since = last_ts + tf_ms

        if until_ms is not None and next_since >= until_ms:
            break

        # Protection against endless loops when exchange returns identical last bar
        if next_since <= since_ms:
            break

        since_ms = next_since

    return all_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch OHLCV via ccxt and save to CSV")
    parser.add_argument("--exchange", default="binance", help="ccxt exchange id, e.g., binance")
    parser.add_argument("--symbol", default="BTC/USDT", help="Trading pair symbol, e.g., BTC/USDT")
    parser.add_argument("--timeframe", default="1d", help="Timeframe, e.g., 1m, 1h, 1d")
    parser.add_argument("--years", type=float, default=3.0, help="How many years of data to fetch")
    parser.add_argument(
        "--out",
        default=None,
        help="Output CSV path. If not set, will be data/{exchange}_{symbol_underscored}_{timeframe}_last{years}y.csv",
    )
    parser.add_argument("--limit", type=int, default=1000, help="Per-request candle limit (Binance up to 1000)")

    args = parser.parse_args()

    # Construct exchange instance
    if not hasattr(ccxt, args.exchange):
        print(f"Unknown exchange id: {args.exchange}", file=sys.stderr)
        sys.exit(1)

    ExchangeClass = getattr(ccxt, args.exchange)
    exchange: ccxt.Exchange = ExchangeClass({
        "enableRateLimit": True,
        # Add API keys here only if needed for private endpoints; public OHLCV is unauthenticated
    })

    # Validate capabilities and market
    exchange.load_markets()
    if not exchange.has.get("fetchOHLCV", False):
        print(f"Exchange '{args.exchange}' does not support fetch_ohlcv", file=sys.stderr)
        sys.exit(1)

    if args.symbol not in exchange.markets:
        # Try to find a unified symbol match (e.g., strip margin/spot differences)
        # Fallback: raise an error
        print(f"Symbol '{args.symbol}' not found on {args.exchange}", file=sys.stderr)
        sys.exit(1)

    now_utc = datetime.now(timezone.utc)
    # Add a small buffer to ensure complete coverage
    start_dt = (now_utc - timedelta(days=int(365 * args.years) + 3)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    since_ms = int(start_dt.timestamp() * 1000)
    until_ms = int(now_utc.timestamp() * 1000)

    print(
        f"Fetching {args.symbol} {args.timeframe} from {args.exchange} starting {exchange.iso8601(since_ms)} (last {args.years}y)"
    )

    rows = fetch_ohlcv_range(
        exchange=exchange,
        symbol=args.symbol,
        timeframe=args.timeframe,
        since_ms=since_ms,
        until_ms=until_ms,
        limit=args.limit,
        params=None,  # Keep simple and portable; forward pagination via since+limit
    )

    if not rows:
        print("No data fetched.")
        sys.exit(0)

    # Deduplicate by timestamp
    seen = set()
    uniq_rows = []
    for r in rows:
        ts = r[0]
        if ts not in seen:
            uniq_rows.append(r)
            seen.add(ts)

    df = pd.DataFrame(uniq_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])  # type: ignore
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

    # Ensure we only keep the last N years from start_dt
    df = df[df["datetime"] >= pd.Timestamp(start_dt)]

    # Sort by time ascending
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Derive default output path if needed
    if args.out is None:
        sym_safe = args.symbol.replace("/", "-")
        fname = f"{args.exchange}_{sym_safe}_{args.timeframe}_last{args.years}y.csv"
        out_path = os.path.join("data", fname)
    else:
        out_path = args.out

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)

    print(
        f"Saved {len(df)} rows to {out_path}. First: {df['datetime'].iloc[0]}, Last: {df['datetime'].iloc[-1]}"
    )


if __name__ == "__main__":
    main()
