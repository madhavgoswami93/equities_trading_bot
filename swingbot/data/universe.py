"""Build the 'every stock trading on NSE today' universe, tagged with market-cap class
and industry, from NSE's own public files.

    EQUITY_L.csv                      every listed equity: symbol, name, series, ISIN, listing date
    Nifty 100 / Midcap 150 /          membership gives the SEBI/AMFI cap class: large = top 100,
    Smallcap 250 / Microcap 250       mid = 101-250, small = 251-500, micro = 501-750, rest = "other"
    (the index files also carry the industry)

Caveat: the cap class and the stock list are as of today (not point-in-time), so a
backtest over earlier months carries some survivorship bias.
"""

from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import pandas as pd

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
EQUITY_LIST = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
INDEX_LISTS = {  # cap class -> constituent file (niftyindices.com)
    "large": "https://niftyindices.com/IndexConstituent/ind_nifty100list.csv",
    "mid": "https://niftyindices.com/IndexConstituent/ind_niftymidcap150list.csv",
    "small": "https://niftyindices.com/IndexConstituent/ind_niftysmallcap250list.csv",
    "micro": "https://niftyindices.com/IndexConstituent/ind_niftymicrocap250_list.csv",
}
TRADABLE_SERIES = ("EQ", "BE")  # EQ = normal; BE = trade-for-trade (delivery only) -- both fine for swing


def _get_csv(url: str) -> pd.DataFrame:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "text/csv,*/*"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return pd.read_csv(io.BytesIO(r.read()))


def build_nse_universe(out: str | Path, series: tuple[str, ...] = TRADABLE_SERIES) -> pd.DataFrame:
    eq = _get_csv(EQUITY_LIST)
    eq.columns = [c.strip().lower() for c in eq.columns]
    eq = eq.rename(columns={"name of company": "name", "isin number": "isin", "date of listing": "listed"})
    eq["series"] = eq["series"].str.strip()
    eq = eq[eq["series"].isin(series)][["symbol", "name", "series", "isin", "listed"]].copy()
    eq["listed"] = pd.to_datetime(eq["listed"], format="%d-%b-%Y", errors="coerce").dt.date

    cap, industry = {}, {}
    for cls, url in INDEX_LISTS.items():
        idx = _get_csv(url)
        for _, r in idx.iterrows():
            cap.setdefault(r["Symbol"], cls)
            industry.setdefault(r["Symbol"], r["Industry"])
    eq["cap"] = eq["symbol"].map(cap).fillna("other")
    eq["industry"] = eq["symbol"].map(industry).fillna("Unclassified")
    eq = eq.sort_values("symbol").reset_index(drop=True)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    eq.to_csv(out, index=False)
    return eq


def universe_meta(path: str | Path) -> pd.DataFrame:
    """Per-symbol static attributes (cap class, industry, series) if the universe file has them."""
    df = pd.read_csv(path)
    keep = [c for c in ("symbol", "cap", "industry", "series", "listed") if c in df]
    return df[keep].set_index("symbol")
