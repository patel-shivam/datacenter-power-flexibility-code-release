#!/usr/bin/env python
"""Download the public inputs of the datacenter simulation and reduce them to small CSVs.

    python analyze/fetch_datacenter_inputs.py [--only NAME,...]

Raw files land in a cache directory (``--cache``, default ``.cache/dc_public_inputs/<source>/``
under the repository's root), each source in its own directory; they are read as data only. What the simulation reads is the reduced
output in ``data/datacenter/`` -- small, versioned, with every source's URL,
access date, byte size and SHA-256 recorded in ``data/datacenter/SOURCES.json``.

Sources (each reduced to what analyze/datacenter_sim.py uses):

  azure_<kind>_2024_week  Azure LLM inference trace 2024, one week (Microsoft, AzurePublicDataset,
                        CC-BY-4.0), conversation and code: arrivals per 5 minutes -> a rate
                        profile with peak = 1. Only the timing is used; request shapes stay the
                        calibrated task's.
  eia930_<BA>_<year>    EIA-930 Hourly Electric Grid Monitor, balancing-authority hourly demand,
                        for PJM and ERCOT (ERCO), 2024 and 2025.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

import pandas as pd

E = Path(__file__).resolve().parent.parent
OUT = E / "data" / "datacenter"
CACHE = E / ".cache" / "dc_public_inputs"

EIA930_URL = "https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_{year}_{half}.csv"
EIA930_BAS = ("PJM", "ERCO")
EIA930_YEARS = (2024, 2025)


def _sources() -> dict:
    p = OUT / "SOURCES.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _record(name: str, url: str, path: Path, note: str = "") -> None:
    src = _sources()
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    src[name] = {"url": url, "accessed": dt.date.today().isoformat(), "bytes": path.stat().st_size,
                 "sha256": h.hexdigest(), "note": note}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "SOURCES.json").write_text(json.dumps(src, indent=2, sort_keys=True) + "\n")


def download(url: str, dest: Path) -> Path:
    """Fetch ``url`` into ``dest`` (its own directory) unless it is already there."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "datacenter-energy-research/1.0"})
    with urllib.request.urlopen(req, timeout=600) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    tmp.rename(dest)
    print(f"  downloaded {url} -> {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


# ======================================================================================
# EIA-930: hourly demand per balancing authority
# ======================================================================================

def eia930(cache: Path) -> None:
    for year in EIA930_YEARS:
        parts = []
        for half in ("Jan_Jun", "Jul_Dec"):
            url = EIA930_URL.format(year=year, half=half)
            try:
                raw = download(url, cache / "eia930" / Path(url).name)
            except Exception as e:                       # a half-year not yet published
                print(f"  skip {url}: {e}")
                continue
            _record(f"eia930_{year}_{half}", url, raw, "EIA-930 BALANCE six-month file")
            cols = ["Balancing Authority", "UTC Time at End of Hour", "Demand (MW)",
                    "Demand (MW) (Adjusted)"]
            d = pd.read_csv(raw, usecols=lambda c: c in cols, thousands=",", low_memory=False)
            parts.append(d[d["Balancing Authority"].isin(EIA930_BAS)])
        if not parts:
            continue
        d = pd.concat(parts, ignore_index=True)
        # the hour STARTS one hour before the stamped end
        d["t"] = pd.to_datetime(d["UTC Time at End of Hour"], format="%m/%d/%Y %I:%M:%S %p",
                                utc=True) - pd.Timedelta(hours=1)
        for ba in EIA930_BAS:
            x = d[d["Balancing Authority"] == ba].sort_values("t").drop_duplicates("t")
            dem = x["Demand (MW) (Adjusted)"].fillna(x["Demand (MW)"])
            out = pd.DataFrame({"t": x.t, "demand_mw": dem.to_numpy()})
            out = out[out.t.dt.year == year]
            # a year is kept only if it is (nearly) complete: the curtailment method needs the peak
            n_hours = 8784 if year % 4 == 0 else 8760
            if out.demand_mw.notna().sum() < 0.98 * n_hours:
                print(f"  {ba} {year}: only {out.demand_mw.notna().sum()} of {n_hours} hours, skipped")
                continue
            out["demand_mw"] = out.demand_mw.interpolate(limit=3)
            path = OUT / f"eia930_{ba}_{year}.csv.gz"
            out.to_csv(path, index=False, float_format="%.0f")
            print(f"  wrote {path}: {len(out)} hours, peak {out.demand_mw.max():,.0f} MW")


# ======================================================================================
# Azure LLM inference trace 2024 (one week): the arrival-rate profile
# ======================================================================================

AZURE_URL = ("https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024/"
             "AzureLLMInferenceTrace_{kind}_1week.csv")
BIN = "5min"


def azure(cache: Path) -> None:
    """Arrivals per 5 minutes from the week-long Azure LLM traces (conversation; code as a
    sensitivity), with each bin's mean prompt and output lengths kept for reference only."""
    for kind in ("conv", "code"):
        url = AZURE_URL.format(kind=kind)
        raw = download(url, cache / f"azure_{kind}_1week" / Path(url).name)
        _record(f"azure_llm_2024_{kind}_1week", url, raw,
                "Azure LLM inference trace 2024 (CC-BY-4.0); Splitwise ISCA'24 / DynamoLLM HPCA'25")
        acc = []
        for ch in pd.read_csv(raw, usecols=["TIMESTAMP", "ContextTokens", "GeneratedTokens"],
                              chunksize=2_000_000):
            t = pd.to_datetime(ch["TIMESTAMP"], utc=True, format="ISO8601").dt.floor(BIN)
            g = ch.assign(t=t).groupby("t").agg(reqs=("ContextTokens", "size"),
                                                ctx=("ContextTokens", "sum"),
                                                gen=("GeneratedTokens", "sum"))
            acc.append(g)
        g = pd.concat(acc).groupby(level=0).sum().sort_index()
        full = pd.date_range(g.index.min(), g.index.max(), freq=BIN, tz="UTC")
        g = g.reindex(full, fill_value=0)
        out = pd.DataFrame({"t": g.index, "reqs": g.reqs.to_numpy(),
                            "rate_reqs_s": g.reqs.to_numpy() / pd.Timedelta(BIN).total_seconds()})
        out["shape"] = out.rate_reqs_s / out.rate_reqs_s.max()
        out["mean_context_tokens"] = (g.ctx / g.reqs.where(g.reqs > 0)).to_numpy()
        out["mean_generated_tokens"] = (g.gen / g.reqs.where(g.reqs > 0)).to_numpy()
        path = OUT / f"azure_{kind}_2024_week.csv"
        out.to_csv(path, index=False, float_format="%.6g")
        print(f"  wrote {path}: {len(out)} bins, {out.reqs.sum():,} requests, "
              f"mean {out.rate_reqs_s.mean():.1f} req/s, peak {out.rate_reqs_s.max():.1f}, "
              f"trough {out.rate_reqs_s.min():.1f}")


SOURCES = {"eia930": eia930, "azure": azure}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default=",".join(SOURCES), help=f"subset of {tuple(SOURCES)}")
    ap.add_argument("--cache", type=Path, default=CACHE)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for name in args.only.split(","):
        print(f"[{name}]")
        SOURCES[name](args.cache)


if __name__ == "__main__":
    main()
