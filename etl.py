"""
ETL for the Chicago bird-window-collision study.

One command replaces the manual workflow this analysis used to be — download two
files by hand, retype columns, delete bad rows, VLOOKUP the light scores onto the
collision dates, then add season columns. `python etl.py` does all of it and is
reproducible end to end:

    1. collect   fetch both source files (cached under data/raw/)
    2. clean     type dates, drop invalid/incomplete records, normalise labels
    3. merge     join each McCormick Place collision day to that day's light score
    4. features  derive year / month / season / decade
    5. write     data/processed/collisions_clean.csv  and  mp_daily.csv

Source: Winger, Weeks, Farnsworth, Jones, Hennen & Willard (2019),
"Nocturnal flight-calling behaviour predicts vulnerability to artificial light
in migratory birds", Proc. R. Soc. B — distributed via the TidyTuesday mirror.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"

SOURCES = {
    "bird_collisions.csv": "https://raw.githubusercontent.com/rfordatascience/"
    "tidytuesday/master/data/2019/2019-04-30/bird_collisions.csv",
    "mp_light.csv": "https://raw.githubusercontent.com/rfordatascience/"
    "tidytuesday/master/data/2019/2019-04-30/mp_light.csv",
}

# The dataset only spans migration months; label them by season.
SEASON = {3: "Spring", 4: "Spring", 5: "Spring",
          8: "Fall", 9: "Fall", 10: "Fall", 11: "Fall"}

LOCALITY = {"MP": "McCormick Place", "CHI": "Greater Chicago"}


def collect() -> None:
    """Download the source files once; skip if already cached."""
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        dest = RAW / name
        if dest.exists():
            print(f"  cached   {name}")
            continue
        print(f"  fetching {name} …")
        urllib.request.urlretrieve(url, dest)


def clean_collisions() -> pd.DataFrame:
    """Type, filter, and normalise the record-level collision data."""
    df = pd.read_csv(RAW / "bird_collisions.csv")
    n0 = len(df)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "genus", "species", "family"])
    # a handful of records fall outside the labelled migration months
    df = df[df["date"].dt.month.isin(SEASON)]

    df["family"] = df["family"].str.strip()
    df["flight_call"] = df["flight_call"].str.strip().str.title()
    df["locality"] = df["locality"].map(LOCALITY).fillna(df["locality"])
    df["scientific_name"] = df["genus"].str.strip() + " " + df["species"].str.strip()

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["season"] = df["month"].map(SEASON)
    df["decade"] = (df["year"] // 10 * 10).astype("Int64")

    print(f"  cleaned  {n0:,} -> {len(df):,} collision records "
          f"({n0 - len(df):,} dropped)")
    return df


def build_mp_daily(coll: pd.DataFrame) -> pd.DataFrame:
    """McCormick Place daily collision counts joined to that day's light score."""
    light = pd.read_csv(RAW / "mp_light.csv")
    light["date"] = pd.to_datetime(light["date"], errors="coerce")
    light = light.dropna(subset=["date", "light_score"])

    mp = (coll[coll["locality"] == "McCormick Place"]
          .groupby("date").size().rename("collisions").reset_index())
    daily = mp.merge(light, on="date", how="inner")
    daily["month"] = daily["date"].dt.month
    daily["season"] = daily["month"].map(SEASON)
    print(f"  merged   {len(daily):,} McCormick Place days with a light score")
    return daily


def main() -> None:
    print("[1/2] collect")
    collect()
    print("[2/2] clean + merge + features")
    coll = clean_collisions()
    daily = build_mp_daily(coll)

    PROC.mkdir(parents=True, exist_ok=True)
    coll.to_csv(PROC / "collisions_clean.csv", index=False)
    daily.to_csv(PROC / "mp_daily.csv", index=False)
    print(f"\nwrote {PROC.relative_to(ROOT)}/collisions_clean.csv "
          f"and mp_daily.csv")


if __name__ == "__main__":
    try:
        main()
    except urllib.error.URLError as e:
        sys.exit(f"download failed ({e}); check your connection and retry.")
