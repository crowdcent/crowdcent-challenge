"""Helper functions for loading and merging multi-source training/inference data.

Simple high-level API:
    from utils import load_training_data, load_inference_data, summarize, plot_cv

    client = cc.ChallengeClient("hyperliquid-ranking")
    merged = load_training_data(client)  # One-liner!
    summarize(merged)
"""

import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import requests

DATA_DIR = Path("data")
YIEDL_API_URL = "https://api.yiedl.ai/yiedl/v1/downloadDataset"


# =============================================================================
# PUBLIC API
# =============================================================================


def load_training_data(
    client, include_yiedl=True, include_numerai=True
) -> pl.DataFrame:
    """Load and merge all training data sources."""
    cc_data = _load_crowdcent(client)
    cc_symbols = set(cc_data["id"].unique().to_list())

    yiedl_data = _load_yiedl(cc_symbols, cc_data) if include_yiedl else None
    numerai_data = _load_numerai() if include_numerai else None

    return _merge_datasets(cc_data, yiedl_data, numerai_data)


def load_inference_data(
    client, include_yiedl=True, include_numerai=True
) -> pl.DataFrame:
    """Load and merge inference data with live YIEDL/Numerai."""
    path = DATA_DIR / "cc_inference.parquet"
    DATA_DIR.mkdir(exist_ok=True)
    client.download_inference_data("latest", str(path))
    cc_data = pl.read_parquet(path)
    print(f"CrowdCent inference: {cc_data.shape}")

    cc_symbols = set(cc_data["id"].unique().to_list())

    yiedl_data = _load_yiedl_live(cc_symbols) if include_yiedl else None
    numerai_data = _load_numerai_live() if include_numerai else None

    return _merge_datasets(cc_data, yiedl_data, numerai_data)


def plot_cv(cv, df: pl.DataFrame):
    """
    Plot cross-validation folds along a time axis.

    Each fold shows training (steelblue) and validation (orange) periods.

    Args:
        cv: TimeGapSplit cross-validator
        df: DataFrame with 'date' column
    """
    X_index_df = cv._join_date_and_x(df.to_pandas())
    X_index_df_pd = X_index_df.to_pandas()

    fig, ax = plt.subplots(figsize=(16, 4))
    for i, (train_idx, val_idx) in enumerate(cv.split(df)):
        train_dates = X_index_df_pd.iloc[train_idx]["__date__"].unique()
        val_dates = X_index_df_pd.iloc[val_idx]["__date__"].unique()
        ax.plot(train_dates, i * np.ones(train_dates.shape), c="steelblue", linewidth=4)
        ax.plot(val_dates, i * np.ones(val_dates.shape), c="orange", linewidth=4)

    ax.legend(("training", "validation"), loc="upper left")
    ax.set_ylabel("Fold")
    ax.set_yticks(range(cv.n_splits))
    ax.set_title("Cross-validation fold splits over time")
    plt.tight_layout()
    plt.show()


def summarize(df: pl.DataFrame) -> None:
    """Print summary stats for merged DataFrame."""
    print(f"Shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    cols = df.columns
    meta = {"id", "eodhd_id", "date", "target_10d", "target_30d"}
    cc_feats = [
        c for c in cols if not c.startswith(("yiedl_", "nmr_")) and c not in meta
    ]
    yiedl_feats = [c for c in cols if c.startswith("yiedl_")]
    nmr_feats = [c for c in cols if c.startswith("nmr_")]

    print(f"\nFeatures: {len(cc_feats) + len(yiedl_feats) + len(nmr_feats)} total")
    print(f"  - CrowdCent: {len(cc_feats)}")
    print(f"  - YIEDL:     {len(yiedl_feats)}")
    print(f"  - Numerai:   {len(nmr_feats)}")


# =============================================================================
# PRIVATE HELPERS
# =============================================================================


def _load_crowdcent(client) -> pl.DataFrame:
    """Load CrowdCent training data."""
    path = DATA_DIR / "cc_train.parquet"
    if not path.exists():
        print("Downloading CrowdCent training data...")
        DATA_DIR.mkdir(exist_ok=True)
        client.download_training_dataset("latest", str(path))
    df = pl.read_parquet(path)
    print(f"CrowdCent: {df.shape}")
    return df


def _load_yiedl(cc_symbols: set, cc_data: pl.DataFrame) -> pl.DataFrame | None:
    """Load YIEDL training data. Downloads and filters if not cached."""
    path = DATA_DIR / "yiedl_train.parquet"
    if path.exists():
        df = pl.read_parquet(path).filter(pl.col("symbol").is_in(cc_symbols))
        print(f"YIEDL: {df.shape}")
        return df

    # Download, filter, and cache
    try:
        df = _download_and_filter_yiedl(cc_symbols, cc_data)
        print(f"YIEDL: {df.shape}")
        return df
    except Exception as e:
        print(f"YIEDL download failed ({e}), skipping.")
        # Clean up partial files
        for p in [DATA_DIR / "yiedl_historical.zip", DATA_DIR / "yiedl_historical.parquet"]:
            p.unlink(missing_ok=True)
        return None


def _download_and_filter_yiedl(
    cc_symbols: set, cc_data: pl.DataFrame
) -> pl.DataFrame:
    """Download YIEDL historical data, filter to CC universe, and cache."""
    import shutil
    import zipfile

    DATA_DIR.mkdir(exist_ok=True)
    zip_path = DATA_DIR / "yiedl_historical.zip"
    raw_path = DATA_DIR / "yiedl_historical.parquet"
    filtered_path = DATA_DIR / "yiedl_train.parquet"

    # Stream download with progress
    print("Downloading YIEDL historical data (~9.7 GB)...")
    response = requests.get(f"{YIEDL_API_URL}?type=historical", stream=True)
    response.raise_for_status()

    downloaded = 0
    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            downloaded += len(chunk)
            if downloaded % (100 * 1024 * 1024) < len(chunk):
                print(f"\r  {downloaded / (1024**3):.1f} GB downloaded...", end="", flush=True)
    print(f"\r  Download complete: {downloaded / (1024**3):.1f} GB")

    # Extract parquet from zip
    print("Extracting...")
    with zipfile.ZipFile(zip_path) as zf:
        parquet_files = [n for n in zf.namelist() if n.endswith(".parquet")]
        if not parquet_files:
            raise ValueError("No parquet file found in YIEDL zip")
        with zf.open(parquet_files[0]) as src, open(raw_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
    zip_path.unlink()

    # Filter to CC universe and date range
    cc_min_date = cc_data["date"].min().date()
    cc_max_date = cc_data["date"].max().date()
    print(f"Filtering to {len(cc_symbols)} symbols, {cc_min_date} to {cc_max_date}...")
    yiedl_filtered = (
        pl.scan_parquet(raw_path)
        .filter(
            pl.col("symbol").is_in(cc_symbols)
            & (pl.col("date") >= cc_min_date)
            & (pl.col("date") <= cc_max_date)
        )
        .collect()
    )

    # Save filtered and clean up raw
    yiedl_filtered.write_parquet(filtered_path)
    raw_path.unlink()
    size_mb = filtered_path.stat().st_size / 1024 / 1024
    print(f"Cached: {size_mb:.0f} MB")
    return yiedl_filtered


def _load_yiedl_live(cc_symbols: set) -> pl.DataFrame | None:
    """Fetch live YIEDL data from API."""
    print("Fetching YIEDL live data...")
    response = requests.get(f"{YIEDL_API_URL}?type=latest")
    response.raise_for_status()
    df = pl.read_parquet(io.BytesIO(response.content))
    df = df.filter(pl.col("symbol").is_in(cc_symbols))
    print(f"YIEDL live: {df.shape}")
    return df


def _load_numerai() -> pl.DataFrame | None:
    """Load Numerai Crypto v2.0 training data."""
    from numerapi import CryptoAPI

    path = DATA_DIR / "numerai_train.parquet"
    if not path.exists():
        print("Downloading Numerai training data...")
        DATA_DIR.mkdir(exist_ok=True)
        CryptoAPI().download_dataset("v2.0/train.parquet", str(path))
    df = pl.read_parquet(path)
    print(f"Numerai: {df.shape}")
    return df


def _load_numerai_live() -> pl.DataFrame | None:
    """Fetch live Numerai data."""
    from numerapi import CryptoAPI

    print("Fetching Numerai live data...")
    path = DATA_DIR / "numerai_live.parquet"
    CryptoAPI().download_dataset("v2.0/live.parquet", str(path))
    df = pl.read_parquet(path)
    print(f"Numerai live: {df.shape}")
    return df


def _forward_fill_numerai(nmr_data: pl.DataFrame, cc_symbols: set) -> pl.DataFrame:
    """Forward-fill Numerai weekends by symbol.

    Numerai only has weekday data (Mon-Fri). This fills Sat/Sun with Friday's values.
    Only fills within each symbol's existing date range (doesn't extrapolate).
    """
    # Filter to overlapping symbols and convert date to date type
    nmr_filtered = nmr_data.filter(pl.col("symbol").is_in(cc_symbols))
    nmr_filtered = (
        nmr_filtered.with_columns(pl.col("date").dt.date().alias("date_d"))
        .drop("date")
        .rename({"date_d": "date"})
    )

    feature_cols = [c for c in nmr_filtered.columns if c.startswith("feature_")]

    # Get each symbol's date range
    symbol_ranges = nmr_filtered.group_by("symbol").agg(
        pl.col("date").min().alias("min_date"), pl.col("date").max().alias("max_date")
    )

    # Create per-symbol scaffold with all dates in their range
    scaffolds = []
    for row in symbol_ranges.iter_rows(named=True):
        dates = pl.date_range(row["min_date"], row["max_date"], eager=True)
        scaffolds.append(pl.DataFrame({"symbol": row["symbol"], "date": dates}))

    scaffold = pl.concat(scaffolds)

    # Join and forward-fill by symbol
    nmr_filled = scaffold.join(
        nmr_filtered.select(["symbol", "date"] + feature_cols),
        on=["symbol", "date"],
        how="left",
    ).sort(["symbol", "date"])

    nmr_filled = nmr_filled.with_columns(
        [pl.col(c).forward_fill().over("symbol") for c in feature_cols]
    )

    print(f"Numerai after weekend ffill: {nmr_filled.shape}")
    return nmr_filled


def _merge_datasets(cc_data, yiedl_data=None, numerai_data=None) -> pl.DataFrame:
    """Merge CC with optional YIEDL and Numerai data.

    Filters to 2020+ since Numerai starts 2020-01-01.
    Forward-fills Numerai weekends (Numerai only has Mon-Fri data).
    """
    from datetime import date

    # Filter CC to 2020+ (Numerai starts 2020-01-01)
    cc_filtered = cc_data.filter(pl.col("date") >= date(2020, 1, 1))
    print(f"CC after 2020 filter: {cc_filtered.shape}")

    cc_symbols = set(cc_filtered["id"].unique().to_list())
    merged = cc_filtered.with_columns(pl.col("date").dt.date().alias("join_date"))

    # YIEDL join (already has weekends, no ffill needed)
    if yiedl_data is not None:
        feature_cols = [c for c in yiedl_data.columns if c not in ["symbol", "date"]]
        yiedl_renamed = yiedl_data.rename(
            {c: f"yiedl_{c}" for c in feature_cols} | {"date": "join_date"}
        )
        merged = merged.join(
            yiedl_renamed,
            left_on=["id", "join_date"],
            right_on=["symbol", "join_date"],
            how="left",
        )
        print(f"After YIEDL join: {merged.shape}")

    # Numerai join (with weekend forward-fill)
    if numerai_data is not None and "symbol" in numerai_data.columns:
        nmr_filled = _forward_fill_numerai(numerai_data, cc_symbols)

        nmr_cols = [c for c in nmr_filled.columns if c.startswith("feature_")]
        nmr_renamed = nmr_filled.rename(
            {c: f"nmr_{c}" for c in nmr_cols} | {"date": "join_date"}
        )
        merged = merged.join(
            nmr_renamed,
            left_on=["id", "join_date"],
            right_on=["symbol", "join_date"],
            how="left",
        )
        print(f"After Numerai join: {merged.shape}")

    merged = merged.drop("join_date")
    print(f"Final merged: {merged.shape}")
    return merged
