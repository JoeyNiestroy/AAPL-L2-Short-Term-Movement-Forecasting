"""
02_extract_tick_features.py

Extracts one feature row for every MBP-10 event.

Each invocation processes exactly one Parquet file selected by its
zero-based index in the sorted input directory.

"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

INPUT_DIR = Path("aapl_xnas_mbp10_parquet")
OUTPUT_DIR = Path("aapl_xnas_mbp10_tick_features")

N_LEVELS = 10
EPS = 1e-12

ROLLING_WINDOWS = {
    "100ms": "100ms",
    "1s": "1s",
    "10s": "10s",
    "60s": "60s",
    "5m": "5min",
}


# -----------------------------------------------------------------------------
# Arguments
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract per-tick features from one Parquet file selected "
            "by its index in the sorted input directory."
        )
    )

    parser.add_argument(
        "--index",
        type=int,
        required=True,
        help=(
            "Zero-based index of the input Parquet file after sorting "
            "the files alphabetically."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file when it already exists.",
    )

    return parser.parse_args()


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def safe_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
) -> np.ndarray:
    """
    Divide two arrays safely.

    Returns zero wherever the denominator is effectively zero.
    """
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)

    result = np.zeros_like(numerator, dtype=np.float64)

    np.divide(
        numerator,
        denominator,
        out=result,
        where=np.abs(denominator) > EPS,
    )

    return result


def time_window_return(
    timestamps_ns: np.ndarray,
    values: np.ndarray,
    window: pd.Timedelta,
) -> np.ndarray:
    """
    Calculate the log return over a trailing clock-time window.

    A value is produced for every tick. There is no resampling.

    For tick t, the lagged value is the first available observation at
    or after t - window.
    """
    window_ns = window.value
    target_times = timestamps_ns - window_ns

    lag_indices = np.searchsorted(
        timestamps_ns,
        target_times,
        side="left",
    )

    lag_values = values[lag_indices]

    result = np.zeros(len(values), dtype=np.float64)

    valid = (
        np.isfinite(values)
        & np.isfinite(lag_values)
        & (values > 0)
        & (lag_values > 0)
    )

    result[valid] = 10_000.0 * np.log(
        values[valid] / lag_values[valid]
    )

    return result


# -----------------------------------------------------------------------------
# Snapshot and event features
# -----------------------------------------------------------------------------

def extract_snapshot_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract current-book, current-event, and one-tick-change features.

    The output contains exactly one row for every row in the input.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            "The input Parquet index must be a DatetimeIndex."
        )

    # Stable sorting preserves source order when timestamps are identical.
    df = df.sort_index(kind="stable")

    features = pd.DataFrame(index=df.index)

    bid_px_cols = [
        f"bid_px_{level:02d}"
        for level in range(N_LEVELS)
    ]
    ask_px_cols = [
        f"ask_px_{level:02d}"
        for level in range(N_LEVELS)
    ]

    bid_sz_cols = [
        f"bid_sz_{level:02d}"
        for level in range(N_LEVELS)
    ]
    ask_sz_cols = [
        f"ask_sz_{level:02d}"
        for level in range(N_LEVELS)
    ]

    bid_ct_cols = [
        f"bid_ct_{level:02d}"
        for level in range(N_LEVELS)
    ]
    ask_ct_cols = [
        f"ask_ct_{level:02d}"
        for level in range(N_LEVELS)
    ]

    required_columns = (
        bid_px_cols
        + ask_px_cols
        + bid_sz_cols
        + ask_sz_cols
        + bid_ct_cols
        + ask_ct_cols
        + [
            "action",
            "side",
            "depth",
            "price",
            "size",
            "sequence",
        ]
    )

    missing_columns = sorted(
        set(required_columns) - set(df.columns)
    )

    if missing_columns:
        raise KeyError(
            f"Missing required columns: {missing_columns}"
        )

    bid_px = df[bid_px_cols].to_numpy(
        dtype=np.float64,
        copy=False,
    )
    ask_px = df[ask_px_cols].to_numpy(
        dtype=np.float64,
        copy=False,
    )

    bid_sz = df[bid_sz_cols].to_numpy(
        dtype=np.float64,
        copy=False,
    )
    ask_sz = df[ask_sz_cols].to_numpy(
        dtype=np.float64,
        copy=False,
    )

    bid_ct = df[bid_ct_cols].to_numpy(
        dtype=np.float64,
        copy=False,
    )
    ask_ct = df[ask_ct_cols].to_numpy(
        dtype=np.float64,
        copy=False,
    )

    best_bid = bid_px[:, 0]
    best_ask = ask_px[:, 0]

    best_bid_size = bid_sz[:, 0]
    best_ask_size = ask_sz[:, 0]

    mid = 0.5 * (best_bid + best_ask)
    spread = best_ask - best_bid

    # -------------------------------------------------------------------------
    # Top-of-book features
    # -------------------------------------------------------------------------

    features["mid"] = mid
    features["spread"] = spread

    features["spread_bps"] = (
        10_000.0 * safe_ratio(spread, mid)
    )

    microprice_denominator = (
        best_bid_size + best_ask_size
    )

    microprice = safe_ratio(
        best_bid_size * best_ask
        + best_ask_size * best_bid,
        microprice_denominator,
    )

    features["microprice"] = microprice

    features["microprice_minus_mid_bps"] = (
        10_000.0
        * safe_ratio(
            microprice - mid,
            mid,
        )
    )

    # -------------------------------------------------------------------------
    # Individual level features
    # -------------------------------------------------------------------------

    for level in range(N_LEVELS):
        level_name = f"{level + 1:02d}"

        features[f"bid_distance_bps_{level_name}"] = (
            10_000.0
            * safe_ratio(
                mid - bid_px[:, level],
                mid,
            )
        )

        features[f"ask_distance_bps_{level_name}"] = (
            10_000.0
            * safe_ratio(
                ask_px[:, level] - mid,
                mid,
            )
        )

        features[f"log_bid_size_{level_name}"] = np.log1p(
            bid_sz[:, level]
        )

        features[f"log_ask_size_{level_name}"] = np.log1p(
            ask_sz[:, level]
        )

        features[f"log_bid_count_{level_name}"] = np.log1p(
            bid_ct[:, level]
        )

        features[f"log_ask_count_{level_name}"] = np.log1p(
            ask_ct[:, level]
        )

    # -------------------------------------------------------------------------
    # Adjacent level gaps
    # -------------------------------------------------------------------------

    for level in range(N_LEVELS - 1):
        gap_name = f"{level + 1:02d}_{level + 2:02d}"

        features[f"bid_gap_bps_{gap_name}"] = (
            10_000.0
            * safe_ratio(
                bid_px[:, level]
                - bid_px[:, level + 1],
                mid,
            )
        )

        features[f"ask_gap_bps_{gap_name}"] = (
            10_000.0
            * safe_ratio(
                ask_px[:, level + 1]
                - ask_px[:, level],
                mid,
            )
        )

    # -------------------------------------------------------------------------
    # Cumulative depth and imbalance
    # -------------------------------------------------------------------------

    for n_levels in (1, 3, 5, 10):
        bid_size_sum = bid_sz[:, :n_levels].sum(axis=1)
        ask_size_sum = ask_sz[:, :n_levels].sum(axis=1)

        bid_count_sum = bid_ct[:, :n_levels].sum(axis=1)
        ask_count_sum = ask_ct[:, :n_levels].sum(axis=1)

        total_size = bid_size_sum + ask_size_sum
        total_count = bid_count_sum + ask_count_sum

        features[f"log_bid_depth_{n_levels}"] = np.log1p(
            bid_size_sum
        )

        features[f"log_ask_depth_{n_levels}"] = np.log1p(
            ask_size_sum
        )

        features[f"log_total_depth_{n_levels}"] = np.log1p(
            total_size
        )

        features[f"size_imbalance_{n_levels}"] = safe_ratio(
            bid_size_sum - ask_size_sum,
            total_size,
        )

        features[f"count_imbalance_{n_levels}"] = safe_ratio(
            bid_count_sum - ask_count_sum,
            total_count,
        )

    # -------------------------------------------------------------------------
    # Depth-weighted book distances
    # -------------------------------------------------------------------------

    bid_distance = mid[:, None] - bid_px
    ask_distance = ask_px - mid[:, None]

    bid_weighted_distance = safe_ratio(
        (bid_distance * bid_sz).sum(axis=1),
        bid_sz.sum(axis=1),
    )

    ask_weighted_distance = safe_ratio(
        (ask_distance * ask_sz).sum(axis=1),
        ask_sz.sum(axis=1),
    )

    features["bid_depth_distance_bps_10"] = (
        10_000.0
        * safe_ratio(
            bid_weighted_distance,
            mid,
        )
    )

    features["ask_depth_distance_bps_10"] = (
        10_000.0
        * safe_ratio(
            ask_weighted_distance,
            mid,
        )
    )

    features["depth_distance_asymmetry_bps_10"] = (
        features["ask_depth_distance_bps_10"]
        - features["bid_depth_distance_bps_10"]
    )

    # -------------------------------------------------------------------------
    # Current event features
    # -------------------------------------------------------------------------

    action = df["action"].astype(str)
    side = df["side"].astype(str)

    features["event_is_add"] = (
        action.eq("A").astype(np.int8)
    )
    features["event_is_cancel"] = (
        action.eq("C").astype(np.int8)
    )
    features["event_is_modify"] = (
        action.eq("M").astype(np.int8)
    )
    features["event_is_clear"] = (
        action.eq("R").astype(np.int8)
    )
    features["event_is_trade"] = (
        action.eq("T").astype(np.int8)
    )

    # Bid-side event: +1
    # Ask-side event: -1
    # Unspecified event: 0
    side_sign = np.select(
        [
            side.eq("B").to_numpy(),
            side.eq("A").to_numpy(),
        ],
        [
            1.0,
            -1.0,
        ],
        default=0.0,
    )

    event_size = df["size"].to_numpy(
        dtype=np.float64,
        copy=False,
    )

    event_price = df["price"].to_numpy(
        dtype=np.float64,
        copy=False,
    )

    features["event_side_sign"] = side_sign.astype(
        np.int8
    )

    features["event_depth"] = df["depth"].astype(
        np.int16
    )

    features["log_event_size"] = np.log1p(
        event_size
    )

    features["event_price_minus_mid_bps"] = (
        10_000.0
        * safe_ratio(
            event_price - mid,
            mid,
        )
    )

    features["signed_event_size"] = (
        side_sign * event_size
    )

    features["signed_trade_size"] = (
        side_sign
        * event_size
        * features["event_is_trade"].to_numpy()
    )

    # -------------------------------------------------------------------------
    # Changes from the previous tick
    # -------------------------------------------------------------------------

    mid_series = pd.Series(
        mid,
        index=df.index,
    )

    spread_series = pd.Series(
        spread,
        index=df.index,
    )

    best_bid_series = pd.Series(
        best_bid,
        index=df.index,
    )

    best_ask_series = pd.Series(
        best_ask,
        index=df.index,
    )

    features["mid_return_bps_1tick"] = (
        10_000.0 * np.log(mid_series).diff()
    )

    features["spread_change_1tick"] = (
        spread_series.diff()
    )

    features["best_bid_change_bps_1tick"] = (
        10_000.0
        * safe_ratio(
            best_bid_series.diff().to_numpy(),
            mid,
        )
    )

    features["best_ask_change_bps_1tick"] = (
        10_000.0
        * safe_ratio(
            best_ask_series.diff().to_numpy(),
            mid,
        )
    )

    features["bid_size_change_1tick"] = pd.Series(
        best_bid_size,
        index=df.index,
    ).diff()

    features["ask_size_change_1tick"] = pd.Series(
        best_ask_size,
        index=df.index,
    ).diff()

    features["imbalance_1_change_1tick"] = (
        features["size_imbalance_1"].diff()
    )

    features["imbalance_10_change_1tick"] = (
        features["size_imbalance_10"].diff()
    )

    # -------------------------------------------------------------------------
    # Timing and sequence features
    # -------------------------------------------------------------------------

    timestamps_ns = df.index.asi8

    dt_ns = np.empty(
        len(df),
        dtype=np.float64,
    )

    dt_ns[0] = np.nan
    dt_ns[1:] = np.diff(timestamps_ns)

    features["log_dt_microseconds"] = np.log1p(
        np.maximum(
            dt_ns / 1_000.0,
            0.0,
        )
    )

    sequence = df["sequence"].to_numpy(
        dtype=np.float64,
        copy=False,
    )

    sequence_gap = np.empty(
        len(df),
        dtype=np.float64,
    )

    sequence_gap[0] = np.nan
    sequence_gap[1:] = np.diff(sequence)

    features["sequence_gap"] = sequence_gap

    # -------------------------------------------------------------------------
    # Intraday position
    # -------------------------------------------------------------------------

    seconds_since_midnight = (
        df.index.hour * 3600
        + df.index.minute * 60
        + df.index.second
        + df.index.microsecond / 1e6
        + df.index.nanosecond / 1e9
    )

    features["seconds_since_open"] = np.asarray(
        seconds_since_midnight - 9.5 * 3600,
        dtype=np.float32,
    )

    return features


# -----------------------------------------------------------------------------
# Rolling clock-time features
# -----------------------------------------------------------------------------

def add_rolling_features(
    features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add trailing clock-time features at every tick.

    This does not resample or reduce the number of rows.
    """
    if not features.index.is_monotonic_increasing:
        features = features.sort_index(kind="stable")

    timestamps_ns = features.index.asi8

    event_count = pd.Series(
        np.ones(
            len(features),
            dtype=np.float32,
        ),
        index=features.index,
    )

    trade_indicator = features[
        "event_is_trade"
    ].astype(np.float32)

    event_size = np.expm1(
        features["log_event_size"].astype(np.float64)
    )

    signed_event_size = features[
        "signed_event_size"
    ].astype(np.float64)

    signed_trade_size = features[
        "signed_trade_size"
    ].astype(np.float64)

    squared_tick_return = (
        features["mid_return_bps_1tick"]
        .fillna(0.0)
        .astype(np.float64)
        .pow(2)
    )

    mid = features["mid"].to_numpy(
        dtype=np.float64,
        copy=False,
    )

    for suffix, window_string in ROLLING_WINDOWS.items():
        print(f"  Rolling window: {suffix}")

        window = pd.Timedelta(window_string)

        features[f"event_count_{suffix}"] = (
            event_count
            .rolling(
                window,
                min_periods=1,
            )
            .sum()
        )

        features[f"trade_count_{suffix}"] = (
            trade_indicator
            .rolling(
                window,
                min_periods=1,
            )
            .sum()
        )

        features[f"log_total_event_size_{suffix}"] = (
            np.log1p(
                event_size
                .rolling(
                    window,
                    min_periods=1,
                )
                .sum()
            )
        )

        features[f"signed_event_flow_{suffix}"] = (
            signed_event_size
            .rolling(
                window,
                min_periods=1,
            )
            .sum()
        )

        features[f"signed_trade_flow_{suffix}"] = (
            signed_trade_size
            .rolling(
                window,
                min_periods=1,
            )
            .sum()
        )

        features[f"realized_vol_bps_{suffix}"] = (
            np.sqrt(
                squared_tick_return
                .rolling(
                    window,
                    min_periods=1,
                )
                .sum()
            )
        )

        features[f"mean_imbalance_1_{suffix}"] = (
            features["size_imbalance_1"]
            .rolling(
                window,
                min_periods=1,
            )
            .mean()
        )

        features[f"mean_imbalance_10_{suffix}"] = (
            features["size_imbalance_10"]
            .rolling(
                window,
                min_periods=1,
            )
            .mean()
        )

        features[f"mid_return_bps_{suffix}"] = (
            time_window_return(
                timestamps_ns=timestamps_ns,
                values=mid,
                window=window,
            )
        )

    return features


# -----------------------------------------------------------------------------
# Process one file
# -----------------------------------------------------------------------------

def process_file(
    input_path: Path,
    output_path: Path,
) -> None:
    start_time = time.perf_counter()

    print(f"Reading: {input_path}")

    df = pd.read_parquet(input_path)

    print(f"Input ticks : {len(df):,}")
    print(f"Input cols  : {df.shape[1]:,}")

    features = extract_snapshot_features(df)

    print(
        f"Snapshot features complete: "
        f"{features.shape[1]:,} columns"
    )

    features = add_rolling_features(features)

    features.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True,
    )

    # Reduce final output size after all calculations are finished.
    float64_columns = features.select_dtypes(
        include=["float64"]
    ).columns

    features[float64_columns] = features[
        float64_columns
    ].astype(np.float32)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    features.to_parquet(
        output_path,
        compression="zstd",
        index=True,
    )

    elapsed = time.perf_counter() - start_time

    print()
    print(f"Output rows     : {len(features):,}")
    print(f"Output features : {features.shape[1]:,}")
    print(
        f"Output size     : "
        f"{output_path.stat().st_size / 1e9:.3f} GB"
    )
    print(f"Elapsed time    : {elapsed / 60:.2f} minutes")
    print(f"Saved           : {output_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.index < 0:
        raise ValueError(
            f"--index must be nonnegative, received {args.index}"
        )

    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Input directory does not exist: {INPUT_DIR}"
        )

    # Every array task independently creates the same sorted file list.
    input_files = sorted(
        INPUT_DIR.glob("*.parquet"),
        key=lambda path: path.name,
    )

    if not input_files:
        raise FileNotFoundError(
            f"No Parquet files found in: {INPUT_DIR}"
        )

    if args.index >= len(input_files):
        raise IndexError(
            f"Index {args.index} is out of range. "
            f"The directory contains {len(input_files)} Parquet files, "
            f"so valid indices are 0 through {len(input_files) - 1}."
        )

    input_path = input_files[args.index]
    output_path = OUTPUT_DIR / input_path.name

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print(
        f"Slurm job ID      : "
        f"{os.environ.get('SLURM_JOB_ID', 'manual')}"
    )
    print(
        f"Slurm array ID    : "
        f"{os.environ.get('SLURM_ARRAY_JOB_ID', 'manual')}"
    )
    print(
        f"Slurm task ID     : "
        f"{os.environ.get('SLURM_ARRAY_TASK_ID', 'manual')}"
    )
    print(f"Requested index   : {args.index}")
    print(
        f"Selected file     : "
        f"{args.index + 1} of {len(input_files)}"
    )
    print(f"Input             : {input_path}")
    print(f"Output            : {output_path}")
    print("=" * 80)
    print()

    if output_path.exists() and not args.overwrite:
        print(
            "Output already exists, so this task will be skipped."
        )
        print(f"Existing output: {output_path}")
        print("Pass --overwrite to regenerate it.")
        return

    process_file(
        input_path=input_path,
        output_path=output_path,
    )

    print()
    print(
        f"Task index {args.index} completed successfully."
    )


if __name__ == "__main__":
    main()
