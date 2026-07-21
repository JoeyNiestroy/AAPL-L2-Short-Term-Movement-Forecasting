"""
03_make_tick_labels.py

Creates one label row for every MBP-10 event.

For an original event i:

    entry_index = i + latency_ticks

The exit is the first event whose timestamp is at least:

    entry_timestamp + horizon

The script creates:

1. ask_direction_return_bps
   Future ask versus delayed-entry ask.

2. direction_label
   -1 = ask decreased materially
    0 = no significant movement
    1 = ask increased materially

3. long_return_bps
   Buy at delayed-entry ask and sell at future bid.

4. short_return_bps
   Sell at delayed-entry bid and buy back at future ask.

5. executable_label
   -1 = profitable short after spread/slippage
    0 = neither direction clears the threshold
    1 = profitable long after spread/slippage

Each invocation processes exactly one Parquet file selected by --index.
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
OUTPUT_DIR = Path("aapl_xnas_mbp10_labels")

PX_NULL_THRESHOLD = 1e9

DEFAULT_LATENCY_TICKS = 1_000
DEFAULT_HORIZON_MINUTES = 60
DEFAULT_MOVE_THRESHOLD_BPS = 5.0
DEFAULT_SLIPPAGE_BPS = 0.0


# -----------------------------------------------------------------------------
# Arguments
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one-hour tick labels for one Parquet file "
            "selected by index."
        )
    )

    parser.add_argument(
        "--index",
        type=int,
        required=True,
        help=(
            "Zero-based index of the input Parquet file after "
            "alphabetically sorting the input directory."
        ),
    )

    parser.add_argument(
        "--latency-ticks",
        type=int,
        default=DEFAULT_LATENCY_TICKS,
        help=(
            "Number of events between the feature tick and assumed "
            f"execution. Default: {DEFAULT_LATENCY_TICKS}"
        ),
    )

    parser.add_argument(
        "--horizon-minutes",
        type=float,
        default=DEFAULT_HORIZON_MINUTES,
        help=(
            "Forecast horizon measured from the delayed entry tick. "
            f"Default: {DEFAULT_HORIZON_MINUTES}"
        ),
    )

    parser.add_argument(
        "--move-threshold-bps",
        type=float,
        default=DEFAULT_MOVE_THRESHOLD_BPS,
        help=(
            "Absolute return below which the label is no movement. "
            f"Default: {DEFAULT_MOVE_THRESHOLD_BPS} bps"
        ),
    )

    parser.add_argument(
        "--slippage-bps",
        type=float,
        default=DEFAULT_SLIPPAGE_BPS,
        help=(
            "Additional one-way slippage applied to executable "
            f"entry and exit prices. Default: {DEFAULT_SLIPPAGE_BPS}"
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing label file.",
    )

    return parser.parse_args()


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------

def validate_args(args: argparse.Namespace) -> None:
    if args.index < 0:
        raise ValueError(
            f"--index must be nonnegative, received {args.index}"
        )

    if args.latency_ticks < 0:
        raise ValueError(
            "--latency-ticks must be nonnegative."
        )

    if args.horizon_minutes <= 0:
        raise ValueError(
            "--horizon-minutes must be greater than zero."
        )

    if args.move_threshold_bps < 0:
        raise ValueError(
            "--move-threshold-bps must be nonnegative."
        )

    if args.slippage_bps < 0:
        raise ValueError(
            "--slippage-bps must be nonnegative."
        )


def valid_price(price: np.ndarray) -> np.ndarray:
    return (
        np.isfinite(price)
        & (price > 0)
        & (price < PX_NULL_THRESHOLD)
    )


# -----------------------------------------------------------------------------
# Label calculation
# -----------------------------------------------------------------------------

def make_labels(
    df: pd.DataFrame,
    latency_ticks: int,
    horizon_minutes: float,
    move_threshold_bps: float,
    slippage_bps: float,
) -> pd.DataFrame:
    """
    Construct labels for every event in df.

    Invalid rows are retained but have:

        label_valid = False
        direction_label = NaN
        executable_label = NaN

    Rows near the end of the day are invalid when there is no complete
    one-hour future horizon.
    """
    required_columns = {
        "bid_px_00",
        "ask_px_00",
    }

    missing_columns = sorted(
        required_columns - set(df.columns)
    )

    if missing_columns:
        raise KeyError(
            f"Missing required columns: {missing_columns}"
        )

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            "Input Parquet index must be a DatetimeIndex."
        )

    # Match the stable sorting used by the feature script.
    df = df.sort_index(kind="stable")

    if not df.index.is_monotonic_increasing:
        raise ValueError(
            "Timestamps are not monotonically increasing after sorting."
        )

    n_rows = len(df)

    if n_rows == 0:
        raise ValueError("Input file contains no rows.")

    timestamps_ns = df.index.asi8

    bid = df["bid_px_00"].to_numpy(
        dtype=np.float64,
        copy=False,
    )

    ask = df["ask_px_00"].to_numpy(
        dtype=np.float64,
        copy=False,
    )

    event_index = np.arange(
        n_rows,
        dtype=np.int64,
    )

    # -------------------------------------------------------------------------
    # Delayed entry index
    # -------------------------------------------------------------------------

    entry_index = event_index + latency_ticks

    has_entry = entry_index < n_rows

    # Use -1 as the invalid-index sentinel.
    stored_entry_index = np.full(
        n_rows,
        -1,
        dtype=np.int64,
    )

    stored_exit_index = np.full(
        n_rows,
        -1,
        dtype=np.int64,
    )

    stored_entry_time_ns = np.full(
        n_rows,
        -1,
        dtype=np.int64,
    )

    stored_target_exit_time_ns = np.full(
        n_rows,
        -1,
        dtype=np.int64,
    )

    stored_exit_time_ns = np.full(
        n_rows,
        -1,
        dtype=np.int64,
    )

    source_rows_with_entry = event_index[has_entry]
    valid_entry_indices = entry_index[has_entry]

    stored_entry_index[has_entry] = valid_entry_indices
    stored_entry_time_ns[has_entry] = timestamps_ns[
        valid_entry_indices
    ]

    # -------------------------------------------------------------------------
    # One-hour target measured from delayed entry
    # -------------------------------------------------------------------------

    horizon_ns = pd.Timedelta(
        minutes=horizon_minutes
    ).value

    target_exit_time_ns = (
        timestamps_ns[valid_entry_indices]
        + horizon_ns
    )

    stored_target_exit_time_ns[
        source_rows_with_entry
    ] = target_exit_time_ns

    candidate_exit_indices = np.searchsorted(
        timestamps_ns,
        target_exit_time_ns,
        side="left",
    )

    has_exit = candidate_exit_indices < n_rows

    source_rows_with_exit = source_rows_with_entry[
        has_exit
    ]

    entry_indices_with_exit = valid_entry_indices[
        has_exit
    ]

    exit_indices_with_exit = candidate_exit_indices[
        has_exit
    ]

    target_times_with_exit = target_exit_time_ns[
        has_exit
    ]

    # -------------------------------------------------------------------------
    # Do not accidentally cross into another trading day
    # -------------------------------------------------------------------------

    normalized_day_ns = df.index.normalize().asi8

    same_trading_day = (
        normalized_day_ns[entry_indices_with_exit]
        == normalized_day_ns[exit_indices_with_exit]
    )

    source_rows_candidate = source_rows_with_exit[
        same_trading_day
    ]

    entry_indices_candidate = entry_indices_with_exit[
        same_trading_day
    ]

    exit_indices_candidate = exit_indices_with_exit[
        same_trading_day
    ]

    target_times_candidate = target_times_with_exit[
        same_trading_day
    ]

    # -------------------------------------------------------------------------
    # Price validity
    # -------------------------------------------------------------------------

    entry_bid_candidate = bid[entry_indices_candidate]
    entry_ask_candidate = ask[entry_indices_candidate]

    exit_bid_candidate = bid[exit_indices_candidate]
    exit_ask_candidate = ask[exit_indices_candidate]

    prices_are_valid = (
        valid_price(entry_bid_candidate)
        & valid_price(entry_ask_candidate)
        & valid_price(exit_bid_candidate)
        & valid_price(exit_ask_candidate)
        & (entry_ask_candidate >= entry_bid_candidate)
        & (exit_ask_candidate >= exit_bid_candidate)
    )

    valid_source_rows = source_rows_candidate[
        prices_are_valid
    ]

    valid_entry_indices = entry_indices_candidate[
        prices_are_valid
    ]

    valid_exit_indices = exit_indices_candidate[
        prices_are_valid
    ]

    valid_target_times = target_times_candidate[
        prices_are_valid
    ]

    entry_bid = entry_bid_candidate[
        prices_are_valid
    ]

    entry_ask = entry_ask_candidate[
        prices_are_valid
    ]

    exit_bid = exit_bid_candidate[
        prices_are_valid
    ]

    exit_ask = exit_ask_candidate[
        prices_are_valid
    ]

    # -------------------------------------------------------------------------
    # Store valid index and timing information
    # -------------------------------------------------------------------------

    stored_entry_index[
        valid_source_rows
    ] = valid_entry_indices

    stored_exit_index[
        valid_source_rows
    ] = valid_exit_indices

    stored_entry_time_ns[
        valid_source_rows
    ] = timestamps_ns[valid_entry_indices]

    stored_target_exit_time_ns[
        valid_source_rows
    ] = valid_target_times

    stored_exit_time_ns[
        valid_source_rows
    ] = timestamps_ns[valid_exit_indices]

    label_valid = np.zeros(
        n_rows,
        dtype=bool,
    )

    label_valid[valid_source_rows] = True

    # -------------------------------------------------------------------------
    # Allocate outputs
    # -------------------------------------------------------------------------

    entry_bid_output = np.full(
        n_rows,
        np.nan,
        dtype=np.float32,
    )

    entry_ask_output = np.full(
        n_rows,
        np.nan,
        dtype=np.float32,
    )

    exit_bid_output = np.full(
        n_rows,
        np.nan,
        dtype=np.float32,
    )

    exit_ask_output = np.full(
        n_rows,
        np.nan,
        dtype=np.float32,
    )

    ask_direction_return_bps = np.full(
        n_rows,
        np.nan,
        dtype=np.float32,
    )

    long_return_bps = np.full(
        n_rows,
        np.nan,
        dtype=np.float32,
    )

    short_return_bps = np.full(
        n_rows,
        np.nan,
        dtype=np.float32,
    )

    direction_label = np.full(
        n_rows,
        np.nan,
        dtype=np.float32,
    )

    executable_label = np.full(
        n_rows,
        np.nan,
        dtype=np.float32,
    )

    horizon_overshoot_ms = np.full(
        n_rows,
        np.nan,
        dtype=np.float32,
    )

    # -------------------------------------------------------------------------
    # Ask-to-ask directional return
    # -------------------------------------------------------------------------

    valid_ask_direction_return = (
        10_000.0
        * np.log(exit_ask / entry_ask)
    )

    valid_direction_label = np.zeros(
        len(valid_source_rows),
        dtype=np.int8,
    )

    valid_direction_label[
        valid_ask_direction_return
        > move_threshold_bps
    ] = 1

    valid_direction_label[
        valid_ask_direction_return
        < -move_threshold_bps
    ] = -1

    # -------------------------------------------------------------------------
    # Executable long and short returns
    # -------------------------------------------------------------------------

    slippage_fraction = (
        slippage_bps / 10_000.0
    )

    # Long:
    # Buy at ask plus adverse entry slippage.
    # Sell at bid minus adverse exit slippage.
    executable_long_entry = (
        entry_ask
        * (1.0 + slippage_fraction)
    )

    executable_long_exit = (
        exit_bid
        * (1.0 - slippage_fraction)
    )

    valid_long_return_bps = (
        10_000.0
        * np.log(
            executable_long_exit
            / executable_long_entry
        )
    )

    # Short:
    # Sell at bid minus adverse entry slippage.
    # Buy back at ask plus adverse exit slippage.
    executable_short_entry = (
        entry_bid
        * (1.0 - slippage_fraction)
    )

    executable_short_exit = (
        exit_ask
        * (1.0 + slippage_fraction)
    )

    valid_short_return_bps = (
        10_000.0
        * np.log(
            executable_short_entry
            / executable_short_exit
        )
    )

    # Choose whichever executable direction produces the larger return,
    # but only when that return exceeds the no-movement threshold.
    valid_executable_label = np.zeros(
        len(valid_source_rows),
        dtype=np.int8,
    )

    best_executable_return = np.maximum(
        valid_long_return_bps,
        valid_short_return_bps,
    )

    tradeable = (
        best_executable_return
        > move_threshold_bps
    )

    choose_long = (
        valid_long_return_bps
        >= valid_short_return_bps
    )

    valid_executable_label[
        tradeable & choose_long
    ] = 1

    valid_executable_label[
        tradeable & ~choose_long
    ] = -1

    # -------------------------------------------------------------------------
    # Horizon timing error
    # -------------------------------------------------------------------------

    valid_horizon_overshoot_ms = (
        timestamps_ns[valid_exit_indices]
        - valid_target_times
    ) / 1_000_000.0

    # -------------------------------------------------------------------------
    # Write valid values into full-length arrays
    # -------------------------------------------------------------------------

    entry_bid_output[
        valid_source_rows
    ] = entry_bid.astype(np.float32)

    entry_ask_output[
        valid_source_rows
    ] = entry_ask.astype(np.float32)

    exit_bid_output[
        valid_source_rows
    ] = exit_bid.astype(np.float32)

    exit_ask_output[
        valid_source_rows
    ] = exit_ask.astype(np.float32)

    ask_direction_return_bps[
        valid_source_rows
    ] = valid_ask_direction_return.astype(np.float32)

    long_return_bps[
        valid_source_rows
    ] = valid_long_return_bps.astype(np.float32)

    short_return_bps[
        valid_source_rows
    ] = valid_short_return_bps.astype(np.float32)

    direction_label[
        valid_source_rows
    ] = valid_direction_label.astype(np.float32)

    executable_label[
        valid_source_rows
    ] = valid_executable_label.astype(np.float32)

    horizon_overshoot_ms[
        valid_source_rows
    ] = valid_horizon_overshoot_ms.astype(np.float32)

    # -------------------------------------------------------------------------
    # Output DataFrame
    # -------------------------------------------------------------------------

    labels = pd.DataFrame(
        {
            # Positional identifier. This is safer than joining solely
            # on timestamp because multiple ticks may share a timestamp.
            "event_index": event_index,

            "entry_event_index": stored_entry_index,
            "exit_event_index": stored_exit_index,

            # Nanoseconds since Unix epoch. -1 means unavailable.
            "entry_time_ns": stored_entry_time_ns,
            "target_exit_time_ns": stored_target_exit_time_ns,
            "exit_time_ns": stored_exit_time_ns,

            "horizon_overshoot_ms": horizon_overshoot_ms,

            "entry_bid": entry_bid_output,
            "entry_ask": entry_ask_output,
            "exit_bid": exit_bid_output,
            "exit_ask": exit_ask_output,

            "ask_direction_return_bps": (
                ask_direction_return_bps
            ),

            "long_return_bps": long_return_bps,
            "short_return_bps": short_return_bps,

            # Values are -1, 0, 1, or NaN.
            "direction_label": direction_label,
            "executable_label": executable_label,

            "label_valid": label_valid,
        },
        index=df.index,
    )

    return labels


# -----------------------------------------------------------------------------
# File processing
# -----------------------------------------------------------------------------

def process_file(
    input_path: Path,
    output_path: Path,
    latency_ticks: int,
    horizon_minutes: float,
    move_threshold_bps: float,
    slippage_bps: float,
) -> None:
    start_time = time.perf_counter()

    print(f"Reading: {input_path}")

    # Only load the two price columns needed for labels.
    # The stored timestamp index is also loaded.
    df = pd.read_parquet(
        input_path,
        columns=[
            "bid_px_00",
            "ask_px_00",
        ],
    )

    print(f"Input ticks: {len(df):,}")

    labels = make_labels(
        df=df,
        latency_ticks=latency_ticks,
        horizon_minutes=horizon_minutes,
        move_threshold_bps=move_threshold_bps,
        slippage_bps=slippage_bps,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    labels.to_parquet(
        output_path,
        compression="zstd",
        index=True,
    )

    elapsed = time.perf_counter() - start_time

    valid = labels["label_valid"]

    print()
    print(f"Total rows : {len(labels):,}")
    print(f"Valid rows : {valid.sum():,}")
    print(f"Invalid    : {(~valid).sum():,}")

    if valid.any():
        direction_counts = (
            labels.loc[valid, "direction_label"]
            .astype(np.int8)
            .value_counts()
            .sort_index()
        )

        executable_counts = (
            labels.loc[valid, "executable_label"]
            .astype(np.int8)
            .value_counts()
            .sort_index()
        )

        print()
        print("Direction-label counts:")
        print(direction_counts.to_string())

        print()
        print("Executable-label counts:")
        print(executable_counts.to_string())

        print()
        print(
            "Median horizon overshoot: "
            f"{labels.loc[valid, 'horizon_overshoot_ms'].median():.3f} ms"
        )

    print()
    print(
        f"Output size : "
        f"{output_path.stat().st_size / 1e6:.1f} MB"
    )
    print(f"Elapsed     : {elapsed / 60:.2f} minutes")
    print(f"Saved       : {output_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    validate_args(args)

    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Input directory does not exist: {INPUT_DIR}"
        )

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
            f"There are {len(input_files)} input files, so valid "
            f"indices are 0 through {len(input_files) - 1}."
        )

    input_path = input_files[args.index]
    output_path = OUTPUT_DIR / input_path.name

    print("=" * 80)
    print(
        f"Slurm job ID       : "
        f"{os.environ.get('SLURM_JOB_ID', 'manual')}"
    )
    print(
        f"Slurm array task   : "
        f"{os.environ.get('SLURM_ARRAY_TASK_ID', 'manual')}"
    )
    print(f"Requested index    : {args.index}")
    print(
        f"Selected file      : "
        f"{args.index + 1} of {len(input_files)}"
    )
    print(f"Input              : {input_path}")
    print(f"Output             : {output_path}")
    print(f"Latency            : {args.latency_ticks:,} ticks")
    print(f"Horizon            : {args.horizon_minutes:g} minutes")
    print(
        f"Movement threshold : "
        f"{args.move_threshold_bps:g} bps"
    )
    print(f"Slippage           : {args.slippage_bps:g} bps per side")
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
        latency_ticks=args.latency_ticks,
        horizon_minutes=args.horizon_minutes,
        move_threshold_bps=args.move_threshold_bps,
        slippage_bps=args.slippage_bps,
    )

    print()
    print(
        f"Task index {args.index} completed successfully."
    )


if __name__ == "__main__":
    main()