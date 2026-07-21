"""
04_train_tick_models.py

Chronologically benchmarks several multiclass models on the tick-level
feature and label Parquet files.

Classes
-------
-1 : down
 0 : neutral / no significant movement
 1 : up

Core models
-----------
- dummy
- logistic
- random_forest
- hist_gb

Optional models
---------------
- xgboost
- lightgbm

Important
---------
The feature and label directories must contain matching filenames.

Example
-------
python 04_train_tick_models.py \
    --label-column direction_label \
    --rows-per-file 20000 \
    --models dummy logistic random_forest hist_gb xgboost lightgbm
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


# =============================================================================
# Configuration
# =============================================================================

FEATURE_DIR = Path("aapl_xnas_mbp10_tick_features")
LABEL_DIR = Path("aapl_xnas_mbp10_labels")
OUTPUT_DIR = Path("aapl_xnas_mbp10_model_results")

CLASS_VALUES = np.array([-1, 0, 1], dtype=np.int8)

CLASS_TO_ENCODED = {
    -1: 0,
    0: 1,
    1: 2,
}

ENCODED_TO_CLASS = {
    0: -1,
    1: 0,
    2: 1,
}

CLASS_NAMES = {
    0: "down",
    1: "neutral",
    2: "up",
}

SUPPORTED_MODELS = {
    "dummy",
    "logistic",
    "random_forest",
    "hist_gb",
    "xgboost",
    "lightgbm",
}

# These features identify rows rather than describe the market.
ALWAYS_DROP_COLUMNS = {
    "event_index",
}

# Absolute AAPL price can let a model identify historical regimes rather than
# learn stationary market structure. Normalized bps features remain available.
ABSOLUTE_PRICE_COLUMNS = {
    "mid",
    "microprice",
    "spread",
}


# =============================================================================
# Arguments
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train and compare multiclass models on tick-level "
            "order-book features."
        )
    )

    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=FEATURE_DIR,
    )

    parser.add_argument(
        "--label-dir",
        type=Path,
        default=LABEL_DIR,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
    )

    parser.add_argument(
        "--label-column",
        choices=[
            "direction_label",
            "executable_label",
        ],
        default="direction_label",
        help=(
            "Target column from the label files. "
            "Default: direction_label."
        ),
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "dummy",
            "logistic",
            "random_forest",
            "hist_gb",
            "xgboost",
            "lightgbm",
        ],
        help="Models to benchmark.",
    )

    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.70,
        help="Chronological fraction of files used for training.",
    )

    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.15,
        help="Chronological fraction of files used for validation.",
    )

    parser.add_argument(
        "--rows-per-file",
        type=int,
        default=20_000,
        help=(
            "Maximum valid rows sampled from each daily file. "
            "Use 0 to retain every valid tick. Default: 20000."
        ),
    )

    parser.add_argument(
        "--row-stride",
        type=int,
        default=1,
        help=(
            "Use every Nth valid tick before optional sampling. "
            "Default: 1, meaning every valid tick is eligible."
        ),
    )

    parser.add_argument(
        "--keep-absolute-prices",
        action="store_true",
        help="Keep raw mid, microprice, and spread features.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--n-jobs",
        type=int,
        default=int(
            os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow an existing output directory to be reused.",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    unknown_models = set(args.models) - SUPPORTED_MODELS

    if unknown_models:
        raise ValueError(
            f"Unknown models: {sorted(unknown_models)}. "
            f"Supported models: {sorted(SUPPORTED_MODELS)}"
        )

    if not 0 < args.train_fraction < 1:
        raise ValueError("--train-fraction must be between 0 and 1.")

    if not 0 < args.validation_fraction < 1:
        raise ValueError(
            "--validation-fraction must be between 0 and 1."
        )

    if args.train_fraction + args.validation_fraction >= 1:
        raise ValueError(
            "Train and validation fractions must sum to less than 1."
        )

    if args.rows_per_file < 0:
        raise ValueError("--rows-per-file cannot be negative.")

    if args.row_stride < 1:
        raise ValueError("--row-stride must be at least 1.")


# =============================================================================
# File discovery and chronological split
# =============================================================================

def discover_file_pairs(
    feature_dir: Path,
    label_dir: Path,
) -> list[tuple[Path, Path]]:
    if not feature_dir.exists():
        raise FileNotFoundError(
            f"Feature directory does not exist: {feature_dir}"
        )

    if not label_dir.exists():
        raise FileNotFoundError(
            f"Label directory does not exist: {label_dir}"
        )

    feature_files = {
        path.name: path
        for path in feature_dir.glob("*.parquet")
    }

    label_files = {
        path.name: path
        for path in label_dir.glob("*.parquet")
    }

    if not feature_files:
        raise FileNotFoundError(
            f"No feature Parquet files found in {feature_dir}"
        )

    if not label_files:
        raise FileNotFoundError(
            f"No label Parquet files found in {label_dir}"
        )

    missing_labels = sorted(
        set(feature_files) - set(label_files)
    )

    missing_features = sorted(
        set(label_files) - set(feature_files)
    )

    if missing_labels:
        raise FileNotFoundError(
            "Feature files without matching label files:\n"
            + "\n".join(missing_labels[:20])
        )

    if missing_features:
        warnings.warn(
            f"{len(missing_features)} label files have no matching "
            "feature file and will be ignored."
        )

    # The Databento filenames contain YYYYMMDD, so alphabetical sorting
    # is chronological for these files.
    common_names = sorted(
        set(feature_files) & set(label_files)
    )

    return [
        (
            feature_files[name],
            label_files[name],
        )
        for name in common_names
    ]


def chronological_split(
    pairs: list[tuple[Path, Path]],
    train_fraction: float,
    validation_fraction: float,
) -> tuple[
    list[tuple[Path, Path]],
    list[tuple[Path, Path]],
    list[tuple[Path, Path]],
]:
    n_files = len(pairs)

    if n_files < 10:
        raise ValueError(
            "At least 10 matched daily files are recommended for a "
            "chronological train/validation/test split."
        )

    train_end = int(n_files * train_fraction)
    validation_end = train_end + int(
        n_files * validation_fraction
    )

    train_pairs = pairs[:train_end]
    validation_pairs = pairs[train_end:validation_end]
    test_pairs = pairs[validation_end:]

    if not train_pairs or not validation_pairs or not test_pairs:
        raise ValueError(
            "The requested split produced an empty partition."
        )

    return train_pairs, validation_pairs, test_pairs


# =============================================================================
# Data loading
# =============================================================================

def determine_feature_columns(
    features: pd.DataFrame,
    keep_absolute_prices: bool,
) -> list[str]:
    numeric_columns = list(
        features.select_dtypes(
            include=[
                "number",
                "bool",
            ]
        ).columns
    )

    drop_columns = set(ALWAYS_DROP_COLUMNS)

    if not keep_absolute_prices:
        drop_columns |= ABSOLUTE_PRICE_COLUMNS

    selected = [
        column
        for column in numeric_columns
        if column not in drop_columns
    ]

    if not selected:
        raise ValueError("No usable numeric feature columns found.")

    return selected


def encode_labels(
    raw_labels: np.ndarray,
) -> np.ndarray:
    raw_labels = np.asarray(raw_labels)

    rounded = np.rint(raw_labels).astype(np.int8)

    unexpected = set(np.unique(rounded)) - {-1, 0, 1}

    if unexpected:
        raise ValueError(
            f"Unexpected label values found: {sorted(unexpected)}"
        )

    encoded = np.empty(
        len(rounded),
        dtype=np.int8,
    )

    encoded[rounded == -1] = 0
    encoded[rounded == 0] = 1
    encoded[rounded == 1] = 2

    return encoded


def sample_positions(
    valid_positions: np.ndarray,
    rows_per_file: int,
    row_stride: int,
    rng: np.random.Generator,
) -> np.ndarray:
    positions = valid_positions[::row_stride]

    if (
        rows_per_file > 0
        and len(positions) > rows_per_file
    ):
        selected = rng.choice(
            positions,
            size=rows_per_file,
            replace=False,
        )

        # Preserve chronological order inside each day.
        positions = np.sort(selected)

    return positions


def validate_row_alignment(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    filename: str,
) -> None:
    if len(features) != len(labels):
        raise ValueError(
            f"Feature/label row-count mismatch for {filename}: "
            f"{len(features):,} versus {len(labels):,}"
        )

    if (
        "event_index" in features.columns
        and "event_index" in labels.columns
    ):
        feature_index = features[
            "event_index"
        ].to_numpy()

        label_index = labels[
            "event_index"
        ].to_numpy()

        if not np.array_equal(
            feature_index,
            label_index,
        ):
            raise ValueError(
                f"event_index alignment failed for {filename}"
            )

    # Duplicate event timestamps are possible. Compare positionally rather
    # than joining solely on the timestamp.
    if not features.index.equals(labels.index):
        raise ValueError(
            f"Timestamp-index alignment failed for {filename}"
        )


def load_dataset_partition(
    pairs: list[tuple[Path, Path]],
    partition_name: str,
    label_column: str,
    rows_per_file: int,
    row_stride: int,
    seed: int,
    keep_absolute_prices: bool,
    feature_columns: list[str] | None = None,
) -> tuple[
    pd.DataFrame,
    np.ndarray,
    pd.DataFrame,
    list[str],
]:
    feature_chunks: list[pd.DataFrame] = []
    label_chunks: list[np.ndarray] = []
    metadata_chunks: list[pd.DataFrame] = []

    selected_feature_columns = feature_columns

    total_valid_rows = 0
    total_selected_rows = 0

    print()
    print(f"Loading {partition_name} partition")
    print("-" * 80)

    for file_number, (feature_path, label_path) in enumerate(
        pairs
    ):
        print(
            f"[{file_number + 1:>4}/{len(pairs):<4}] "
            f"{feature_path.name}",
            flush=True,
        )

        features = pd.read_parquet(feature_path)
        labels = pd.read_parquet(label_path)

        validate_row_alignment(
            features=features,
            labels=labels,
            filename=feature_path.name,
        )

        required_label_columns = {
            "label_valid",
            label_column,
            "long_return_bps",
            "short_return_bps",
        }

        missing_label_columns = (
            required_label_columns - set(labels.columns)
        )

        if missing_label_columns:
            raise KeyError(
                f"{label_path.name} is missing label columns: "
                f"{sorted(missing_label_columns)}"
            )

        if selected_feature_columns is None:
            selected_feature_columns = determine_feature_columns(
                features=features,
                keep_absolute_prices=keep_absolute_prices,
            )

            print(
                f"Selected {len(selected_feature_columns):,} "
                "numeric feature columns."
            )

        missing_features = (
            set(selected_feature_columns)
            - set(features.columns)
        )

        if missing_features:
            raise KeyError(
                f"{feature_path.name} is missing features: "
                f"{sorted(missing_features)}"
            )

        raw_y = labels[
            label_column
        ].to_numpy(dtype=np.float64)

        valid_mask = (
            labels["label_valid"].to_numpy(dtype=bool)
            & np.isfinite(raw_y)
        )

        valid_positions = np.flatnonzero(valid_mask)

        total_valid_rows += len(valid_positions)

        rng = np.random.default_rng(
            seed + file_number
        )

        selected_positions = sample_positions(
            valid_positions=valid_positions,
            rows_per_file=rows_per_file,
            row_stride=row_stride,
            rng=rng,
        )

        total_selected_rows += len(selected_positions)

        if len(selected_positions) == 0:
            del features, labels
            gc.collect()
            continue

        X_chunk = (
            features
            .iloc[selected_positions][selected_feature_columns]
            .replace([np.inf, -np.inf], np.nan)
            .astype(np.float32)
            .reset_index(drop=True)
        )

        y_chunk = encode_labels(
            raw_y[selected_positions]
        )

        if "event_index" in labels.columns:
            event_index = labels[
                "event_index"
            ].to_numpy(dtype=np.int64)[selected_positions]
        else:
            event_index = selected_positions.astype(np.int64)

        timestamp_ns = labels.index.asi8[
            selected_positions
        ]

        metadata_chunk = pd.DataFrame(
            {
                "source_file": feature_path.name,
                "event_index": event_index,
                "timestamp_ns": timestamp_ns,
                "long_return_bps": labels[
                    "long_return_bps"
                ].to_numpy(dtype=np.float32)[selected_positions],
                "short_return_bps": labels[
                    "short_return_bps"
                ].to_numpy(dtype=np.float32)[selected_positions],
            }
        )

        feature_chunks.append(X_chunk)
        label_chunks.append(y_chunk)
        metadata_chunks.append(metadata_chunk)

        del (
            features,
            labels,
            X_chunk,
            y_chunk,
            metadata_chunk,
        )
        gc.collect()

    if selected_feature_columns is None:
        raise ValueError(
            f"No feature columns were found for {partition_name}."
        )

    if not feature_chunks:
        raise ValueError(
            f"No usable rows were loaded for {partition_name}."
        )

    X = pd.concat(
        feature_chunks,
        axis=0,
        ignore_index=True,
        copy=False,
    )

    y = np.concatenate(label_chunks)

    metadata = pd.concat(
        metadata_chunks,
        axis=0,
        ignore_index=True,
        copy=False,
    )

    print()
    print(f"{partition_name} valid rows    : {total_valid_rows:,}")
    print(f"{partition_name} selected rows : {total_selected_rows:,}")
    print(f"{partition_name} matrix shape  : {X.shape}")

    return X, y, metadata, selected_feature_columns


def remove_all_nan_features(
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    list[str],
]:
    all_nan_columns = list(
        X_train.columns[
            X_train.isna().all(axis=0)
        ]
    )

    if all_nan_columns:
        print(
            f"Dropping {len(all_nan_columns)} all-NaN "
            "training features."
        )

        X_train = X_train.drop(
            columns=all_nan_columns
        )
        X_validation = X_validation.drop(
            columns=all_nan_columns
        )
        X_test = X_test.drop(
            columns=all_nan_columns
        )

    return (
        X_train,
        X_validation,
        X_test,
        all_nan_columns,
    )


# =============================================================================
# Models
# =============================================================================

def create_models(
    model_names: list[str],
    seed: int,
    n_jobs: int,
) -> dict[str, Any]:
    models: dict[str, Any] = {}

    if "dummy" in model_names:
        models["dummy"] = DummyClassifier(
            strategy="prior",
        )

    if "logistic" in model_names:
        models["logistic"] = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                    ),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        C=0.1,
                        solver="lbfgs",
                        max_iter=300,
                        random_state=seed,
                    ),
                ),
            ]
        )

    if "random_forest" in model_names:
        models["random_forest"] = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                    ),
                ),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=18,
                        min_samples_leaf=50,
                        max_features="sqrt",
                        bootstrap=True,
                        max_samples=0.50,
                        n_jobs=n_jobs,
                        random_state=seed,
                        verbose=1,
                    ),
                ),
            ]
        )

    if "hist_gb" in model_names:
        models["hist_gb"] = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=0.05,
            max_iter=400,
            max_leaf_nodes=31,
            min_samples_leaf=100,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=seed,
            verbose=1,
        )

    if "xgboost" in model_names:
        try:
            import xgboost as xgb

            models["xgboost"] = xgb.XGBClassifier(
                objective="multi:softprob",
                num_class=3,
                n_estimators=2_000,
                learning_rate=0.03,
                max_depth=8,
                min_child_weight=25,
                subsample=0.80,
                colsample_bytree=0.80,
                reg_alpha=0.0,
                reg_lambda=1.0,
                tree_method="hist",
                eval_metric="mlogloss",
                early_stopping_rounds=75,
                n_jobs=n_jobs,
                random_state=seed,
            )

        except ImportError:
            warnings.warn(
                "xgboost is not installed; skipping XGBoost."
            )

    if "lightgbm" in model_names:
        try:
            import lightgbm as lgb

            models["lightgbm"] = lgb.LGBMClassifier(
                objective="multiclass",
                num_class=3,
                n_estimators=2_000,
                learning_rate=0.03,
                num_leaves=63,
                max_depth=-1,
                min_child_samples=100,
                subsample=0.80,
                subsample_freq=1,
                colsample_bytree=0.80,
                reg_alpha=0.0,
                reg_lambda=1.0,
                n_jobs=n_jobs,
                random_state=seed,
                verbosity=-1,
            )

        except ImportError:
            warnings.warn(
                "lightgbm is not installed; skipping LightGBM."
            )

    return models


def fit_model(
    model_name: str,
    model: Any,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_validation: pd.DataFrame,
    y_validation: np.ndarray,
    sample_weight: np.ndarray,
) -> Any:
    if model_name == "dummy":
        model.fit(
            X_train,
            y_train,
        )

    elif model_name in {
        "logistic",
        "random_forest",
    }:
        model.fit(
            X_train,
            y_train,
            classifier__sample_weight=sample_weight,
        )

    elif model_name == "hist_gb":
        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weight,
        )

    elif model_name == "xgboost":
        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weight,
            eval_set=[
                (
                    X_validation,
                    y_validation,
                )
            ],
            verbose=True,
        )

    elif model_name == "lightgbm":
        import lightgbm as lgb

        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weight,
            eval_set=[
                (
                    X_validation,
                    y_validation,
                )
            ],
            eval_metric="multi_logloss",
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=75,
                    verbose=True,
                ),
                lgb.log_evaluation(period=25),
            ],
        )

    else:
        raise ValueError(
            f"Unhandled model: {model_name}"
        )

    return model


# =============================================================================
# Prediction and evaluation
# =============================================================================

def aligned_predict_proba(
    model: Any,
    X: pd.DataFrame,
) -> np.ndarray:
    raw_probabilities = model.predict_proba(X)

    model_classes = np.asarray(
        model.classes_,
        dtype=np.int64,
    )

    probabilities = np.zeros(
        (
            len(X),
            3,
        ),
        dtype=np.float64,
    )

    probabilities[:, model_classes] = raw_probabilities

    return probabilities


def evaluate_predictions(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    metadata: pd.DataFrame,
    training_seconds: float,
    prediction_seconds: float,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    np.ndarray,
    pd.DataFrame,
]:
    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    balanced_accuracy = balanced_accuracy_score(
        y_true,
        y_pred,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    multiclass_log_loss = log_loss(
        y_true,
        probabilities,
        labels=[0, 1, 2],
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=[
            "down",
            "neutral",
            "up",
        ],
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1, 2],
    )

    original_predictions = CLASS_VALUES[
        y_pred
    ]

    signal_mask = original_predictions != 0

    long_returns = metadata[
        "long_return_bps"
    ].to_numpy(dtype=np.float64)

    short_returns = metadata[
        "short_return_bps"
    ].to_numpy(dtype=np.float64)

    selected_return = np.where(
        original_predictions == 1,
        long_returns,
        np.where(
            original_predictions == -1,
            short_returns,
            0.0,
        ),
    )

    valid_signal_return = (
        signal_mask
        & np.isfinite(selected_return)
    )

    signal_rate = float(
        np.mean(signal_mask)
    )

    if valid_signal_return.any():
        mean_signal_return_bps = float(
            np.mean(
                selected_return[valid_signal_return]
            )
        )

        median_signal_return_bps = float(
            np.median(
                selected_return[valid_signal_return]
            )
        )

        signal_win_rate = float(
            np.mean(
                selected_return[valid_signal_return] > 0
            )
        )

        directional_accuracy_on_signals = float(
            np.mean(
                y_pred[signal_mask]
                == y_true[signal_mask]
            )
        )

    else:
        mean_signal_return_bps = np.nan
        median_signal_return_bps = np.nan
        signal_win_rate = np.nan
        directional_accuracy_on_signals = np.nan

    metrics = {
        "model": model_name,
        "n_test_rows": int(len(y_true)),
        "training_seconds": training_seconds,
        "prediction_seconds": prediction_seconds,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "log_loss": multiclass_log_loss,
        "signal_rate": signal_rate,
        "directional_accuracy_on_signals": (
            directional_accuracy_on_signals
        ),
        "mean_signal_return_bps": mean_signal_return_bps,
        "median_signal_return_bps": median_signal_return_bps,
        "signal_win_rate": signal_win_rate,
        "down_recall": report["down"]["recall"],
        "neutral_recall": report["neutral"]["recall"],
        "up_recall": report["up"]["recall"],
        "down_f1": report["down"]["f1-score"],
        "neutral_f1": report["neutral"]["f1-score"],
        "up_f1": report["up"]["f1-score"],
    }

    predictions = metadata.copy()

    predictions["y_true"] = CLASS_VALUES[
        y_true
    ]

    predictions["y_pred"] = original_predictions

    predictions["probability_down"] = probabilities[:, 0].astype(
        np.float32
    )

    predictions["probability_neutral"] = probabilities[:, 1].astype(
        np.float32
    )

    predictions["probability_up"] = probabilities[:, 2].astype(
        np.float32
    )

    predictions["selected_return_bps"] = selected_return.astype(
        np.float32
    )

    return metrics, report, matrix, predictions


# =============================================================================
# Model diagnostics
# =============================================================================

def unwrap_estimator(
    model: Any,
) -> Any:
    if isinstance(model, Pipeline):
        return model.named_steps[
            "classifier"
        ]

    return model


def save_model_diagnostics(
    model_name: str,
    model: Any,
    feature_columns: list[str],
    model_dir: Path,
) -> None:
    estimator = unwrap_estimator(model)

    if hasattr(estimator, "feature_importances_"):
        importance = np.asarray(
            estimator.feature_importances_,
            dtype=np.float64,
        )

        if len(importance) == len(feature_columns):
            importance_frame = pd.DataFrame(
                {
                    "feature": feature_columns,
                    "importance": importance,
                }
            ).sort_values(
                "importance",
                ascending=False,
            )

            importance_frame.to_csv(
                model_dir / "feature_importance.csv",
                index=False,
            )

    if hasattr(estimator, "coef_"):
        coefficients = np.asarray(
            estimator.coef_,
            dtype=np.float64,
        )

        if (
            coefficients.ndim == 2
            and coefficients.shape[1]
            == len(feature_columns)
        ):
            rows = []

            for class_index in range(
                coefficients.shape[0]
            ):
                class_value = ENCODED_TO_CLASS[
                    class_index
                ]

                for feature, coefficient in zip(
                    feature_columns,
                    coefficients[class_index],
                ):
                    rows.append(
                        {
                            "class": class_value,
                            "feature": feature,
                            "coefficient": coefficient,
                        }
                    )

            coefficient_frame = pd.DataFrame(rows)

            coefficient_frame[
                "absolute_coefficient"
            ] = coefficient_frame[
                "coefficient"
            ].abs()

            coefficient_frame = coefficient_frame.sort_values(
                [
                    "class",
                    "absolute_coefficient",
                ],
                ascending=[
                    True,
                    False,
                ],
            )

            coefficient_frame.to_csv(
                model_dir / "coefficients.csv",
                index=False,
            )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()
    validate_args(args)

    if args.output_dir.exists() and not args.overwrite:
        existing_contents = list(
            args.output_dir.iterdir()
        )

        if existing_contents:
            raise FileExistsError(
                f"Output directory is not empty: {args.output_dir}\n"
                "Use --overwrite to reuse it."
            )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pairs = discover_file_pairs(
        feature_dir=args.feature_dir,
        label_dir=args.label_dir,
    )

    (
        train_pairs,
        validation_pairs,
        test_pairs,
    ) = chronological_split(
        pairs=pairs,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
    )

    print("=" * 80)
    print(f"Matched daily files : {len(pairs)}")
    print(f"Training files      : {len(train_pairs)}")
    print(f"Validation files    : {len(validation_pairs)}")
    print(f"Test files          : {len(test_pairs)}")
    print(f"Target              : {args.label_column}")
    print(f"Rows per file       : {args.rows_per_file}")
    print(f"Row stride          : {args.row_stride}")
    print(f"Models              : {args.models}")
    print(f"CPUs                 : {args.n_jobs}")
    print("=" * 80)

    split_information = {
        "target": args.label_column,
        "train_files": [
            pair[0].name
            for pair in train_pairs
        ],
        "validation_files": [
            pair[0].name
            for pair in validation_pairs
        ],
        "test_files": [
            pair[0].name
            for pair in test_pairs
        ],
    }

    with (
        args.output_dir / "chronological_split.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            split_information,
            file,
            indent=2,
        )

    X_train, y_train, train_metadata, feature_columns = (
        load_dataset_partition(
            pairs=train_pairs,
            partition_name="training",
            label_column=args.label_column,
            rows_per_file=args.rows_per_file,
            row_stride=args.row_stride,
            seed=args.seed,
            keep_absolute_prices=args.keep_absolute_prices,
            feature_columns=None,
        )
    )

    X_validation, y_validation, validation_metadata, _ = (
        load_dataset_partition(
            pairs=validation_pairs,
            partition_name="validation",
            label_column=args.label_column,
            rows_per_file=args.rows_per_file,
            row_stride=args.row_stride,
            seed=args.seed + 100_000,
            keep_absolute_prices=args.keep_absolute_prices,
            feature_columns=feature_columns,
        )
    )

    X_test, y_test, test_metadata, _ = (
        load_dataset_partition(
            pairs=test_pairs,
            partition_name="test",
            label_column=args.label_column,
            rows_per_file=args.rows_per_file,
            row_stride=args.row_stride,
            seed=args.seed + 200_000,
            keep_absolute_prices=args.keep_absolute_prices,
            feature_columns=feature_columns,
        )
    )

    (
        X_train,
        X_validation,
        X_test,
        all_nan_columns,
    ) = remove_all_nan_features(
        X_train=X_train,
        X_validation=X_validation,
        X_test=X_test,
    )

    feature_columns = list(
        X_train.columns
    )

    with (
        args.output_dir / "feature_columns.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            feature_columns,
            file,
            indent=2,
        )

    with (
        args.output_dir / "dropped_all_nan_columns.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            all_nan_columns,
            file,
            indent=2,
        )

    print()
    print("Encoded class counts")
    print("--------------------")

    for name, y in [
        ("train", y_train),
        ("validation", y_validation),
        ("test", y_test),
    ]:
        counts = np.bincount(
            y,
            minlength=3,
        )

        print(
            f"{name:>10}: "
            f"down={counts[0]:,}, "
            f"neutral={counts[1]:,}, "
            f"up={counts[2]:,}"
        )

    train_classes = set(
        np.unique(y_train)
    )

    if train_classes != {0, 1, 2}:
        raise ValueError(
            "All three classes must be represented in training. "
            f"Found encoded classes: {sorted(train_classes)}"
        )

    sample_weight = compute_sample_weight(
        class_weight="balanced",
        y=y_train,
    ).astype(np.float32)

    models = create_models(
        model_names=args.models,
        seed=args.seed,
        n_jobs=args.n_jobs,
    )

    if not models:
        raise RuntimeError(
            "No requested models are available."
        )

    summary_rows = []

    for model_name, model in models.items():
        print()
        print("=" * 80)
        print(f"Training model: {model_name}")
        print("=" * 80)

        model_dir = (
            args.output_dir / model_name
        )

        model_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        training_start = time.perf_counter()

        try:
            model = fit_model(
                model_name=model_name,
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_validation=X_validation,
                y_validation=y_validation,
                sample_weight=sample_weight,
            )

            training_seconds = (
                time.perf_counter()
                - training_start
            )

            prediction_start = time.perf_counter()

            probabilities = aligned_predict_proba(
                model=model,
                X=X_test,
            )

            y_pred = np.argmax(
                probabilities,
                axis=1,
            ).astype(np.int8)

            prediction_seconds = (
                time.perf_counter()
                - prediction_start
            )

            (
                metrics,
                report,
                matrix,
                predictions,
            ) = evaluate_predictions(
                model_name=model_name,
                y_true=y_test,
                y_pred=y_pred,
                probabilities=probabilities,
                metadata=test_metadata,
                training_seconds=training_seconds,
                prediction_seconds=prediction_seconds,
            )

            summary_rows.append(metrics)

            joblib.dump(
                model,
                model_dir / "model.joblib",
                compress=3,
            )

            predictions.to_parquet(
                model_dir / "test_predictions.parquet",
                compression="zstd",
                index=False,
            )

            pd.DataFrame(
                matrix,
                index=[
                    "actual_down",
                    "actual_neutral",
                    "actual_up",
                ],
                columns=[
                    "predicted_down",
                    "predicted_neutral",
                    "predicted_up",
                ],
            ).to_csv(
                model_dir / "confusion_matrix.csv"
            )

            with (
                model_dir / "classification_report.json"
            ).open("w", encoding="utf-8") as file:
                json.dump(
                    report,
                    file,
                    indent=2,
                )

            with (
                model_dir / "metrics.json"
            ).open("w", encoding="utf-8") as file:
                json.dump(
                    metrics,
                    file,
                    indent=2,
                )

            save_model_diagnostics(
                model_name=model_name,
                model=model,
                feature_columns=feature_columns,
                model_dir=model_dir,
            )

            print()
            print(f"Accuracy          : {metrics['accuracy']:.4f}")
            print(
                "Balanced accuracy : "
                f"{metrics['balanced_accuracy']:.4f}"
            )
            print(f"Macro F1          : {metrics['macro_f1']:.4f}")
            print(f"Log loss          : {metrics['log_loss']:.4f}")
            print(f"Signal rate       : {metrics['signal_rate']:.4f}")
            print(
                "Mean signal return: "
                f"{metrics['mean_signal_return_bps']:.4f} bps"
            )

        except Exception as exception:
            print(
                f"MODEL FAILED: {model_name}\n"
                f"{type(exception).__name__}: {exception}"
            )

            failure = {
                "model": model_name,
                "error_type": type(exception).__name__,
                "error": str(exception),
            }

            with (
                model_dir / "failure.json"
            ).open("w", encoding="utf-8") as file:
                json.dump(
                    failure,
                    file,
                    indent=2,
                )

        gc.collect()

    if not summary_rows:
        raise RuntimeError(
            "Every model failed. Review the model failure files."
        )

    summary = pd.DataFrame(
        summary_rows
    ).sort_values(
        [
            "balanced_accuracy",
            "macro_f1",
        ],
        ascending=False,
    )

    summary.to_csv(
        args.output_dir / "model_comparison.csv",
        index=False,
    )

    print()
    print("=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    display_columns = [
        "model",
        "balanced_accuracy",
        "macro_f1",
        "log_loss",
        "signal_rate",
        "mean_signal_return_bps",
        "signal_win_rate",
    ]

    print(
        summary[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.5f}",
        )
    )

    print()
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()