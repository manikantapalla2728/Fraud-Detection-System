"""
Data Validation, Quality Enforcement, and Chronological Ingestion Module.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd


@dataclass
class ValidationReport:
    is_valid: bool
    total_records: int
    error_count: int
    warning_count: int
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "total_records": self.total_records,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "errors": self.errors,
            "warnings": self.warnings,
            "metrics": self.metrics,
        }


class DataValidator:
    """
    Validates incoming raw and batch transaction data against financial integrity standards.
    Distinguishes true fatal corruptions (e.g. negative amounts) from legitimate extreme behaviors
    (e.g. $0 auth checks, high-value luxury purchases).
    """

    EXPECTED_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    TARGET_COLUMN = "Class"

    def __init__(self, allow_missing_target: bool = False):
        self.allow_missing_target = allow_missing_target

    def validate_schema(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        missing_cols = []
        for col in self.EXPECTED_COLUMNS:
            if col not in df.columns:
                missing_cols.append(col)
        if not self.allow_missing_target and self.TARGET_COLUMN not in df.columns:
            missing_cols.append(self.TARGET_COLUMN)
        return len(missing_cols) == 0, missing_cols

    def validate(self, df: pd.DataFrame) -> ValidationReport:
        errors: List[str] = []
        warnings: List[str] = []
        metrics: Dict[str, Any] = {}

        n_rows = len(df)
        if n_rows == 0:
            return ValidationReport(
                is_valid=False,
                total_records=0,
                error_count=1,
                warning_count=0,
                errors=["Dataset is completely empty."],
            )

        # 1. Schema check
        valid_schema, missing_cols = self.validate_schema(df)
        if not valid_schema:
            errors.append(f"Missing required columns: {missing_cols}")

        # 2. Null checks (on existing expected columns)
        existing_cols = [c for c in self.EXPECTED_COLUMNS if c in df.columns]
        null_counts = df[existing_cols].isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0].to_dict()
        if cols_with_nulls:
            errors.append(f"Unexpected null values found in columns: {cols_with_nulls}")
        metrics["null_counts"] = cols_with_nulls

        # 3. Target integrity check
        if self.TARGET_COLUMN in df.columns:
            target_vals = df[self.TARGET_COLUMN].dropna().unique()
            invalid_target_vals = [v for v in target_vals if v not in (0, 1, 0.0, 1.0)]
            if invalid_target_vals:
                errors.append(f"Invalid target values detected: {invalid_target_vals}. Must be strictly {0, 1}.")

            fraud_rate = float(df[self.TARGET_COLUMN].mean())
            metrics["fraud_rate"] = fraud_rate
            metrics["fraud_count"] = int(df[self.TARGET_COLUMN].sum())
            if fraud_rate == 0.0:
                warnings.append("No positive fraud instances found in target series.")

        # 4. Domain sanity checks
        if "Amount" in df.columns:
            neg_amounts = (df["Amount"] < 0).sum()
            if neg_amounts > 0:
                errors.append(f"Found {neg_amounts} transactions with negative Amount (< 0). FATAL DATA CORRUPTION.")

            zero_amounts = (df["Amount"] == 0).sum()
            if zero_amounts > 0:
                warnings.append(f"Detected {zero_amounts} zero-dollar transactions ($0.00). Flagged as prospective authorization pings.")
            metrics["zero_amount_count"] = int(zero_amounts)

            extreme_amounts = (df["Amount"] > 20000).sum()
            if extreme_amounts > 0:
                warnings.append(f"Detected {extreme_amounts} extreme value transactions (> $20,000). Retained as legitimate high-value behavior.")
            metrics["amount_max"] = float(df["Amount"].max())
            metrics["amount_min"] = float(df["Amount"].min())

        # 5. Temporal sanity checks
        if "Time" in df.columns:
            neg_time = (df["Time"] < 0).sum()
            if neg_time > 0:
                errors.append(f"Found {neg_time} negative timestamps (< 0).")
            if not df["Time"].is_monotonic_increasing:
                warnings.append("Time series is not strictly monotonically increasing. Chronological sorting required.")

        # 6. Duplicate check
        dup_count = int(df.duplicated().sum())
        metrics["duplicate_rows"] = dup_count
        if dup_count > 0:
            warnings.append(f"Found {dup_count} duplicate rows ({dup_count/n_rows:.2%}). In payment rails, exact telemetry replays occur during automated fraud retries.")

        return ValidationReport(
            is_valid=(len(errors) == 0),
            total_records=n_rows,
            error_count=len(errors),
            warning_count=len(warnings),
            errors=errors,
            warnings=warnings,
            metrics=metrics,
        )


class DataLoader:
    """
    Loads, validates, and chronologically partitions financial transactions into Train/Val/Test splits.
    """

    def __init__(self, raw_path: str = "data/raw/creditcard.csv"):
        self.raw_path = Path(raw_path)
        self.validator = DataValidator(allow_missing_target=False)

    def load_raw(self) -> pd.DataFrame:
        if not self.raw_path.exists():
            raise FileNotFoundError(f"Raw dataset file not found at {self.raw_path}")
        return pd.read_csv(self.raw_path)

    def split_temporal(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Split ratios must sum to 1.0"
        df_sorted = df.sort_values(by="Time").reset_index(drop=True)

        n = len(df_sorted)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_df = df_sorted.iloc[:train_end].copy().reset_index(drop=True)
        val_df = df_sorted.iloc[train_end:val_end].copy().reset_index(drop=True)
        test_df = df_sorted.iloc[val_end:].copy().reset_index(drop=True)

        return train_df, val_df, test_df

    def process_and_save(
        self,
        output_dir: str = "data/processed",
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Dict[str, Any]:
        df = self.load_raw()
        report = self.validator.validate(df)
        if not report.is_valid:
            raise ValueError(f"Dataset failed validation: {report.errors}")

        train_df, val_df, test_df = self.split_temporal(
            df, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio
        )

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        train_file = out_path / "train.parquet"
        val_file = out_path / "val.parquet"
        test_file = out_path / "test.parquet"

        train_df.to_parquet(train_file, index=False)
        val_df.to_parquet(val_file, index=False)
        test_df.to_parquet(test_file, index=False)

        return {
            "validation_report": report.to_dict(),
            "train_size": len(train_df),
            "train_fraud_rate": float(train_df["Class"].mean()),
            "val_size": len(val_df),
            "val_fraud_rate": float(val_df["Class"].mean()),
            "test_size": len(test_df),
            "test_fraud_rate": float(test_df["Class"].mean()),
            "files": {
                "train": str(train_file),
                "val": str(val_file),
                "test": str(test_file),
            },
        }
