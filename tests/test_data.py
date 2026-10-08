"""
Unit tests for data validation, schema enforcement, and chronological partitioning.
"""

import numpy as np
import pandas as pd
import pytest

from src.data import DataValidator, DataLoader


def test_validator_valid_dataset():
    validator = DataValidator()
    df = pd.DataFrame({
        "Time": [0.0, 10.0, 20.0],
        **{f"V{i}": [0.1, -0.2, 0.3] for i in range(1, 29)},
        "Amount": [10.0, 50.0, 100.0],
        "Class": [0, 0, 1],
    })
    report = validator.validate(df)
    assert report.is_valid
    assert len(report.errors) == 0


def test_validator_negative_amount_fatal_error():
    validator = DataValidator()
    df = pd.DataFrame({
        "Time": [0.0, 10.0],
        **{f"V{i}": [0.1, -0.2] for i in range(1, 29)},
        "Amount": [10.0, -5.0],  # Negative amount
        "Class": [0, 1],
    })
    report = validator.validate(df)
    assert not report.is_valid
    assert any("negative Amount" in e for e in report.errors)


def test_validator_missing_schema_column():
    validator = DataValidator()
    df = pd.DataFrame({
        "Time": [0.0, 10.0],
        "Amount": [10.0, 20.0],
        "Class": [0, 1],
    })
    report = validator.validate(df)
    assert not report.is_valid
    assert any("Missing required columns" in e for e in report.errors)


def test_validator_zero_amount_is_warning_not_error():
    validator = DataValidator()
    df = pd.DataFrame({
        "Time": [0.0, 10.0],
        **{f"V{i}": [0.1, -0.2] for i in range(1, 29)},
        "Amount": [0.0, 25.0],
        "Class": [0, 0],
    })
    report = validator.validate(df)
    assert report.is_valid
    assert any("zero-dollar" in w for w in report.warnings)


def test_loader_temporal_split():
    df = pd.DataFrame({
        "Time": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
        **{f"V{i}": [0.0] * 10 for i in range(1, 29)},
        "Amount": [10.0] * 10,
        "Class": [0] * 9 + [1],
    })
    loader = DataLoader()
    train_df, val_df, test_df = loader.split_temporal(df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    assert len(train_df) == 6
    assert len(val_df) == 2
    assert len(test_df) == 2
    assert train_df["Time"].max() <= val_df["Time"].min()
    assert val_df["Time"].max() <= test_df["Time"].min()
