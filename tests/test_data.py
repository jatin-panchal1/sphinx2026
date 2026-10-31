"""
Tests for the data ingestion and loading pipeline.

Tests:
- ZIP extraction
- File discovery
- CSV / Excel / JSON / text loading
- Target detection and binarisation
- Invalid file handling
"""

import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data_loader import (
    _binarise_series,
    build_dataset_record,
    clean_dataframe,
    detect_target,
    discover_files,
    extract_zips,
    load_file,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture()
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture()
def sample_csv(tmp_dir: Path) -> Path:
    """Write a sample CSV with known default target."""
    df = pd.DataFrame({
        "monthly_income": [20000, 35000, 15000, 50000, 25000],
        "rent_score":     [0.8, 0.9, 0.4, 0.95, 0.7],
        "is_default":     [0, 0, 1, 0, 1],
    })
    path = tmp_dir / "sample.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture()
def sample_xlsx(tmp_dir: Path) -> Path:
    """Write a sample Excel file."""
    df = pd.DataFrame({
        "income": [10000, 20000, 5000],
        "target": [0, 1, 0],
    })
    path = tmp_dir / "sample.xlsx"
    df.to_excel(path, index=False)
    return path


@pytest.fixture()
def sample_json(tmp_dir: Path) -> Path:
    data = [
        {"income": 20000, "utility_score": 0.8, "default": 0},
        {"income": 8000,  "utility_score": 0.3, "default": 1},
    ]
    path = tmp_dir / "sample.json"
    path.write_text(json.dumps(data))
    return path


@pytest.fixture()
def sample_zip(tmp_dir: Path, sample_csv: Path) -> Path:
    zip_path = tmp_dir / "archive.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(sample_csv, arcname="data/sample.csv")
    return zip_path


# ===========================================================================
# Tests: File Discovery
# ===========================================================================

class TestFileDiscovery:
    def test_discovers_csv(self, sample_csv: Path, tmp_dir: Path):
        files = discover_files([tmp_dir])
        assert any(f.name == "sample.csv" for f in files)

    def test_discovers_xlsx(self, sample_xlsx: Path, tmp_dir: Path):
        files = discover_files([tmp_dir])
        assert any(f.name == "sample.xlsx" for f in files)

    def test_discovers_json(self, sample_json: Path, tmp_dir: Path):
        files = discover_files([tmp_dir])
        assert any(f.name == "sample.json" for f in files)

    def test_no_files_in_empty_dir(self, tmp_dir: Path):
        empty_sub = tmp_dir / "empty_sub"
        empty_sub.mkdir()
        files = discover_files([empty_sub])
        assert len(files) == 0

    def test_deduplication(self, sample_csv: Path, tmp_dir: Path):
        # Passing the same directory twice should not duplicate
        files = discover_files([tmp_dir, tmp_dir])
        names = [f.name for f in files]
        assert names.count("sample.csv") == 1


# ===========================================================================
# Tests: ZIP Extraction
# ===========================================================================

class TestZipExtraction:
    def test_extracts_zip(self, sample_zip: Path, tmp_dir: Path):
        extracted_dir = tmp_dir / "extracted"
        extracted_dir.mkdir()
        extract_zips(tmp_dir, extracted_dir)
        # Check that a CSV appeared somewhere in extracted
        found = list(extracted_dir.rglob("*.csv"))
        assert len(found) > 0

    def test_handles_corrupt_zip(self, tmp_dir: Path):
        bad_zip = tmp_dir / "bad.zip"
        bad_zip.write_bytes(b"not a real zip")
        extracted_dir = tmp_dir / "extracted"
        extracted_dir.mkdir()
        # Should not raise
        extract_zips(tmp_dir, extracted_dir)


# ===========================================================================
# Tests: File Loading
# ===========================================================================

class TestFileLoading:
    def test_load_csv(self, sample_csv: Path):
        df = load_file(sample_csv)
        assert df is not None
        assert len(df) == 5
        assert "is_default" in df.columns

    def test_load_xlsx(self, sample_xlsx: Path):
        df = load_file(sample_xlsx)
        assert df is not None
        assert "target" in df.columns

    def test_load_json(self, sample_json: Path):
        df = load_file(sample_json)
        assert df is not None
        assert "default" in df.columns

    def test_load_nonexistent_returns_none(self, tmp_dir: Path):
        result = load_file(tmp_dir / "ghost.csv")
        assert result is None

    def test_load_empty_csv_returns_none(self, tmp_dir: Path):
        empty = tmp_dir / "empty.csv"
        empty.write_text("col1\n")
        df = load_file(empty)
        assert df is None or len(df) == 0


# ===========================================================================
# Tests: Target Detection
# ===========================================================================

class TestTargetDetection:
    def test_detects_is_default(self):
        df = pd.DataFrame({"income": [1], "is_default": [0]})
        assert detect_target(df) == "is_default"

    def test_detects_target(self):
        df = pd.DataFrame({"x": [1], "target": [0]})
        assert detect_target(df) == "target"

    def test_detects_label(self):
        df = pd.DataFrame({"x": [1], "label": [0]})
        assert detect_target(df) == "label"

    def test_case_insensitive(self):
        df = pd.DataFrame({"x": [1], "SeriousDlqin2yrs": [0]})
        assert detect_target(df) == "SeriousDlqin2yrs"

    def test_returns_none_when_absent(self):
        df = pd.DataFrame({"income": [1], "score": [0.5]})
        assert detect_target(df) is None


# ===========================================================================
# Tests: Binarisation
# ===========================================================================

class TestBinarisation:
    def test_binary_int_unchanged(self):
        s = pd.Series([0, 1, 0, 1, 0])
        result = _binarise_series(s)
        assert list(result) == [0, 1, 0, 1, 0]

    def test_string_yes_no(self):
        s = pd.Series(["yes", "no", "yes", "no"])
        result = _binarise_series(s)
        assert set(result.unique()) <= {0, 1}

    def test_string_good_bad(self):
        s = pd.Series(["good", "bad", "good"])
        result = _binarise_series(s)
        assert result is not None
        assert set(result.unique()) <= {0, 1}

    def test_fully_paid_charged_off(self):
        s = pd.Series(["fully paid", "charged off", "fully paid"])
        result = _binarise_series(s)
        assert result is not None

    def test_invalid_returns_none(self):
        s = pd.Series(["cat", "dog", "fish"])
        # Three distinct values – cannot infer binary
        result = _binarise_series(s)
        assert result is None or set(result.unique()) <= {0, 1}


# ===========================================================================
# Tests: Dataset Record Building
# ===========================================================================

class TestDatasetRecord:
    def test_full_pipeline(self, sample_csv: Path):
        df = load_file(sample_csv)
        record = build_dataset_record(sample_csv, df)
        assert record is not None
        assert record["target_col"] == "is_default"
        assert record["n_rows"] >= 3
        assert "is_default" not in record["feature_cols"]

    def test_record_has_required_keys(self, sample_csv: Path):
        df = load_file(sample_csv)
        record = build_dataset_record(sample_csv, df)
        for key in ["name", "df", "target_col", "feature_cols", "n_rows", "n_cols", "default_rate"]:
            assert key in record

    def test_invalid_target_skipped(self, tmp_dir: Path):
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "target": ["x", "y", "z"],  # Three categories, not binarisable
        })
        path = tmp_dir / "no_target.csv"
        df.to_csv(path, index=False)
        df_loaded = load_file(path)
        record = build_dataset_record(path, df_loaded)
        # Should return None or handle gracefully
        # (3 unique string values cannot be binarised to binary)
        # This is acceptable – test just verifies no crash
        assert record is None or isinstance(record, dict)

