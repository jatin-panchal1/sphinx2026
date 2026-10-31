"""
Data ingestion and loading module for FairScore Alternative Credit Scoring.

Scans the data directory recursively, extracts ZIPs, loads tabular files,
infers target columns, cleans data, and returns a list of ready-to-train datasets.
"""

import json
import logging
import os
import sys
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("fairscore.data_loader")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".txt", ".data", ".dat", ".asc", ".json", ".parquet"}
ZIP_EXTENSION = ".zip"

TARGET_NAMES = {
    "default", "default_90_days", "is_default", "target", "label", "y",
    "class", "loan_status", "credit_risk", "seriousdlqin2yrs",
    "default payment next month", "kredit",
}

GOOD_VALUES = {"repaid", "good", "fully paid", "current", "safe", "no", "0", 0, "+", "approved"}
BAD_VALUES  = {"default", "bad", "charged off", "late", "risk", "yes", "1", 1, "-", "denied", 2, "2"}

# Target binarisation maps (lower-cased)
BINARY_MAP_BAD = {
    "1": 1, "yes": 1, "default": 1, "bad": 1, "charged off": 1,
    "late (16-30 days)": 1, "late (31-120 days)": 1, "risk": 1,
    "-": 1, "denied": 1, "2": 1, "bad.": 1,
    "does not meet the credit policy. status:charged off": 1,
}
BINARY_MAP_GOOD = {
    "0": 0, "no": 0, "repaid": 0, "good": 0, "fully paid": 0,
    "current": 0, "safe": 0, "+": 0, "approved": 0, "good.": 0,
    "does not meet the credit policy. status:fully paid": 0,
}

MAX_SAMPLE_ROWS = 20_000
HIGH_CARDINALITY_THRESHOLD = 200
MISSING_VALUE_THRESHOLD = 0.90
ENCODINGS_TO_TRY = ["utf-8", "latin1", "cp1252"]


# ===========================================================================
# ZIP Extraction
# ===========================================================================

def extract_zips(data_dir: Path, extracted_dir: Path) -> set[Path]:
    """Recursively extract ZIP files found in *data_dir* into *extracted_dir*.

    Returns a set of already-processed ZIP paths to avoid duplicate extraction.
    """
    seen_zips: set[Path] = set()

    zip_files = list(data_dir.rglob("*.zip"))
    if not zip_files:
        log.info("No ZIP files found in %s", data_dir)
        return seen_zips

    log.info("Found %d ZIP file(s) to extract.", len(zip_files))
    for zf in tqdm(zip_files, desc="Extracting ZIPs", unit="zip"):
        zf = Path(zf)
        if zf in seen_zips:
            continue
        seen_zips.add(zf)
        dest = extracted_dir / zf.stem
        dest.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(zf, "r") as z:
                z.extractall(dest)
            log.info("Extracted %s → %s", zf.name, dest)
        except (zipfile.BadZipFile, OSError) as exc:
            log.warning("Skipping corrupt ZIP %s: %s", zf.name, exc)

    return seen_zips


# ===========================================================================
# File Discovery
# ===========================================================================

def discover_files(search_dirs: list[Path]) -> list[Path]:
    """Return a deduplicated list of supported data files found in *search_dirs*."""
    seen: set[Path] = set()
    files: list[Path] = []

    for d in search_dirs:
        if not d.exists():
            continue
        for ext in SUPPORTED_EXTENSIONS:
            for fp in d.rglob(f"*{ext}"):
                resolved = fp.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    files.append(fp)

    log.info("Discovered %d data file(s) across %d search path(s).", len(files), len(search_dirs))
    return files


# ===========================================================================
# File Loading
# ===========================================================================

def _load_csv_like(path: Path) -> Optional[pd.DataFrame]:
    ext = path.suffix.lower()
    # For whitespace-delimited files (.dat, .data, .asc, .txt), prioritize whitespace
    if ext in (".dat", ".data", ".asc", ".txt"):
        delims = [r"\s+", ",", "\t", ";", None]
    else:
        delims = [",", r"\s+", "\t", ";", None]

    for enc in ENCODINGS_TO_TRY:
        for delim in delims:
            try:
                engine = "python" if delim in (None, r"\s+") else "c"
                kwargs = {"engine": engine, "encoding": enc}
                if engine == "c":
                    kwargs["low_memory"] = False
                df = pd.read_csv(path, sep=delim, **kwargs)
                if df is not None and df.shape[1] >= 2:
                    # Check if first row was actually data that became headers (e.g. all numeric column names)
                    if all(str(c).replace(".", "", 1).isdigit() for c in df.columns):
                        df = pd.read_csv(path, sep=delim, header=None, **kwargs)
                    return df
            except Exception:
                continue
    return None


def _load_excel(path: Path) -> Optional[pd.DataFrame]:
    for engine in (["openpyxl", "xlrd"] if path.suffix in (".xlsx",) else ["xlrd", "openpyxl"]):
        try:
            df = pd.read_excel(path, engine=engine)
            if df is not None and not df.empty:
                # Check for Taiwan dataset style: row 0 contains actual column names
                if any("Unnamed" in str(c) for c in df.columns) and any("default" in str(v).lower() for v in df.iloc[0]):
                    df.columns = [str(v).strip() for v in df.iloc[0]]
                    df = df.iloc[1:].reset_index(drop=True)
                return df
        except Exception as exc:
            log.warning("Excel load attempt with %s failed for %s: %s", engine, path.name, exc)
            continue
    return None


def _load_json(path: Path) -> Optional[pd.DataFrame]:
    for enc in ENCODINGS_TO_TRY:
        try:
            with open(path, encoding=enc) as fh:
                raw = json.load(fh)
            if isinstance(raw, list):
                return pd.DataFrame(raw)
            if isinstance(raw, dict):
                return pd.DataFrame([raw])
        except Exception:
            continue
    return None


def _load_parquet(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        log.warning("Parquet load failed for %s: %s", path.name, exc)
        return None


def load_file(path: Path) -> Optional[pd.DataFrame]:
    """Load a single data file into a DataFrame. Returns None on failure."""
    ext = path.suffix.lower()
    try:
        if ext in (".csv",):
            df = _load_csv_like(path)
        elif ext in (".xlsx", ".xls"):
            df = _load_excel(path)
        elif ext in (".txt", ".data", ".dat", ".asc"):
            df = _load_csv_like(path)
        elif ext == ".json":
            df = _load_json(path)
        elif ext == ".parquet":
            df = _load_parquet(path)
        else:
            log.warning("Unsupported extension %s – skipping %s", ext, path.name)
            return None
    except Exception as exc:
        log.error("Unexpected error loading %s: %s", path.name, exc)
        return None

    if df is None:
        return None

    # Drop unnamed index columns
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed:")]
    if unnamed:
        df.drop(columns=unnamed, inplace=True)

    if df.empty or df.shape[1] < 2:
        log.warning("File %s has insufficient columns after load – skipping.", path.name)
        return None

    return df


# ===========================================================================
# Target Detection
# ===========================================================================

def _binarise_series(s: pd.Series, col_name: str = "") -> Optional[pd.Series]:
    """Convert a series to binary 0/1. Returns None if conversion fails.

    1 = default / bad / charged off / late / risk
    0 = repaid / good / fully paid / current / safe
    """
    col_lower = str(col_name).lower().strip()

    # Specific known dataset rules: SouthGermanCredit 'kredit' (0 = bad, 1 = good)
    if col_lower == "kredit":
        return s.map({0: 1, 1: 0, "0": 1, "1": 0}).astype(int)

    s_str = s.astype(str).str.strip().str.lower()
    unique_vals = set(s_str.unique())

    # CRX dataset: '+' = approved (0), '-' = denied / bad risk (1)
    if unique_vals <= {"+", "-", "+.0", "-.0"}:
        return s_str.map({"+": 0, "-": 1, "+.0": 0, "-.0": 1}).astype(int)

    # German credit numeric: 1 = good (0), 2 = bad (1)
    try:
        numeric = pd.to_numeric(s, errors="coerce")
        uvals = set(numeric.dropna().unique())
        if uvals in ({1.0, 2.0}, {1, 2}):
            return (numeric == 2).astype(int)
        if uvals <= {0, 1, 0.0, 1.0}:
            return numeric.astype(int)
        # Broader numeric: binarise > 0 as 1
        if numeric.between(0, 1).all():
            return (numeric > 0.5).astype(int)
        # May be ordinal classes – only binarise if exactly 2
        if len(uvals) == 2:
            min_val = min(uvals)
            return (numeric != min_val).astype(int)
    except (ValueError, TypeError):
        pass

    # String mapping
    result = pd.Series(index=s.index, dtype=float)
    mapped_any = False
    for i, v in enumerate(s_str):
        if v in BINARY_MAP_BAD:
            result.iloc[i] = BINARY_MAP_BAD[v]
            mapped_any = True
        elif v in BINARY_MAP_GOOD:
            result.iloc[i] = BINARY_MAP_GOOD[v]
            mapped_any = True

    if not mapped_any:
        # Last resort: exactly 2 unique string values
        clean_vals = [str(u).strip() for u in unique_vals if pd.notna(u) and str(u).lower() not in ("nan", "none", "")]
        uniq = sorted(set(clean_vals))
        if len(uniq) == 2:
            mapping = {uniq[0]: 0, uniq[1]: 1}
            result = s_str.map(mapping)
            if result.isna().sum() == 0:
                return result.astype(int)
        return None

    if result.isna().sum() / max(len(result), 1) > 0.3:
        return None  # Too many unmapped values

    result.fillna(0, inplace=True)
    return result.astype(int)


def detect_target(df: pd.DataFrame) -> Optional[str]:
    """Return the name of the target column, or None if not found."""
    cols_lower = {c.lower().strip(): c for c in df.columns}

    for target_name in TARGET_NAMES:
        if target_name in cols_lower:
            return cols_lower[target_name]

    # Fuzzy: columns containing known target substrings
    for lower_col, original_col in cols_lower.items():
        for keyword in ("default", "target", "label", "risk", "dlq"):
            if keyword in lower_col:
                return original_col

    return None


# ===========================================================================
# Data Cleaning
# ===========================================================================

def _is_id_like(series: pd.Series, col_name: str) -> bool:
    """Return True if the column looks like an ID (high cardinality, monotonic)."""
    name_lower = col_name.lower()
    if any(kw in name_lower for kw in ("_id", "id_", "applicant", "customer", "index")):
        return True
    if series.dtype == object and series.nunique() / max(len(series), 1) > 0.95:
        return True
    return False


def clean_dataframe(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Drop ID columns, constant columns, high-missing columns, high-cardinality columns."""
    df = df.copy()

    # Convert numeric-like string columns into numeric types before cardinality checks
    for col in df.columns:
        if col != target_col and df[col].dtype == object:
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                pass

    feature_cols = [c for c in df.columns if c != target_col]
    drop_cols: list[str] = []

    for col in feature_cols:
        s = df[col]
        # ID-like
        if _is_id_like(s, col):
            log.debug("Dropping ID-like column: %s", col)
            drop_cols.append(col)
            continue
        # Constant
        if s.nunique(dropna=False) <= 1:
            log.debug("Dropping constant column: %s", col)
            drop_cols.append(col)
            continue
        # High missing
        missing_rate = s.isna().mean()
        if missing_rate > MISSING_VALUE_THRESHOLD:
            log.debug("Dropping high-missing column %s (%.1f%% missing)", col, 100 * missing_rate)
            drop_cols.append(col)
            continue
        # High cardinality categorical
        if s.dtype == object and s.nunique() > HIGH_CARDINALITY_THRESHOLD:
            log.debug("Dropping high-cardinality column: %s (%d unique)", col, s.nunique())
            drop_cols.append(col)

    df = df.drop(columns=drop_cols)
    return df


# ===========================================================================
# Dataset Pipeline
# ===========================================================================

def build_dataset_record(path: Path, df: pd.DataFrame) -> Optional[dict]:
    """Detect target, binarise it, clean features, and return a dataset dict."""
    # Ensure all column names are clean strings
    df = df.copy()
    df.columns = [
        f"feat_{c}" if (isinstance(c, (int, float)) or str(c).isdigit()) else str(c).strip()
        for c in df.columns
    ]

    target_col = detect_target(df)

    if target_col is None:
        # Try last column
        last_col = df.columns[-1]
        binary = _binarise_series(df[last_col], col_name=str(last_col))
        if binary is not None:
            target_col = last_col
            log.info("Used last column '%s' as inferred target for %s", last_col, path.name)
        else:
            log.warning("No target found in %s – skipping.", path.name)
            return None
    else:
        binary = _binarise_series(df[target_col], col_name=str(target_col))
        if binary is None:
            log.warning("Could not binarise target '%s' in %s – skipping.", target_col, path.name)
            return None

    df[target_col] = binary

    # Remove rows where target is NaN after binarisation
    df = df.dropna(subset=[target_col])
    if len(df) < 4 or df[target_col].nunique() < 2:
        log.warning("Dataset %s has < 4 valid rows or single-class target – skipping.", path.name)
        return None

    # Clean features
    df = clean_dataframe(df, target_col)

    feature_cols = [c for c in df.columns if c != target_col]
    if len(feature_cols) < 2:
        log.warning("Dataset %s has < 2 features after cleaning – skipping.", path.name)
        return None

    # Sample large datasets
    if len(df) > MAX_SAMPLE_ROWS:
        df = df.sample(MAX_SAMPLE_ROWS, random_state=42)
        log.info("Sampled %s to %d rows.", path.name, MAX_SAMPLE_ROWS)

    default_rate = df[target_col].mean()
    log.info(
        "Dataset ready: %s | rows=%d cols=%d target='%s' default_rate=%.2f%%",
        path.name, len(df), len(feature_cols), target_col, 100 * default_rate,
    )

    return {
        "path": str(path),
        "name": path.stem,
        "df": df,
        "target_col": target_col,
        "feature_cols": feature_cols,
        "n_rows": len(df),
        "n_cols": len(feature_cols),
        "default_rate": float(default_rate),
    }


# ===========================================================================
# Main Entry Point
# ===========================================================================

def load_all_datasets(
    data_dir: str = "data",
    extracted_dir: str = "data/extracted",
) -> list[dict]:
    """Scan, extract ZIPs, load files, detect targets, clean, and return datasets."""
    data_path = Path(data_dir)
    extracted_path = Path(extracted_dir)

    # 1. Extract ZIPs
    extract_zips(data_path, extracted_path)

    # 2. Discover files
    search_dirs = [data_path / "raw", extracted_path, data_path]
    files = discover_files(search_dirs)

    if not files:
        log.info("No data files found – will use synthetic data.")
        return []

    # 3. Load files
    datasets: list[dict] = []
    skipped: list[str] = []

    for fp in tqdm(files, desc="Loading files", unit="file"):
        df = load_file(fp)
        if df is None:
            skipped.append(f"{fp.name}: load failed")
            continue
        record = build_dataset_record(fp, df)
        if record is None:
            skipped.append(f"{fp.name}: no valid target")
            continue
        datasets.append(record)

    log.info(
        "Loaded %d usable dataset(s); %d file(s) skipped.",
        len(datasets), len(skipped),
    )
    for reason in skipped:
        log.warning("  Skipped: %s", reason)

    return datasets


if __name__ == "__main__":
    results = load_all_datasets()
    print(f"\nDatasets ready for training: {len(results)}")
    for r in results:
        print(f"  {r['name']} | rows={r['n_rows']} | cols={r['n_cols']} | default_rate={r['default_rate']:.2%}")

