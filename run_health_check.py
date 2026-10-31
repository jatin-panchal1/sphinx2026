import os
import sys
import re
import importlib
import subprocess
import pathlib
import py_compile
from datetime import datetime

ROOT = pathlib.Path.cwd()

REPORT = []
PASS = []
WARN = []
FAIL = []

missing_dirs = []
missing_files = []
missing_required = []

def section(title):
    line = f"\n{'=' * 70}\n{title}\n{'=' * 70}"
    print(line)
    REPORT.append(line)

def ok(msg):
    PASS.append(msg)
    print(f"✅ {msg}")
    REPORT.append(f"✅ {msg}")

def warn(msg):
    WARN.append(msg)
    print(f"⚠️ {msg}")
    REPORT.append(f"⚠️ {msg}")

def fail(msg):
    FAIL.append(msg)
    print(f"❌ {msg}")
    REPORT.append(f"❌ {msg}")

section("PROJECT HEALTH CHECK")
print(f"Current directory: {ROOT}")
REPORT.append(f"Current directory: {ROOT}")

# ------------------------------------------------------------
# 1. PYTHON VERSION
# ------------------------------------------------------------

section("1. Python Version")

if sys.version_info >= (3, 9):
    ok(f"Python version is compatible: {sys.version.split()[0]}")
else:
    warn(f"Python version may be too old: {sys.version.split()[0]}. Recommended: 3.9+")

# ------------------------------------------------------------
# 2. PROJECT STRUCTURE
# ------------------------------------------------------------

section("2. Project Structure")

required_dirs = [
    "data",
    "data/raw",
    "data/extracted",
    "models",
    "backend",
    "frontend",
    "scripts",
    "tests",
    "outputs",
    "outputs/metrics",
    "outputs/reports",
    "outputs/logs",
]

for d in required_dirs:
    path = ROOT / d
    if path.is_dir():
        ok(f"Directory exists: {d}")
    else:
        missing_dirs.append(d)
        fail(f"Missing directory: {d}")

required_files = [
    "requirements.txt",
    "README.md",
    "backend/main.py",
    "frontend/app.py",
    "tests/test_backend.py",
]

for f in required_files:
    path = ROOT / f
    if path.is_file():
        ok(f"File exists: {f}")
    else:
        missing_files.append(f)
        fail(f"Missing file: {f}")

# ------------------------------------------------------------
# 3. DATA FILES
# ------------------------------------------------------------

section("3. Data Files")

data_extensions = {
    "csv",
    "txt",
    "xlsx",
    "xls",
    "data",
    "dat",
    "asc",
    "json",
    "parquet",
    "zip",
}

counts = {ext: 0 for ext in data_extensions}
data_files = []

exclude_parts = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    ".ipynb_checkpoints",
}

for p in ROOT.rglob("*"):
    if not p.is_file():
        continue

    if any(part in exclude_parts for part in p.parts):
        continue

    ext = p.suffix.lower().lstrip(".")
    if ext in data_extensions:
        counts[ext] += 1
        data_files.append(str(p.relative_to(ROOT)))

total_data_files = sum(counts.values())

if total_data_files == 0:
    fail("No data files found. Add CSV/Excel/TXT/JSON/Parquet/ZIP files or generate synthetic data.")
else:
    ok(f"Found {total_data_files} data file(s).")

for ext, count in counts.items():
    if count > 0:
        ok(f"Found {count} .{ext} file(s)")

zip_count = counts.get("zip", 0)
if zip_count == 0:
    warn("No ZIP files found. If your datasets are zipped, upload them.")
else:
    ok(f"Found {zip_count} ZIP file(s).")

# ------------------------------------------------------------
# 4. DEPENDENCIES
# ------------------------------------------------------------

section("4. Dependencies")

required_imports = {
    "sklearn": "scikit-learn",
    "pandas": "pandas",
    "numpy": "numpy",
    "joblib": "joblib",
    "tqdm": "tqdm",
    "fastapi": "fastapi",
    "pydantic": "pydantic",
    "streamlit": "streamlit",
    "uvicorn": "uvicorn",
    "pytest": "pytest",
    "httpx": "httpx",
}

optional_imports = {
    "openpyxl": "openpyxl",
    "xlrd": "xlrd",
    "xgboost": "xgboost",
    "shap": "shap",
    "pyarrow": "pyarrow",
}

for module_name, pip_name in required_imports.items():
    try:
        importlib.import_module(module_name)
        ok(f"Required package available: {module_name}")
    except Exception:
        missing_required.append(pip_name)
        fail(f"Missing required package: {pip_name}")

missing_optional = []

for module_name, pip_name in optional_imports.items():
    try:
        importlib.import_module(module_name)
        ok(f"Optional package available: {module_name}")
    except Exception:
        missing_optional.append(pip_name)
        warn(f"Missing optional package: {pip_name}")

if missing_required:
    print("\nTo install missing required packages, run:")
    print(f"pip install {' '.join(sorted(set(missing_required)))}")

if missing_optional:
    print("\nTo install optional packages, run:")
    print(f"pip install {' '.join(sorted(set(missing_optional)))}")

# ------------------------------------------------------------
# 5. SYNTAX CHECK
# ------------------------------------------------------------

section("5. Python Syntax Check")

py_files = []

for p in ROOT.rglob("*.py"):
    if any(part in exclude_parts for part in p.parts):
        continue
    py_files.append(p)

syntax_errors = []

for f in py_files:
    try:
        py_compile.compile(str(f), doraise=True)
    except Exception as e:
        syntax_errors.append((str(f), str(e)))

if not py_files:
    fail("No Python files found. This may not be a Python project yet.")
elif syntax_errors:
    for file_path, error in syntax_errors:
        fail(f"Syntax error in {file_path}: {error}")
else:
    ok(f"All {len(py_files)} Python file(s) passed syntax check.")

# ------------------------------------------------------------
# 6. FASTAPI STATIC CHECK
# ------------------------------------------------------------

section("6. FastAPI Static Check")

backend_candidates = [
    ROOT / "backend" / "main.py",
    ROOT / "backend" / "app.py",
    ROOT / "main.py",
    ROOT / "app.py",
]

backend_file = None

for candidate in backend_candidates:
    if candidate.exists():
        backend_file = candidate
        break

if backend_file is None:
    fail("No FastAPI backend file found. Expected backend/main.py, backend/app.py, main.py, or app.py.")
else:
    ok(f"Backend file found: {backend_file.relative_to(ROOT)}")

    text = backend_file.read_text(errors="ignore")

    if re.search(r"FastAPI\(", text):
        ok("FastAPI instance detected in backend file.")
    else:
        fail("FastAPI instance not detected. Add: app = FastAPI()")

    expected_routes = [
        ("get", "/health"),
        ("get", "/models"),
        ("post", "/predict"),
        ("post", "/explain"),
        ("get", "/metrics"),
        ("get", "/fairness"),
    ]

    for method, path in expected_routes:
        pattern = rf"@\w+\.{method}\(\s*['\"]{re.escape(path)}['\"]"
        if re.search(pattern, text):
            ok(f"Backend route found: {method.upper()} {path}")
        else:
            fail(f"Missing backend route: {method.upper()} {path}")

# ------------------------------------------------------------
# 7. FASTAPI DYNAMIC CHECK
# ------------------------------------------------------------

section("7. FastAPI Dynamic Check")

app_obj = None
app_module_name = None
last_import_error = None

sys.path.insert(0, str(ROOT))

modules_to_try = [
    "backend.main",
    "backend.app",
    "main",
    "app",
]

for mod_name in modules_to_try:
    try:
        mod = importlib.import_module(mod_name)
        if hasattr(mod, "app"):
            app_obj = getattr(mod, "app")
            app_module_name = mod_name
            break
    except Exception as e:
        last_import_error = e

if app_obj is None:
    warn(f"Could not import FastAPI app dynamically. Error: {last_import_error}")
else:
    ok(f"FastAPI app imported from module: {app_module_name}")

    try:
        from fastapi.testclient import TestClient

        client = TestClient(app_obj)

        get_endpoints = [
            "/health",
            "/models",
            "/metrics",
            "/fairness",
        ]

        for endpoint in get_endpoints:
            try:
                response = client.get(endpoint)
                if response.status_code == 404:
                    fail(f"GET {endpoint} returned 404. Endpoint missing.")
                else:
                    ok(f"GET {endpoint} responded with status {response.status_code}")
            except Exception as e:
                fail(f"GET {endpoint} failed: {e}")

        sample_payload = {
            "age": 30,
            "occupation": "gig_worker",
            "income_monthly_avg": 18000,
            "income_volatility_score": 55,
            "rent_on_time_pct": 90,
            "utility_on_time_pct": 90,
            "mobile_recharge_regularity_pct": 85,
            "cashflow_zero_weeks_6m": 1,
            "verified_cash_payments_6m": 3,
            "savings_deposits_6m": 4,
            "prior_missed_payments": 0,
            "business_active_days_per_week": 5,
            "loan_amount": 20000,
            "loan_tenure_months": 6,
            "expense_to_income_ratio": 0.72,
            "emi_to_income_ratio": 0.18,
        }

        post_endpoints = [
            "/predict",
            "/explain",
        ]

        for endpoint in post_endpoints:
            try:
                response = client.post(endpoint, json=sample_payload)

                if response.status_code == 404:
                    fail(f"POST {endpoint} returned 404. Endpoint missing.")
                elif response.status_code == 422:
                    warn(f"POST {endpoint} returned 422. Endpoint exists, but request schema may not match.")
                else:
                    ok(f"POST {endpoint} responded with status {response.status_code}")

            except Exception as e:
                fail(f"POST {endpoint} failed: {e}")

    except Exception as e:
        warn(f"Could not run FastAPI TestClient checks: {e}")

# ------------------------------------------------------------
# 8. STREAMLIT FRONTEND CHECK
# ------------------------------------------------------------

section("8. Streamlit Frontend Check")

frontend_file = ROOT / "frontend" / "app.py"

if not frontend_file.exists():
    fail("Missing frontend/app.py. Create a Streamlit app file.")
else:
    ok("frontend/app.py exists.")

    frontend_text = frontend_file.read_text(errors="ignore").lower()

    required_ui_keywords = {
        "consent": "consent",
        "score": "score",
        "explanation": "explanation",
        "dashboard": "dashboard",
        "bias audit": "bias",
    }

    for label, keyword in required_ui_keywords.items():
        if keyword in frontend_text:
            ok(f"Frontend contains {label} content.")
        else:
            warn(f"Frontend may be missing {label} content.")

# ------------------------------------------------------------
# 9. MODEL FILES CHECK
# ------------------------------------------------------------

section("9. Model Files Check")

models_dir = ROOT / "models"

model_files = []

if models_dir.exists():
    model_files.extend(models_dir.glob("*.joblib"))
    model_files.extend(models_dir.glob("*.pkl"))

if not model_files:
    fail("No trained model found in /models. Train and save a model first.")
else:
    ok(f"Found {len(model_files)} model file(s).")

    try:
        import joblib

        model_path = model_files[0]
        model = joblib.load(model_path)

        if hasattr(model, "predict_proba"):
            ok(f"Model loaded successfully and supports predict_proba: {model_path.name}")
        elif hasattr(model, "named_steps") and hasattr(model.named_steps.get("classifier", None), "predict_proba"):
            ok(f"Pipeline model loaded successfully and classifier supports predict_proba: {model_path.name}")
        else:
            warn(f"Model loaded but does not appear to support predict_proba: {model_path.name}")

    except Exception as e:
        fail(f"Could not load model: {e}")

# ------------------------------------------------------------
# 10. TESTS CHECK
# ------------------------------------------------------------

section("10. Pytest Check")

tests_dir = ROOT / "tests"

if not tests_dir.exists():
    fail("Missing tests directory. Add tests/test_backend.py and other tests.")
else:
    ok("tests directory exists.")

    try:
        import pytest

        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q", "tests"],
            capture_output=True,
            text=True,
            cwd=ROOT,
            timeout=120
        )

        if result.returncode == 0:
            ok("Pytest can collect tests successfully.")
        else:
            output = result.stdout + result.stderr
            fail(f"Pytest collection failed. Output snippet: {output[:1000]}")

    except Exception as e:
        warn(f"Could not run pytest collection: {e}")

# ------------------------------------------------------------
# 11. REQUIREMENTS.TXT CHECK
# ------------------------------------------------------------

section("11. requirements.txt Check")

requirements_file = ROOT / "requirements.txt"

if not requirements_file.exists():
    warn("requirements.txt not found.")
else:
    req_text = requirements_file.read_text(errors="ignore").lower()

    required_pip_names = sorted(set(required_imports.values()))
    missing_from_requirements = []

    for pip_name in required_pip_names:
        if pip_name.lower() not in req_text:
            missing_from_requirements.append(pip_name)

    if missing_from_requirements:
        warn("requirements.txt is missing these packages: " + ", ".join(missing_from_requirements))
    else:
        ok("requirements.txt contains all required core packages.")

# ------------------------------------------------------------
# 12. GENERATE REPORT
# ------------------------------------------------------------

section("12. Final Report")

report_path = ROOT / "project_check_report.md"

with open(report_path, "w", errors="ignore") as f:
    f.write("# Project Check Report\n\n")
    f.write(f"Generated at: {datetime.now().isoformat()}\n\n")
    f.write("\n".join(REPORT))

    f.write("\n\n## Summary\n\n")
    f.write(f"- Passed checks: {len(PASS)}\n")
    f.write(f"- Warnings: {len(WARN)}\n")
    f.write(f"- Missing/Failed checks: {len(FAIL)}\n")

    if FAIL:
        f.write("\n## Missing / Failed Items\n\n")
        for item in FAIL:
            f.write(f"- {item}\n")

    if WARN:
        f.write("\n## Warnings\n\n")
        for item in WARN:
            f.write(f"- {item}\n")

    f.write("\n## Suggested Fix Commands\n\n")

    if missing_dirs:
        f.write("Create missing directories:\n\n")
        f.write("```bash\n")
        f.write(f"mkdir -p {' '.join(missing_dirs)}\n")
        f.write("```\n\n")

    if missing_required:
        f.write("Install missing required packages:\n\n")
        f.write("```bash\n")
        f.write(f"pip install {' '.join(sorted(set(missing_required)))}\n")
        f.write("```\n\n")

    if missing_optional:
        f.write("Install optional packages:\n\n")
        f.write("```bash\n")
        f.write(f"pip install {' '.join(sorted(set(missing_optional)))}\n")
        f.write("```\n\n")

    if missing_files:
        f.write("Create missing files:\n\n")
        f.write("```text\n")
        for item in missing_files:
            f.write(f"{item}\n")
        f.write("```\n\n")

    f.write("Run backend:\n\n")
    f.write("```bash\n")
    f.write("uvicorn backend.main:app --reload\n")
    f.write("```\n\n")

    f.write("Run frontend:\n\n")
    f.write("```bash\n")
    f.write("streamlit run frontend/app.py\n")
    f.write("```\n\n")

    f.write("Run tests:\n\n")
    f.write("```bash\n")
    f.write("pytest tests -q\n")
    f.write("```\n")

ok(f"Report saved to: {report_path}")

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(f"Passed checks: {len(PASS)}")
print(f"Warnings: {len(WARN)}")
print(f"Missing/Failed checks: {len(FAIL)}")

if FAIL:
    print("\nMissing / failed items:")
    for item in FAIL:
        print(f"- {item}")

if missing_dirs:
    print("\nCreate missing directories with:")
    print(f"mkdir -p {' '.join(missing_dirs)}")

if missing_required:
    print("\nInstall missing required packages with:")
    print(f"pip install {' '.join(sorted(set(missing_required)))}")

if missing_files:
    print("\nYou need to create these missing files:")
    for item in missing_files:
        print(f"- {item}")
