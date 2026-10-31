# Project Check Report

Generated at: 2026-10-31T12:10:10.632837


======================================================================
PROJECT HEALTH CHECK
======================================================================
Current directory: D:\fairscore (1)\fairscore

======================================================================
1. Python Version
======================================================================
 Python version is compatible: 3.14.6

======================================================================
2. Project Structure
======================================================================
 Directory exists: data
 Directory exists: data/raw
 Directory exists: data/extracted
 Directory exists: models
 Directory exists: backend
 Directory exists: frontend
 Directory exists: scripts
 Directory exists: tests
 Directory exists: outputs
 Directory exists: outputs/metrics
 Directory exists: outputs/reports
 Directory exists: outputs/logs
 File exists: requirements.txt
 File exists: README.md
 File exists: backend/main.py
 File exists: frontend/app.py
 File exists: tests/test_backend.py

======================================================================
3. Data Files
======================================================================
 Found 28 data file(s).
 Found 2 .data file(s)
 Found 1 .dat file(s)
 Found 2 .txt file(s)
 Found 1 .xls file(s)
 Found 1 .csv file(s)
 Found 20 .json file(s)
 Found 1 .asc file(s)
 No ZIP files found. If your datasets are zipped, upload them.

======================================================================
4. Dependencies
======================================================================
 Required package available: sklearn
 Required package available: pandas
 Required package available: numpy
 Required package available: joblib
 Required package available: tqdm
 Required package available: fastapi
 Required package available: pydantic
 Required package available: streamlit
 Required package available: uvicorn
 Required package available: pytest
 Required package available: httpx
 Optional package available: openpyxl
 Optional package available: xlrd
 Optional package available: xgboost
 Optional package available: shap
 Optional package available: pyarrow

======================================================================
5. Python Syntax Check
======================================================================
 All 35 Python file(s) passed syntax check.

======================================================================
6. FastAPI Static Check
======================================================================
 Backend file found: backend\main.py
 FastAPI instance detected in backend file.
 Backend route found: GET /health
 Backend route found: GET /models
 Backend route found: POST /predict
 Backend route found: POST /explain
 Backend route found: GET /metrics
 Backend route found: GET /fairness

======================================================================
7. FastAPI Dynamic Check
======================================================================
 FastAPI app imported from module: backend.main
 GET /health responded with status 200
 GET /models responded with status 200
 GET /metrics responded with status 200
 GET /fairness responded with status 200
 POST /predict responded with status 200
 POST /explain responded with status 200

======================================================================
8. Streamlit Frontend Check
======================================================================
 frontend/app.py exists.
 Frontend contains consent content.
 Frontend contains score content.
 Frontend contains explanation content.
 Frontend contains dashboard content.
 Frontend contains bias audit content.

======================================================================
9. Model Files Check
======================================================================
 Found 6 model file(s).
 Model loaded successfully and supports predict_proba: australian_model.joblib

======================================================================
10. Pytest Check
======================================================================
 tests directory exists.
 Pytest can collect tests successfully.

======================================================================
11. requirements.txt Check
======================================================================
 requirements.txt contains all required core packages.

======================================================================
12. Final Report
======================================================================

## Summary

- Passed checks: 69
- Warnings: 1
- Missing/Failed checks: 0

## Warnings

- No ZIP files found. If your datasets are zipped, upload them.

## Suggested Fix Commands

Run backend:

```bash
uvicorn backend.main:app --reload
```

Run frontend:

```bash
streamlit run frontend/app.py
```

Run tests:

```bash
pytest tests -q
```
