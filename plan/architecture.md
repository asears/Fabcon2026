# FabCon 2026 ML Analysis — Architecture

**Date created:** 2026-03-18
**Status:** In progress

## Overview

Build a scikit-learn ML pipeline for FabCon 2026 session analysis, surfaced as:
- **CLI tool** (`src/eda`) — Click-based, offline analysis
- **Notebooks** (`src/notebooks`) — Exploratory + reproducible research
- **Web app** (`src/viz`) — FastAPI REST API + Streamlit dashboard

---

## Data Flow

```
Obsidian Vault (Sessions/*.md, Workshops/*.md)
  └─ YAML frontmatter + markdown body
       │
       ▼
   loader.py ──► pandas DataFrame
       │
       ▼
 preprocess cmd ──► TF-IDF sparse matrix (.npz)
                 └► encoded DataFrame (.parquet/.csv/.xlsx/.h5)
       │
       ├──► classify cmd  ──► model.joblib + classification_report.json
       ├──► cluster cmd   ──► cluster_labels.csv + silhouette.png
       ├──► reduce cmd    ──► 2D/3D embeddings.csv + scatter.png
       └──► model-select  ──► cv_results.csv + comparison.png
```

---

## Component Details

### src/eda (Click CLI)

| Command | Algorithm options | Input | Output |
|---------|------------------|-------|--------|
| `preprocess` | TF-IDF + LabelEncoder | vault path | parquet/csv/excel/hdf5/json |
| `classify`  | RF, SVC, LR, GBM, NB | vault + target | model.joblib + report |
| `cluster`   | KMeans, DBSCAN, Agglomerative | vault | labels.csv + plots |
| `reduce`    | PCA, t-SNE, UMAP | vault | embeddings.csv + scatter |
| `model-select` | GridSearchCV / RandomizedSearchCV | vault + estimator | cv_results + best_params |

### src/viz (FastAPI + Streamlit)

- **FastAPI** (`viz/api.py`): REST endpoints for /sessions, /predict, /cluster
- **Streamlit** (`viz/dashboard.py`): interactive exploration dashboard

### Obsidian Bases (new)

| Base | Purpose |
|------|---------|
| `Session Clusters.base` | Group sessions by ML cluster label |
| `Level Analysis.base` | Breakdown by technical level + interest |
| `Sessions by Conference.base` | FABCON vs SQLCON comparison |
| `Speaker Network.base` | Speaker metadata + session counts |

---

## pandas Optional Dependencies Used

From [pandas pyproject.toml](https://github.com/pandas-dev/pandas/blob/main/pyproject.toml):

| Extra | Package | Used for |
|-------|---------|---------|
| `pyarrow` | pyarrow>=13 | Parquet I/O |
| `performance` | numba, bottleneck, numexpr | Faster aggregations |
| `computation` | scipy | Statistics, sparse math |
| `excel` | openpyxl, xlsxwriter | Excel output |
| `hdf5` | tables | HDF5/PyTables output |
| `output-formatting` | tabulate | CLI table output |
| `plot` | matplotlib | All charts |
| `parquet` | fastparquet | Alternative Parquet engine |
| `sql-other` | SQLAlchemy | SQLite/Postgres output |

---

## Key Design Decisions

1. **TF-IDF on `title + description`** — The richest available text signal
2. **Target encoding**: `track` (19 classes), `level` (100/200/300/400), `conference` (2 classes)
3. **Sparse matrix stored as NPZ** — scikit-learn pipelines work natively with scipy sparse
4. **Joblib for model serialization** — Standard scikit-learn practice
5. **Rich for CLI output** — Pretty tables, progress bars
6. **Click + subcommands** — Each ML operation is a standalone command
