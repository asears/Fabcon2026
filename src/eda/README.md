# eda — FabCon 2026 Session Analysis CLI

A scikit-learn powered CLI for classifying, clustering, and visualising
FabCon / SQLCON 2026 conference sessions stored in an Obsidian vault.

## Installation

```bash
cd src/eda
uv sync                    # install core deps
uv sync --extra umap       # optional: UMAP dimensionality reduction
uv sync --extra hdf5       # optional: HDF5 I/O
uv sync --extra all        # all optional extras
```

## Commands

| Command | Description |
|---------|-------------|
| `eda preprocess` | Load vault sessions → TF-IDF + encoded DataFrame |
| `eda classify`   | Train classifier (track / level / conference) |
| `eda cluster`    | KMeans / DBSCAN / Agglomerative clustering |
| `eda reduce`     | PCA / t-SNE / UMAP dimensionality reduction |
| `eda model-select` | Hyperparameter search + model comparison |

## Quick Start

```bash
# From vault root (c:\projects\fab\Fabcon2026)
uv run --project src/eda eda preprocess --vault . --output src/eda/data/sessions
uv run --project src/eda eda classify --vault . --target track
uv run --project src/eda eda cluster --vault . --algorithm kmeans
uv run --project src/eda eda reduce --vault . --method tsne --color-by track
uv run --project src/eda eda model-select --vault . --target track
```

## Output Formats

Supported via **pandas optional dependencies**:

| Flag | Format | Dependency |
|------|--------|-----------|
| `--format parquet` | Parquet (default) | pyarrow |
| `--format csv` | CSV | built-in |
| `--format excel` | Excel .xlsx | openpyxl |
| `--format hdf5` | HDF5 | tables (`pip install eda[hdf5]`) |
| `--format sqlite` | SQLite | SQLAlchemy (`pip install eda[sql]`) |
| `--format json` | JSON | built-in |
| `--format feather` | Feather | pyarrow |

## Package Structure

```
eda/
├── main.py            CLI entry point (click groups)
├── commands/
│   ├── preprocess.py  Load + TF-IDF + encode
│   ├── classify.py    Supervised classification
│   ├── cluster.py     Unsupervised clustering
│   ├── reduce.py      Dimensionality reduction
│   └── model_select.py Hyperparameter search
├── data/
│   ├── loader.py      Vault markdown parser
│   └── schema.py      Feature / target definitions
└── utils/
	├── io.py          Multi-format DataFrame I/O
	└── plotting.py    Matplotlib chart helpers
```
