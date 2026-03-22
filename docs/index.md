# FabCon 2026 Session Analysis

ML-powered analysis of Microsoft Fabric and SQL conference sessions using the Obsidian vault as a data source.

## Overview

This project provides a complete data science pipeline on top of the **FabCon 2026** Obsidian vault:

- **CLI tool** (`src/eda`) — Five scikit-learn commands: preprocess, classify, cluster, reduce, model-select
- **Jupyter notebooks** (`src/notebooks`) — Six notebooks walking through the full EDA → modelling workflow
- **Dashboard** (`src/viz`) — FastAPI REST API + Streamlit interactive explorer

## Directory Structure

```
src/
├── eda/                    # Click CLI package
│   ├── eda/
│   │   ├── commands/       # preprocess, classify, cluster, reduce, model_select
│   │   ├── data/           # vault loader, schema definitions
│   │   └── utils/          # I/O, plotting helpers
│   └── pyproject.toml
├── notebooks/              # Jupyter analysis notebooks 01-06
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_classification.ipynb
│   ├── 04_clustering.ipynb
│   ├── 05_dimensionality_reduction.ipynb
│   └── 06_model_selection.ipynb
└── viz/                    # FastAPI + Streamlit dashboard
    ├── viz/
    │   ├── api.py          # REST API
    │   └── dashboard.py    # Streamlit UI
    └── pyproject.toml
```

## Quick Start

```bash
# 1. Install EDA CLI
cd src/eda && pip install -e . && cd ../..

# 2. Preprocess sessions
eda preprocess --vault . --output src/notebooks/data/sessions

# 3. Classify by track
eda classify --vault . --target track --output src/notebooks/results

# 4. Run dashboard
cd src/viz && pip install -e . && cd ../..
streamlit run src/viz/viz/dashboard.py
```

## Documentation

| Doc | Purpose |
|-----|---------|
| [CLI Reference](cli-reference.md) | All commands and flags |
| [ML Pipeline](ml-pipeline.md) | End-to-end workflow guide |
| [Obsidian Analysis](obsidian-analysis.md) | Bases, canvases, and analysis tips |
