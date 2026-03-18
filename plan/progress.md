# Progress Tracker

## Session: 2026-03-18

### Completed
- [x] Read existing codebase (src/eda has stub main.py + empty pyproject.toml)
- [x] Fetched pandas optional deps from GitHub (pyarrow, openpyxl, tables, scipy, etc.)
- [x] Architecture planned (see architecture.md)
- [x] plan/ folder created

### In Progress
- [ ] src/eda — Click CLI with preprocess/classify/cluster/reduce/model-select
- [ ] src/notebooks — 6 Jupyter notebooks
- [ ] src/viz — FastAPI + Streamlit app
- [ ] docs/ — Documentation
- [ ] Obsidian bases + canvases

### File Inventory (target)

#### src/eda/eda/ (new package)
- [ ] `__init__.py`
- [ ] `main.py` (CLI groups)
- [ ] `commands/__init__.py`
- [ ] `commands/preprocess.py`
- [ ] `commands/classify.py`
- [ ] `commands/cluster.py`
- [ ] `commands/reduce.py`
- [ ] `commands/model_select.py`
- [ ] `data/__init__.py`
- [ ] `data/loader.py`
- [ ] `data/schema.py`
- [ ] `utils/__init__.py`
- [ ] `utils/plotting.py`
- [ ] `utils/io.py`

#### Modified
- [ ] `src/eda/pyproject.toml` (add all deps + scripts entry)
- [ ] `src/eda/main.py` (shim to eda.main:cli)
- [ ] `src/eda/README.md` (usage docs)

#### src/notebooks/
- [ ] `01_data_exploration.ipynb`
- [ ] `02_preprocessing.ipynb`
- [ ] `03_classification.ipynb`
- [ ] `04_clustering.ipynb`
- [ ] `05_dimensionality_reduction.ipynb`
- [ ] `06_model_selection.ipynb`

#### src/viz/
- [ ] `pyproject.toml`
- [ ] `README.md`
- [ ] `viz/__init__.py`
- [ ] `viz/api.py` (FastAPI)
- [ ] `viz/dashboard.py` (Streamlit)

#### docs/
- [ ] `index.md`
- [ ] `cli-reference.md`
- [ ] `ml-pipeline.md`
- [ ] `obsidian-analysis.md`

#### Obsidian
- [ ] `Bases/Session Clusters.base`
- [ ] `Bases/Level Analysis.base`
- [ ] `Bases/Sessions by Conference.base`
- [ ] `Bases/Speaker Network.base`
- [ ] `Session Analysis.canvas`
- [ ] `Speaker Network.canvas`
- [ ] `ML Analysis.canvas`
