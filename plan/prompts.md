# Prompts & Interim States

## 2026-03-18 — Initial Build

### User Prompt
> Create click cli tool to do classification of the sessions using scikit learn trained model,
> create separate command for clustering, create separate command for dimensionality reduction,
> create separate command for model selection, create separate command for preprocessing.
> Use latest numpy, scipy, matplotlib and gather the optional dependencies from
> https://github.com/pandas-dev/pandas/blob/main/pyproject.toml to identify other
> possibilities for inputs and outputs. Use subagents, create plan folder to track state and
> memory. Ensure src/notebooks (ipynb) and src/eda (cli tools) and src/viz (fastapi streamlit
> app) and docs/ are updated. Ensure prompt and plans and interim states are captured.
>
> Let's also learn more about obsidian, ensure additional bases and canvases are setup for the
> sessions and speakers for better analysis. We will use scikit models for this exercise.

### Key Decisions Made

1. **Package structure**: `src/eda/eda/` (package inside project root) — clean separation
   of package code from project config.

2. **Text features**: TF-IDF on `title + description` combined — most signal for track/level
   prediction. ngram_range=(1,2), max_features=500, sublinear_tf=True.

3. **Classification targets**: `track` (multi-class, 19 classes), `level` (ordinal 100-400),
   `conference` (binary FABCON/SQLCON).

4. **Clustering algorithm choice**: KMeans for interpretability + DBSCAN for noise handling
   + Agglomerative for hierarchy. Default n_clusters derived from num unique tracks.

5. **Dimensionality reduction**: PCA (fast, linear), t-SNE (2D/3D visualization),
   UMAP (optional dep umap-learn — non-linear, topology-preserving).

6. **Model selection**: RandomizedSearchCV over param grids for RF, SVC, LR, GBM with
   stratified k-fold CV.

7. **I/O formats via pandas optional deps**:
   - pyarrow → Parquet (default, best for ML workflows)
   - openpyxl → Excel (for stakeholders)
   - tables → HDF5 (large arrays)
   - scipy → NPZ sparse matrices (TF-IDF)
   - fastparquet → alternative Parquet engine
   - SQLAlchemy → SQLite output option

8. **Obsidian bases**: Use `file.hasLink(this.file)` pattern for speaker/track pages;
   new bases for ML cluster grouping (after write-back from `cluster --write-back`).

### Interim State: Data Schema

FabCon session YAML fields used as ML features:
```
text features : title, description (TF-IDF)
categorical   : track, day, session_type, level_name, conference, status
numeric       : level (100/200/300/400), duration (minutes), interest (1-5, user)
multi-label   : audience, speakers, tags
date          : date (YYYY-MM-DD)
```

Track labels (19): Admin and Governance, Power BI, Data Engineering, Data Warehousing,
Data Science, Data Integration, Real-Time Intelligence, OneLake, Microsoft Purview,
Data Dev, Developer Experiences, Microsoft Foundry, Power Platform,
SQL Server, Azure SQL, SQL in Fabric, Cosmos DB, PostgreSQL, MySQL

### Pandas Optional Dependencies Captured

From https://github.com/pandas-dev/pandas/blob/main/pyproject.toml:
- pyarrow (parquet, feather)
- fastparquet (parquet alternative)
- openpyxl, xlrd, xlsxwriter (excel)
- tables (hdf5)
- SQLAlchemy (sql)
- scipy (computation)
- numba (performance JIT)
- bottleneck (fast NaN-aggregations)
- numexpr (fast expression evaluation)
- matplotlib (plotting)
- xarray (N-dimensional arrays)
- fsspec (filesystem abstraction)
- s3fs (AWS S3)
- gcsfs (GCP GCS)
- beautifulsoup4 / lxml (HTML/XML parsing)
- tabulate (output formatting)
- jinja2 (template output)
- pyarrow/pyiceberg (iceberg tables)
