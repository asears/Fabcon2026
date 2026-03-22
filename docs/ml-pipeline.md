# ML Pipeline Guide

End-to-end workflow from raw Obsidian vault data to trained models and visualisations.

## Data Flow

```
Obsidian Vault (.md files)
        │
        ▼
  eda preprocess
        │
        ├── sessions.parquet  (structured features)
        ├── sessions_tfidf.npz  (sparse TF-IDF matrix)
        └── sessions_tfidf_vocab.csv
        │
        ├──► eda classify    → model.joblib + confusion_matrix.png
        ├──► eda cluster     → cluster_labels.csv + scatter.png
        ├──► eda reduce      → embeddings.csv + scatter.png
        └──► eda model-select → best_model.joblib + cv_comparison.png
```

## Feature Engineering

The `preprocess` command builds two kinds of features:

### Text Features (TF-IDF)
- Source: `title` + `## Description` body from each session file
- Vectorise with `TfidfVectorizer(stop_words='english', sublinear_tf=True)`
- Output: sparse `NPZ` matrix + vocabulary CSV

### Structured Features
| Field | Type | Notes |
|-------|------|-------|
| `track` | categorical | 19 tracks (FABCON + SQLCON) |
| `level` | numeric | 100/200/300/400 |
| `conference` | categorical | FABCON or SQLCON |
| `day` | categorical | Monday–Friday |
| `session_type` | categorical | Breakout, Workshop, CORENOTE, etc. |
| `duration` | numeric | minutes |
| `interest` | numeric | 1–5 user rating |
| `status` | categorical | Considering/Attending/Skip |

## Classification

Target options and typical accuracy with a default Random Forest:

| Target | Classes | Expected F1 |
|--------|---------|-------------|
| `track` | 19 | ~0.75–0.85 |
| `level` | 4 | ~0.65–0.75 |
| `conference` | 2 | ~0.95+ |
| `level_name` | 4 | ~0.65–0.75 |
| `session_type` | ~8 | ~0.80–0.90 |

> Results depend on vault size and description quality.

## Hyperparameter Tuning

`eda model-select` runs `RandomizedSearchCV` (or `GridSearchCV`) across:
- `random-forest` — n_estimators, max_depth, min_samples_split
- `logistic` — C, solver, max_iter
- `svm` / `linear-svc` — C
- `naive-bayes` — alpha

Results are saved as `cv_results.csv`. The winning model is saved as `best_model.joblib`.

## Clustering

KMeans auto-infers `k` from the number of unique tracks (default) or you can specify it.
The elbow and silhouette plots help you choose the right `k`.

Use `--write-back` to add a `ml_cluster` property to each session's YAML frontmatter,
then create an Obsidian Base filtered on `ml_cluster`.

## Dimensionality Reduction

| Method | Best For |
|--------|---------|
| PCA | Speed, linear structure, feature weight explanation |
| t-SNE | Cluster visualisation (non-linear) |
| UMAP | Topology-preserving embeddings (fastest non-linear) |

All output an `embeddings.csv` consumed by the Streamlit dashboard.

## Recommended Workflow

```bash
# 1. Preprocess once (run again after vault sync)
eda preprocess --vault . --output src/notebooks/data/sessions

# 2. Explore notebooks 01-03
# → Open src/notebooks/01_data_exploration.ipynb

# 3. Model selection
eda model-select --vault . --target track --n-iter 20 \
    --output src/notebooks/results/model_select

# 4. Cluster and write back to vault
eda cluster --vault . --algorithm kmeans --write-back \
    --output src/notebooks/data

# 5. Reduce for scatter plots
eda reduce --vault . --method pca --components 2 \
    --output src/notebooks/data

# 6. Launch dashboard
streamlit run src/viz/viz/dashboard.py
```
