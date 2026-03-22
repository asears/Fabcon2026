# CLI Reference

All `eda` commands share the `--vault` option which points to the root of your Obsidian vault.

Install: `pip install -e src/eda`

---

## `eda preprocess`

Parse vault markdown files into feature matrices ready for ML.

```
eda preprocess [OPTIONS]

Options:
  --vault PATH          Path to Obsidian vault root  [default: .]
  --output TEXT         Output path stem             [default: data/sessions]
  --format [csv|json|parquet|feather|excel|hdf5|sqlite]
                        Output format                [default: parquet]
  --max-tfidf INT       Maximum TF-IDF vocabulary    [default: 500]
  --ngram-min INT       Minimum n-gram size          [default: 1]
  --ngram-max INT       Maximum n-gram size          [default: 2]
  --no-workshops        Exclude workshops
  --no-tfidf            Skip TF-IDF features
  --stats / --no-stats  Print statistics             [default: True]
```

**Outputs:** `<output>.<ext>`, `<output>_tfidf.npz`, `<output>_tfidf_vocab.csv`

---

## `eda classify`

Train a text classifier to predict a session property (e.g., track, level, conference).

```
eda classify [OPTIONS]

Options:
  --vault PATH          Path to Obsidian vault root  [default: .]
  --target TEXT         Classification target        [default: track]
  --model [random-forest|svm|logistic|gradient-boosting|naive-bayes]
                        Classifier algorithm         [default: random-forest]
  --output TEXT         Output directory             [default: results/classify]
  --cv-folds INT        Cross-validation folds       [default: 5]
  --max-tfidf INT       Maximum TF-IDF features      [default: 500]
  --no-save-model       Skip saving model to disk
```

**Outputs:** `model.joblib`, `classification_report.json`, `confusion_matrix.png`,
`feature_importance.png` (RF/GBM only), `cv_summary.json`

**Valid targets:** `track`, `level`, `conference`, `level_name`, `session_type`, `day`

---

## `eda cluster`

Cluster sessions using unsupervised algorithms.

```
eda cluster [OPTIONS]

Options:
  --vault PATH          Path to Obsidian vault root  [default: .]
  --algorithm [kmeans|dbscan|agglomerative]
                        Clustering algorithm         [default: kmeans]
  --n-clusters INT      Number of clusters (0=auto)  [default: 0]
  --k-min INT           Min k for elbow search       [default: 2]
  --k-max INT           Max k for elbow search       [default: 25]
  --dbscan-eps FLOAT    DBSCAN epsilon               [default: 0.5]
  --dbscan-min-samples INT   DBSCAN min samples      [default: 3]
  --output TEXT         Output directory             [default: results/cluster]
  --write-back          Write ml_cluster to vault YAML
```

**Outputs:** `cluster_labels.csv`, `elbow.png` (KMeans), `silhouette.png` (KMeans),
`scatter.png`

---

## `eda reduce`

Project sessions to 2-D or 3-D using PCA, t-SNE, or UMAP.

```
eda reduce [OPTIONS]

Options:
  --vault PATH          Path to Obsidian vault root  [default: .]
  --method [pca|tsne|umap]
                        Reduction method             [default: pca]
  --components INT      Number of output dimensions  [default: 2]
  --color-by TEXT       Property for scatter colour  [default: track]
  --output TEXT         Output directory             [default: results/reduce]
  --tsne-perplexity FLOAT    t-SNE perplexity        [default: 30.0]
  --tsne-iterations INT      t-SNE iterations        [default: 1000]
  --umap-neighbors INT       UMAP n_neighbors        [default: 15]
  --pca-whiten          Whiten PCA output
```

**Outputs:** `embeddings.csv`, `scatter.png`

> UMAP requires `pip install umap-learn` (optional extra in `src/eda`).

---

## `eda model-select`

Systematic hyperparameter search across multiple classifiers.

```
eda model-select [OPTIONS]

Options:
  --vault PATH            Path to Obsidian vault root  [default: .]
  --target TEXT           Classification target         [default: track]
  --estimators TEXT       Estimators to search          [multiple, default: all]
  --search [random|grid]  Search strategy               [default: random]
  --n-iter INT            Iterations per model          [default: 20]
  --cv-folds INT          Cross-validation folds        [default: 5]
  --scoring TEXT          Scoring metric                [default: f1_weighted]
  --output TEXT           Output directory              [default: results/model_select]
```

**Outputs:** `cv_results.csv`, `best_params.json`, `best_model.joblib`, `cv_comparison.png`

---

## Common Patterns

```bash
# Full pipeline
eda preprocess --vault . --output src/notebooks/data/sessions --format parquet
eda classify   --vault . --target track --model logistic --output src/notebooks/results
eda cluster    --vault . --algorithm kmeans --output src/notebooks/data
eda reduce     --vault . --method pca --components 2 --output src/notebooks/data
eda model-select --vault . --target level --n-iter 10 --cv-folds 5 --output src/notebooks/results
```
