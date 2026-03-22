# viz — FabCon 2026 Analysis Dashboard

Interactive dashboard for exploring FabCon 2026 sessions, cluster results, and ML model predictions.

## Components

| Component | Tech | Purpose |
|-----------|------|---------|
| **REST API** | FastAPI | Serve session data, predictions, embeddings |
| **Dashboard** | Streamlit | Interactive exploration UI |

## Quick Start

```bash
cd src/viz
pip install -e .
```

### Run the API

```bash
# Default port 8000
viz-api

# Or directly with uvicorn
uvicorn viz.api:app --reload --port 8000
```

API docs: http://localhost:8000/docs

### Run the Dashboard

```bash
streamlit run viz/dashboard.py
```

Dashboard: http://localhost:8501

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/sessions` | List sessions with optional filters |
| `GET` | `/api/tracks` | Track counts |
| `GET` | `/api/speakers` | Top speakers by session count |
| `GET` | `/api/embeddings` | Pre-computed 2-D/3-D embeddings |
| `GET` | `/api/clusters` | Pre-computed cluster labels |
| `POST` | `/api/predict` | Predict track for a list of texts |

### Session Filters

```
GET /api/sessions?conference=FABCON&track=Power+BI&day=Wednesday&status=Attending&interest_min=4&limit=100
```

### Predict Example

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Real-time streaming with Microsoft Fabric and Event Hubs"]}'
```

## Dashboard Pages

1. **Session Browser** — Filterable table of all sessions
2. **Track Explorer** — Bar charts and heatmaps by track/day/level
3. **Cluster Explorer** — Cluster × Track heatmap (requires running `eda cluster` first)
4. **Dimension Projection** — 2-D/3-D scatter plots (requires running `eda reduce` first)
5. **Model Predictions** — Interactive prediction UI (requires a trained model)

## Prerequisites

Run the EDA CLI commands first to generate input files:

```bash
cd ../..

# Generate embeddings
eda reduce --vault . --method pca --components 2 --output src/notebooks/data/embeddings

# Generate cluster labels
eda cluster --vault . --algorithm kmeans --output src/notebooks/data

# Train best model
eda model-select --vault . --output src/notebooks/models
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FABCON_VAULT` | `../../..` | Path to the Obsidian vault |
