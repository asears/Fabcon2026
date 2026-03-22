"""FastAPI REST service for FabCon 2026 session analysis.

Run with:
    uvicorn viz.api:app --reload --port 8000
or:
    viz-api
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import joblib
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Bootstrap: allow `src/eda` on PYTHONPATH so we can reuse the loader
# ---------------------------------------------------------------------------
_EDA_SRC = Path(__file__).parent.parent.parent / "eda"
if str(_EDA_SRC) not in sys.path:
    sys.path.insert(0, str(_EDA_SRC))

from eda.data.loader import load_vault  # noqa: E402

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="FabCon 2026 Session Analysis API",
    version="0.1.0",
    description="Explore session data, run predictions, and retrieve cluster/embedding results.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

VAULT_PATH: str = os.environ.get("FABCON_VAULT", str(Path(__file__).parent.parent.parent.parent))
_df_cache: Any = None


def _get_df():
    global _df_cache
    if _df_cache is None:
        _df_cache = load_vault(VAULT_PATH)
    return _df_cache


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    texts: list[str]
    model_path: str = "../../notebooks/models/best_model_track.joblib"


class ClusterRequest(BaseModel):
    n_clusters: int = 8
    algorithm: str = "kmeans"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {"status": "ok", "service": "FabCon 2026 Analysis API"}


@app.get("/api/sessions")
def get_sessions(
    conference: str | None = Query(None),
    track: str | None = Query(None),
    day: str | None = Query(None),
    status: str | None = Query(None),
    interest_min: int = Query(0, ge=0, le=5),
    limit: int = Query(500, ge=1, le=1000),
):
    """Return sessions filtered by optional query parameters."""
    df = _get_df().copy()

    if conference:
        df = df[df["conference"].str.upper() == conference.upper()]
    if track:
        df = df[df["track"].str.lower().str.contains(track.lower(), na=False)]
    if day:
        df = df[df["day"].str.lower() == day.lower()]
    if status:
        df = df[df["status"].str.lower() == status.lower()]
    if interest_min > 0:
        df = df[df["interest"].fillna(0) >= interest_min]

    cols = ["title", "track", "day", "start_time", "room", "level", "conference", "status", "interest", "speakers"]
    available = [c for c in cols if c in df.columns]
    return df[available].head(limit).fillna("").to_dict(orient="records")


@app.get("/api/tracks")
def get_tracks():
    df = _get_df()
    counts = df["track"].value_counts().reset_index()
    counts.columns = ["track", "count"]
    return counts.to_dict(orient="records")


@app.get("/api/speakers")
def get_speakers(top_n: int = Query(50, ge=1, le=500)):
    df = _get_df()
    from collections import Counter

    speaker_list: list[str] = []
    for row in df["speakers"].dropna():
        if isinstance(row, list):
            speaker_list.extend(row)
        elif isinstance(row, str):
            speaker_list.append(row)

    counts = Counter(speaker_list).most_common(top_n)
    return [{"speaker": s, "sessions": c} for s, c in counts]


@app.post("/api/predict")
def predict(req: PredictRequest):
    """Run track prediction on provided texts using a saved model."""
    model_path = Path(req.model_path)
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model not found: {model_path}")

    pipeline = joblib.load(model_path)
    preds = pipeline.predict(req.texts)
    probs = None
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(req.texts).tolist()

    return {"predictions": preds.tolist(), "probabilities": probs}


@app.get("/api/embeddings")
def get_embeddings(path: str = "../../notebooks/data/embeddings.csv"):
    """Return pre-computed 2-D/3-D embeddings from the reduce command."""
    import pandas as pd

    emb_path = Path(path)
    if not emb_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Embeddings file not found: {emb_path}. "
                   "Run `eda reduce --vault <vault>` first.",
        )

    df = pd.read_csv(emb_path)
    # Safety: drop text column (can be large) if present
    df = df.drop(columns=["text", "description"], errors="ignore")
    return df.fillna("").to_dict(orient="records")


@app.get("/api/clusters")
def get_clusters(path: str = "../../notebooks/data/cluster_labels.csv"):
    """Return pre-computed cluster labels from the cluster command."""
    import pandas as pd

    clust_path = Path(path)
    if not clust_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Cluster file not found: {clust_path}. "
                   "Run `eda cluster --vault <vault>` first.",
        )

    df = pd.read_csv(clust_path)
    return df.fillna("").to_dict(orient="records")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def start():
    uvicorn.run("viz.api:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    start()
