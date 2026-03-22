from __future__ import annotations

import ast

import numpy as np
import pandas as pd


def parse_speakers(value) -> list[str]:
    """Normalize speaker values serialized as list, stringified list, or CSV string."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except Exception:
        pass
    return [part.strip() for part in str(value).split(",") if part.strip()]


def extract_cluster_terms(vectorizer, matrix, labels, top_n: int = 8) -> pd.DataFrame:
    """Return top TF-IDF terms by cluster centroid signal."""
    terms = np.array(vectorizer.get_feature_names_out())
    rows = []
    for cluster_id in sorted(set(int(v) for v in labels)):
        mask = np.asarray(labels) == cluster_id
        if not mask.any():
            continue
        scores = matrix[mask].mean(axis=0).A1
        top_idx = scores.argsort()[::-1][:top_n]
        rows.append(
            {
                "cluster_id": int(cluster_id),
                "top_terms": ", ".join(terms[top_idx]),
            }
        )
    return pd.DataFrame(rows)
