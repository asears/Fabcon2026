"""Streamlit dashboard for FabCon 2026 session analysis.

Run with:
    streamlit run viz/dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# Bootstrap eda package
_EDA_SRC = Path(__file__).parent.parent.parent / "eda"
if str(_EDA_SRC) not in sys.path:
    sys.path.insert(0, str(_EDA_SRC))

from eda.data.loader import load_vault  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FabCon 2026 Session Analysis",
    page_icon="🔷",
    layout="wide",
    initial_sidebar_state="expanded",
)

VAULT_DEFAULT = str(Path(__file__).parent.parent.parent.parent)

# ---------------------------------------------------------------------------
# Sidebar — vault path + caching
# ---------------------------------------------------------------------------
st.sidebar.title("FabCon 2026")
vault_path = st.sidebar.text_input("Vault path", value=VAULT_DEFAULT)


@st.cache_data(show_spinner="Loading vault…")
def get_data(vault: str) -> pd.DataFrame:
    return load_vault(vault)


df = get_data(vault_path)

st.sidebar.markdown(f"**{len(df)} sessions loaded**")

# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
page = st.sidebar.selectbox(
    "Page",
    ["Session Browser", "Track Explorer", "Cluster Explorer", "Dimension Projection", "Model Predictions"],
)

# ============================================================
# PAGE 1 — Session Browser
# ============================================================
if page == "Session Browser":
    st.header("Session Browser")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        conferences = ["All"] + sorted(df["conference"].dropna().unique().tolist())
        conf_filter = st.selectbox("Conference", conferences)
    with col2:
        tracks = ["All"] + sorted(df["track"].dropna().unique().tolist())
        track_filter = st.selectbox("Track", tracks)
    with col3:
        days = ["All"] + sorted(df["day"].dropna().unique().tolist())
        day_filter = st.selectbox("Day", days)
    with col4:
        statuses = ["All"] + sorted(df["status"].dropna().unique().tolist())
        status_filter = st.selectbox("Status", statuses)

    interest_min = st.slider("Min Interest", 0, 5, 0)

    mask = pd.Series([True] * len(df), index=df.index)
    if conf_filter != "All":
        mask &= df["conference"].str.upper() == conf_filter.upper()
    if track_filter != "All":
        mask &= df["track"] == track_filter
    if day_filter != "All":
        mask &= df["day"] == day_filter
    if status_filter != "All":
        mask &= df["status"] == status_filter
    if interest_min > 0:
        mask &= df["interest"].fillna(0) >= interest_min

    display_cols = [c for c in ["title", "track", "day", "start_time", "room",
                                  "level_name", "conference", "status", "interest"]
                    if c in df.columns]
    filtered = df[mask][display_cols]
    st.markdown(f"**{len(filtered)} sessions match filters**")
    st.dataframe(filtered, use_container_width=True, height=500)

# ============================================================
# PAGE 2 — Track Explorer
# ============================================================
elif page == "Track Explorer":
    st.header("Track Explorer")

    col_a, col_b = st.columns(2)

    with col_a:
        track_counts = df["track"].value_counts().reset_index()
        track_counts.columns = ["track", "count"]
        fig = px.bar(track_counts.sort_values("count"), x="count", y="track",
                     orientation="h", title="Sessions per Track",
                     color="count", color_continuous_scale="Blues")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        level_counts = df["level_name"].value_counts().reset_index()
        level_counts.columns = ["level", "count"]
        fig = px.pie(level_counts, names="level", values="count",
                     title="Sessions by Level", hole=0.3)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Track × Day Heatmap")
    if "day" in df.columns and "track" in df.columns:
        pivot = df.pivot_table(index="track", columns="day",
                               values="title", aggfunc="count", fill_value=0)
        day_order = [d for d in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
                     if d in pivot.columns]
        pivot = pivot[day_order] if day_order else pivot
        fig = px.imshow(pivot, color_continuous_scale="Blues",
                        title="Session Count per Track per Day")
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGE 3 — Cluster Explorer
# ============================================================
elif page == "Cluster Explorer":
    st.header("Cluster Explorer")

    clust_path_default = str(Path(__file__).parent.parent.parent / "notebooks" / "data" / "cluster_labels.csv")
    clust_path = st.text_input("Cluster labels CSV", value=clust_path_default)

    if Path(clust_path).exists():
        clust_df = pd.read_csv(clust_path)
        merged = df.reset_index().merge(clust_df, left_on="file", right_on="file", how="inner") \
            if "file" in clust_df.columns else clust_df

        n_clusters = merged["cluster"].nunique() if "cluster" in merged.columns else 0
        st.markdown(f"**{n_clusters} clusters found across {len(merged)} sessions**")

        if "cluster" in merged.columns and "track" in merged.columns:
            crosstab = pd.crosstab(merged["track"], merged["cluster"])
            fig = px.imshow(crosstab, title="Track × Cluster Heatmap",
                            color_continuous_scale="Viridis")
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(merged, use_container_width=True, height=400)
    else:
        st.warning(f"No cluster file found at `{clust_path}`. Run `eda cluster --vault <vault>` first.")

# ============================================================
# PAGE 4 — Dimension Projection
# ============================================================
elif page == "Dimension Projection":
    st.header("Dimensionality Reduction")

    emb_path_default = str(Path(__file__).parent.parent.parent / "notebooks" / "data" / "embeddings.csv")
    emb_path = st.text_input("Embeddings CSV", value=emb_path_default)

    if Path(emb_path).exists():
        emb_df = pd.read_csv(emb_path)
        dim_cols = [c for c in emb_df.columns if c.startswith("dim_") or c in ["x", "y", "z",
                    "pca_0", "pca_1", "pca_2", "tsne_0", "tsne_1", "umap_0", "umap_1"]]

        color_by = st.selectbox("Colour by", ["track", "level_name", "conference", "day", "status"], index=0)
        if color_by not in emb_df.columns and color_by in df.columns:
            emb_df[color_by] = df[color_by].values[:len(emb_df)]

        if len(dim_cols) >= 2:
            x_col, y_col = dim_cols[0], dim_cols[1]
            plot_3d = len(dim_cols) >= 3 and st.checkbox("3-D plot")
            if plot_3d:
                fig = px.scatter_3d(emb_df, x=x_col, y=y_col, z=dim_cols[2],
                                    color=color_by if color_by in emb_df.columns else None,
                                    hover_data=["title"] if "title" in emb_df.columns else None,
                                    title="Embedding (3D)")
            else:
                fig = px.scatter(emb_df, x=x_col, y=y_col,
                                 color=color_by if color_by in emb_df.columns else None,
                                 hover_data=["title"] if "title" in emb_df.columns else None,
                                 title="Embedding (2D)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Embedding CSV does not contain expected dimension columns.")
    else:
        st.warning(f"No embeddings file found at `{emb_path}`. Run `eda reduce --vault <vault>` first.")

# ============================================================
# PAGE 5 — Model Predictions
# ============================================================
elif page == "Model Predictions":
    st.header("Predict Track from Text")

    model_path_default = str(Path(__file__).parent.parent.parent / "notebooks" / "models" / "best_model_track.joblib")
    model_path = st.text_input("Model (.joblib)", value=model_path_default)

    session_text = st.text_area(
        "Enter session title + description",
        height=150,
        placeholder="e.g. Real-time streaming with Microsoft Fabric and Event Hubs...",
    )

    if st.button("Predict") and session_text.strip():
        if not Path(model_path).exists():
            st.error(f"Model file not found: `{model_path}`")
        else:
            import joblib as jl
            pipeline = jl.load(model_path)
            pred = pipeline.predict([session_text])[0]
            st.success(f"**Predicted track:** {pred}")

            if hasattr(pipeline, "predict_proba"):
                proba = pipeline.predict_proba([session_text])[0]
                classes = pipeline.classes_ if hasattr(pipeline, "classes_") else [str(i) for i in range(len(proba))]
                prob_df = pd.DataFrame({"track": classes, "probability": proba}).sort_values("probability", ascending=False)
                fig = px.bar(prob_df.head(10), x="probability", y="track",
                             orientation="h", title="Top-10 Track Probabilities")
                st.plotly_chart(fig, use_container_width=True)


def main():
    pass  # Streamlit entry via CLI; code runs at module level
