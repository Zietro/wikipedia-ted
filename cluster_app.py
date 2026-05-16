"""
cluster_app.py — Project 2 Streamlit UI
=========================================
Wikipedia Semi-structured Infobox Document Clustering Tool
COE 543/743 — Lebanese American University, Spring 2026

Tabs:
  1. Similarity Matrix  — heatmap + scannable sample of the 193×193 matrix
  2. Agglomerative      — bottom-up hierarchical clustering + dendrogram table
  3. K-Means            — partitional clustering with multiple restarts
  4. Comparison         — side-by-side algorithm comparison using Dunn Index

All geographic reference data and Precision/Recall/F1 metrics have been
removed. Evaluation is done exclusively via the internal Dunn Index
(Lecture 10 §6), which requires no external ground truth.
"""

import json
import os

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Remove the old import:
# from src.matrix_builder import build_matrix, load_matrix, WORKING_SET

# Replace with:
from src.matrix_builder import build_matrix, load_matrix
from src.collector import UN_MEMBER_STATES as WORKING_SET
from src.clustering import agglomerative, kmeans
from src.cluster_eval import evaluate, print_evaluation

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Country Clustering — Project 2",
    page_icon="🌐",
    layout="wide",
)

st.title("Wikipedia Infobox Country Clustering")
st.caption("COE 543/743 · Project 2 · Lebanese American University · Spring 2026")

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

MATRIX_PATH = os.path.join("data", "un_similarity_matrix_193.json")

# Number of countries shown in the scannable matrix sample
DEFAULT_SAMPLE_SIZE = 20


def _load_matrix_from_disk() -> tuple[dict | None, list[str] | None]:
    """Load the cached matrix from disk; return (None, None) if not found."""
    if not os.path.exists(MATRIX_PATH):
        return None, None
    with open(MATRIX_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["matrix"], data["countries"]


def _build_heatmap(
    matrix: dict,
    countries: list[str],
    title: str = "Pairwise Similarity Matrix",
) -> go.Figure:
    """Build a Plotly heatmap from the similarity matrix."""
    values = [[matrix[c1][c2] for c2 in countries] for c1 in countries]
    fig = px.imshow(
        values,
        x=countries,
        y=countries,
        color_continuous_scale="RdYlGn",
        zmin=0,
        zmax=1,
        aspect="auto",
    )
    fig.update_layout(
        title=title,
        xaxis_tickangle=-45,
        height=max(400, min(900, len(countries) * 14)),
        coloraxis_colorbar_title="Similarity",
    )
    # Hide per-cell text for large matrices (unreadable anyway)
    if len(countries) <= 30:
        fig.update_traces(text=[[f"{v:.2f}" for v in row] for row in values],
                          texttemplate="%{text}", textfont_size=7)
    return fig


def _cluster_table(
    clusters: list[list[str]],
    medoids: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build a tidy DataFrame of cluster assignments.
    Medoid column bug fix: compare by value, not identity.
    """
    medoid_set = set(medoids) if medoids else set()
    rows = []
    for i, cluster in enumerate(clusters, 1):
        for country in sorted(cluster):
            rows.append({
                "Cluster": i,
                "Country": country,
                "Medoid": "★" if country in medoid_set else "",
            })
    return pd.DataFrame(rows)


def _dunn_card(eval_result, label: str = "") -> None:
    """Render a Dunn Index metric card in the Streamlit UI."""
    prefix = f"{label} — " if label else ""
    col1, col2, col3 = st.columns(3)
    col1.metric(
        f"{prefix}Dunn Index",
        f"{eval_result.dunn_index:.4f}",
        help="Higher is better. Ratio of min inter-cluster distance to max intra-cluster diameter.",
    )
    col2.metric("Min inter-cluster dist", f"{eval_result.min_inter_dist:.4f}")
    col3.metric("Max intra-cluster diam", f"{eval_result.max_intra_diam:.4f}")
    st.caption(
        f"{eval_result.n_clusters} clusters · {eval_result.n_objects} countries evaluated"
    )


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Similarity Matrix",
    "🌿 Agglomerative Clustering",
    "🎯 K-Means Clustering",
    "⚖️ Comparison & Evaluation",
])


# ───────────────────────────────────────────────────────────────────────────
# Tab 1 — Similarity Matrix
# ───────────────────────────────────────────────────────────────────────────

with tab1:
    st.header("Similarity Matrix")
    st.caption(
        "Build or refresh the 193×193 UN member state similarity matrix. "
        "Use the sample viewer to inspect any subset without loading the full heatmap."
    )

    col_info, col_btn = st.columns([4, 1])
    with col_btn:
        if st.button("Build / Refresh Matrix", type="primary"):
            with st.spinner(
                "Computing similarity matrix for 193 countries — "
                "this will take several minutes..."
            ):
                try:
                    build_matrix(WORKING_SET, overwrite=True)
                    st.success("Matrix built and cached successfully.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Error building matrix: {exc}")

    matrix, countries = _load_matrix_from_disk()

    if matrix is None:
        st.info(
            "No cached matrix found. Click **Build / Refresh Matrix** to compute it. "
            f"This will process {len(WORKING_SET)} UN member countries."
        )
    else:
        n = len(countries)
        n_pairs = n * (n - 1) // 2
        st.success(f"Matrix loaded: **{n} countries** · **{n_pairs:,} pairs**")

        # ── Scannable sample ──────────────────────────────────────────────
        st.subheader("Matrix Sample")
        st.caption(
            "Select which countries to preview. "
            "The full matrix is too large to display all at once — "
            "use this to inspect specific subsets."
        )

        sample_size = st.slider(
            "Sample size (countries)",
            min_value=5,
            max_value=min(50, n),
            value=min(DEFAULT_SAMPLE_SIZE, n),
            step=5,
        )

        selected_sample = st.multiselect(
            "Countries to include in sample (leave empty to use first N alphabetically)",
            options=sorted(countries),
            default=[],
        )

        if not selected_sample:
            # Default: first N countries in alphabetical order
            display_countries = sorted(countries)[:sample_size]
        else:
            display_countries = selected_sample[:sample_size]

        st.plotly_chart(
            _build_heatmap(
                matrix,
                display_countries,
                title=f"Similarity Sample — {len(display_countries)} countries",
            ),
            use_container_width=True,
        )

        # ── Raw scores table ──────────────────────────────────────────────
        with st.expander(f"Raw scores for sample ({len(display_countries)}×{len(display_countries)})"):
            sample_df = pd.DataFrame(
                [[matrix[c1][c2] for c2 in display_countries] for c1 in display_countries],
                index=display_countries,
                columns=display_countries,
            )
            st.dataframe(sample_df.style.format("{:.4f}").background_gradient(
                cmap="RdYlGn", vmin=0, vmax=1
            ))

        # ── Full matrix download ──────────────────────────────────────────
        with st.expander("Download full matrix as CSV"):
            full_df = pd.DataFrame(
                [[matrix[c1][c2] for c2 in countries] for c1 in countries],
                index=countries,
                columns=countries,
            )
            st.download_button(
                label="Download 193×193 matrix (CSV)",
                data=full_df.to_csv(),
                file_name="similarity_matrix_193.csv",
                mime="text/csv",
            )


# ───────────────────────────────────────────────────────────────────────────
# Tab 2 — Agglomerative Hierarchical Clustering
# ───────────────────────────────────────────────────────────────────────────

with tab2:
    st.header("Agglomerative Hierarchical Clustering")
    st.caption(
        "Bottom-up clustering using average-link similarity (Lecture 10 §5.2). "
        "Builds a full dendrogram; cut at k clusters or a similarity threshold."
    )

    matrix, countries = _load_matrix_from_disk()

    if matrix is None:
        st.info("Build the similarity matrix first (Matrix tab).")
    else:
        col1, col2 = st.columns(2)
        with col1:
            cut_method = st.radio(
                "Cut method",
                ["Number of clusters (k)", "Similarity threshold"],
            )
        with col2:
            if cut_method == "Number of clusters (k)":
                agg_k = st.slider(
                    "k", min_value=2, max_value=min(20, len(countries)), value=7
                )
                agg_threshold = None
            else:
                agg_threshold = st.slider(
                    "Similarity threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.65,
                    step=0.01,
                )
                agg_k = None

        if st.button("Run Agglomerative Clustering", type="primary"):
            with st.spinner("Clustering — this may take a moment for 193 countries..."):
                try:
                    kwargs = (
                        {"k": agg_k}
                        if agg_k is not None
                        else {"threshold": agg_threshold}
                    )
                    agg_result = agglomerative(matrix, countries, **kwargs)
                    st.session_state["agg_result"] = agg_result
                    st.session_state["agg_matrix"] = matrix
                    st.session_state["agg_countries"] = countries
                except Exception as exc:
                    st.error(f"Clustering error: {exc}")

        if "agg_result" in st.session_state:
            result = st.session_state["agg_result"]
            agg_matrix = st.session_state["agg_matrix"]
            agg_countries = st.session_state["agg_countries"]
            n_clusters = len(result.flat_clusters)

            st.subheader(f"Results — {n_clusters} clusters")

            col_left, col_right = st.columns(2)
            with col_left:
                st.dataframe(
                    _cluster_table(result.flat_clusters),
                    use_container_width=True,
                    hide_index=True,
                )
            with col_right:
                sizes = [len(c) for c in result.flat_clusters]
                labels = [f"Cluster {i+1}" for i in range(n_clusters)]
                fig = px.pie(
                    values=sizes,
                    names=labels,
                    title="Cluster size distribution",
                )
                st.plotly_chart(fig, use_container_width=True)

            # Dendrogram merge table
            st.subheader("Dendrogram (merge sequence)")
            st.caption(
                "Merges are shown in order. Earlier rows = higher similarity = "
                "more natural groupings. Later rows = forced merges at lower similarity."
            )
            merge_data = [
                {
                    "Step": i + 1,
                    "Group A": ", ".join(sorted(s.cluster_a)),
                    "Group B": ", ".join(sorted(s.cluster_b)),
                    "Avg-Link Similarity": s.similarity,
                    "Merged size": len(s.merged),
                }
                for i, s in enumerate(result.dendrogram.merges)
            ]
            st.dataframe(
                pd.DataFrame(merge_data),
                use_container_width=True,
                hide_index=True,
            )

            # Internal evaluation — Dunn Index
            st.subheader("Internal Evaluation — Dunn Index")
            st.caption(
                "Dunn Index = min inter-cluster distance / max intra-cluster diameter. "
                "Higher means clusters are well-separated and compact (Lecture 10 §6)."
            )
            try:
                eval_result = evaluate(result.flat_clusters, agg_matrix)
                _dunn_card(eval_result)
            except ValueError as exc:
                st.warning(str(exc))


# ───────────────────────────────────────────────────────────────────────────
# Tab 3 — K-Means Clustering
# ───────────────────────────────────────────────────────────────────────────

with tab3:
    st.header("K-Means Clustering")
    st.caption(
        "Partitional clustering with medoid-based centroids (Lecture 10 §5.1). "
        "Multiple restarts — best result by intra-cluster similarity is kept."
    )

    matrix, countries = _load_matrix_from_disk()

    if matrix is None:
        st.info("Build the similarity matrix first (Matrix tab).")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            km_k = st.slider(
                "Number of clusters (k)",
                min_value=2,
                max_value=min(20, len(countries)),
                value=7,
            )
        with col2:
            km_runs = st.slider("Random restarts", min_value=1, max_value=20, value=5)
        with col3:
            km_max_iter = st.slider(
                "Max iterations per run", min_value=10, max_value=500, value=100
            )

        if st.button("Run K-Means Clustering", type="primary"):
            with st.spinner(
                f"Running K-Means with k={km_k}, {km_runs} restarts..."
            ):
                try:
                    km_result = kmeans(
                        matrix,
                        countries,
                        k=km_k,
                        max_iterations=km_max_iter,
                        n_runs=km_runs,
                    )
                    st.session_state["km_result"] = km_result
                    st.session_state["km_matrix"] = matrix
                    st.session_state["km_countries"] = countries
                except Exception as exc:
                    st.error(f"Clustering error: {exc}")

        if "km_result" in st.session_state:
            result = st.session_state["km_result"]
            km_matrix = st.session_state["km_matrix"]

            st.subheader(
                f"Results — {result.k} clusters · "
                f"converged in {result.iterations_used} iterations"
            )
            st.caption(
                f"Total intra-cluster similarity: **{result.intra_cluster_similarity:.4f}** "
                "(sum of sim(country, medoid) across all clusters)"
            )

            col_left, col_right = st.columns(2)
            with col_left:
                st.dataframe(
                    _cluster_table(result.clusters, result.medoids),
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption("★ = medoid (most representative country in cluster)")
            with col_right:
                sizes = [len(c) for c in result.clusters]
                labels = [
                    f"Cluster {i+1} [{result.medoids[i]}]"
                    for i in range(len(result.clusters))
                ]
                fig = px.pie(
                    values=sizes, names=labels, title="Cluster size distribution"
                )
                st.plotly_chart(fig, use_container_width=True)

            # Internal evaluation — Dunn Index
            st.subheader("Internal Evaluation — Dunn Index")
            st.caption(
                "Dunn Index = min inter-cluster distance / max intra-cluster diameter. "
                "Higher means clusters are well-separated and compact (Lecture 10 §6)."
            )
            try:
                eval_result = evaluate(result.clusters, km_matrix)
                _dunn_card(eval_result)
            except ValueError as exc:
                st.warning(str(exc))


# ───────────────────────────────────────────────────────────────────────────
# Tab 4 — Algorithm Comparison & Evaluation
# ───────────────────────────────────────────────────────────────────────────

with tab4:
    st.header("Algorithm Comparison & Evaluation")
    st.caption(
        "Run both algorithms with the same k and compare cluster assignments "
        "and Dunn Index scores side by side."
    )

    matrix, countries = _load_matrix_from_disk()

    if matrix is None:
        st.info("Build the similarity matrix first (Matrix tab).")
    else:
        col1, col2 = st.columns(2)
        with col1:
            compare_k = st.slider(
                "Number of clusters (k)",
                min_value=2,
                max_value=min(20, len(countries)),
                value=7,
                key="compare_k",
            )
        with col2:
            compare_runs = st.slider(
                "K-Means restarts",
                min_value=1,
                max_value=20,
                value=5,
                key="compare_runs",
            )

        if st.button("Run Both Algorithms", type="primary"):
            with st.spinner("Running Agglomerative..."):
                try:
                    agg = agglomerative(matrix, countries, k=compare_k)
                except Exception as exc:
                    st.error(f"Agglomerative error: {exc}")
                    st.stop()

            with st.spinner(f"Running K-Means ({compare_runs} restarts)..."):
                try:
                    km = kmeans(matrix, countries, k=compare_k, n_runs=compare_runs)
                except Exception as exc:
                    st.error(f"K-Means error: {exc}")
                    st.stop()

            st.session_state["compare_agg"] = agg
            st.session_state["compare_km"] = km
            st.session_state["compare_matrix"] = matrix

        if "compare_agg" in st.session_state:
            agg = st.session_state["compare_agg"]
            km = st.session_state["compare_km"]
            cmp_matrix = st.session_state["compare_matrix"]

            # ── Cluster assignments side by side ──────────────────────────
            st.subheader("Cluster Assignments")
            col_agg, col_km = st.columns(2)
            with col_agg:
                st.markdown("**Agglomerative**")
                st.dataframe(
                    _cluster_table(agg.flat_clusters),
                    use_container_width=True,
                    hide_index=True,
                )
            with col_km:
                st.markdown("**K-Means**")
                st.dataframe(
                    _cluster_table(km.clusters, km.medoids),
                    use_container_width=True,
                    hide_index=True,
                )

            # ── Dunn Index comparison ─────────────────────────────────────
            st.subheader("Internal Evaluation — Dunn Index Comparison")
            st.caption(
                "The Dunn Index is an internal measure (Lecture 10 §6): no external "
                "reference data needed. Higher = more compact, better-separated clusters."
            )

            try:
                agg_eval = evaluate(agg.flat_clusters, cmp_matrix)
                km_eval = evaluate(km.clusters, cmp_matrix)
            except ValueError as exc:
                st.warning(str(exc))
                st.stop()

            # Metric summary table
            metrics_df = pd.DataFrame({
                "Metric": [
                    "Dunn Index",
                    "Min inter-cluster dist",
                    "Max intra-cluster diam",
                    "Clusters",
                    "Countries",
                ],
                "Agglomerative": [
                    agg_eval.dunn_index,
                    agg_eval.min_inter_dist,
                    agg_eval.max_intra_diam,
                    agg_eval.n_clusters,
                    agg_eval.n_objects,
                ],
                "K-Means": [
                    km_eval.dunn_index,
                    km_eval.min_inter_dist,
                    km_eval.max_intra_diam,
                    km_eval.n_clusters,
                    km_eval.n_objects,
                ],
            })
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)

            # Bar chart — Dunn Index only (the meaningful comparison)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name="Agglomerative",
                x=["Dunn Index", "Min inter dist", "Max intra diam"],
                y=[agg_eval.dunn_index, agg_eval.min_inter_dist, agg_eval.max_intra_diam],
                marker_color="#2196F3",
            ))
            fig.add_trace(go.Bar(
                name="K-Means",
                x=["Dunn Index", "Min inter dist", "Max intra diam"],
                y=[km_eval.dunn_index, km_eval.min_inter_dist, km_eval.max_intra_diam],
                marker_color="#4CAF50",
            ))
            fig.update_layout(
                barmode="group",
                title=f"Dunn Index Comparison (k={compare_k})",
                height=400,
                yaxis_title="Value",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Cluster size distribution comparison
            st.subheader("Cluster Size Distribution")
            col_sz_agg, col_sz_km = st.columns(2)
            with col_sz_agg:
                agg_sizes = sorted([len(c) for c in agg.flat_clusters], reverse=True)
                fig_agg = px.bar(
                    x=[f"C{i+1}" for i in range(len(agg_sizes))],
                    y=agg_sizes,
                    title=f"Agglomerative — sizes",
                    labels={"x": "Cluster", "y": "# Countries"},
                    color_discrete_sequence=["#2196F3"],
                )
                st.plotly_chart(fig_agg, use_container_width=True)
            with col_sz_km:
                km_sizes = sorted([len(c) for c in km.clusters], reverse=True)
                fig_km = px.bar(
                    x=[f"C{i+1}" for i in range(len(km_sizes))],
                    y=km_sizes,
                    title=f"K-Means — sizes",
                    labels={"x": "Cluster", "y": "# Countries"},
                    color_discrete_sequence=["#4CAF50"],
                )
                st.plotly_chart(fig_km, use_container_width=True)
