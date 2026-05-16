import json
import os

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

from src.matrix_builder import build_matrix, load_matrix, WORKING_SET
from src.clustering import agglomerative, kmeans
from src.cluster_eval import evaluate, print_evaluation, REFERENCE_CLUSTERS


st.set_page_config(
    page_title="Country Clustering",
    page_icon="🌍",
    layout="wide",
)

st.title("Wikipedia Infobox Country Clustering")
st.caption("COE 543/743 — Project 2 | Lebanese American University, Spring 2026")


def _get_matrix():
    matrix_path = os.path.join("data", "similarity_matrix.json")
    if os.path.exists(matrix_path):
        with open(matrix_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["matrix"], data["countries"]
    return None, None


def _build_heatmap(matrix: dict, countries: list[str]):
    values = [[matrix[c1][c2] for c2 in countries] for c1 in countries]
    fig = px.imshow(
        values,
        x=countries,
        y=countries,
        color_continuous_scale="RdYlGn",
        zmin=0,
        zmax=1,
        text_auto=".2f",
        aspect="auto",
    )
    fig.update_layout(
        title="Pairwise Similarity Matrix",
        xaxis_tickangle=-45,
        height=700,
        coloraxis_colorbar_title="Similarity",
    )
    fig.update_traces(textfont_size=8)
    return fig


def _build_dendrogram(agg_result, countries: list[str]):
    labels = countries
    linkage_matrix = []
    cluster_map = {c: i for i, c in enumerate(countries)}
    next_id = len(countries)

    for step in agg_result.dendrogram.merges:
        a_rep = step.cluster_a[0]
        b_rep = step.cluster_b[0]
        a_id = cluster_map.get(a_rep, next_id - 1)
        b_id = cluster_map.get(b_rep, next_id - 1)
        dist = round(1 - step.similarity, 4)
        size = len(step.merged)
        linkage_matrix.append([a_id, b_id, dist, size])
        for m in step.merged:
            cluster_map[m] = next_id
        next_id += 1

    if not linkage_matrix:
        return None

    fig = ff.create_dendrogram(
        [[i] for i in range(len(countries))],
        labels=labels,
        linkagefun=lambda x: linkage_matrix,
        orientation="bottom",
    )
    fig.update_layout(
        title="Agglomerative Clustering Dendrogram",
        height=500,
        xaxis_tickangle=-45,
    )
    return fig


def _cluster_table(clusters: list[list[str]], medoids: list[str] | None = None) -> pd.DataFrame:
    rows = []
    for i, cluster in enumerate(clusters, 1):
        for country in sorted(cluster):
            is_medoid = medoids is not None and country in medoids
            rows.append({
                "Cluster": i,
                "Country": country,
                "Medoid": "★" if is_medoid else "",
            })
    return pd.DataFrame(rows)


tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Similarity Matrix",
    "🌿 Agglomerative Clustering",
    "🎯 K-Means Clustering",
    "⚖️ Comparison & Evaluation",
])


with tab1:
    st.header("Similarity Matrix")

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("Build / Refresh Matrix", type="primary"):
            with st.spinner("Computing similarity matrix — this may take a few minutes..."):
                try:
                    matrix, countries = build_matrix(WORKING_SET, overwrite=True), None
                    st.success("Matrix built successfully.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error building matrix: {e}")

    matrix, countries = _get_matrix()

    if matrix is None:
        st.info("No matrix found. Click 'Build / Refresh Matrix' to compute it.")
    else:
        st.caption(f"{len(countries)} countries · {len(countries) * (len(countries) - 1) // 2} pairs")
        st.plotly_chart(_build_heatmap(matrix, countries), use_container_width=True)

        with st.expander("Raw similarity scores"):
            df = pd.DataFrame(
                [[matrix[c1][c2] for c2 in countries] for c1 in countries],
                index=countries,
                columns=countries,
            )
            st.dataframe(df.style.format("{:.3f}"))


with tab2:
    st.header("Agglomerative Hierarchical Clustering")
    st.caption("Bottom-up clustering using average-link similarity. No need to specify k upfront.")

    matrix, countries = _get_matrix()

    if matrix is None:
        st.info("Build the similarity matrix first (Matrix tab).")
    else:
        col1, col2 = st.columns(2)
        with col1:
            cut_method = st.radio("Cut method", ["Number of clusters (k)", "Similarity threshold"])
        with col2:
            if cut_method == "Number of clusters (k)":
                agg_k = st.slider("k", min_value=2, max_value=min(10, len(countries)), value=5)
                agg_threshold = None
            else:
                agg_threshold = st.slider("Similarity threshold", min_value=0.0, max_value=1.0, value=0.65, step=0.01)
                agg_k = None

        if st.button("Run Agglomerative Clustering", type="primary"):
            with st.spinner("Clustering..."):
                try:
                    kwargs = {"k": agg_k} if agg_k is not None else {"threshold": agg_threshold}
                    agg_result = agglomerative(matrix, countries, **kwargs)
                    st.session_state["agg_result"] = agg_result
                    st.session_state["agg_countries"] = countries
                except Exception as e:
                    st.error(f"Clustering error: {e}")

        if "agg_result" in st.session_state:
            result = st.session_state["agg_result"]
            agg_countries = st.session_state["agg_countries"]

            st.subheader(f"Results — {len(result.flat_clusters)} clusters")

            col_left, col_right = st.columns(2)
            with col_left:
                st.dataframe(_cluster_table(result.flat_clusters), use_container_width=True, hide_index=True)
            with col_right:
                sizes = [len(c) for c in result.flat_clusters]
                labels = [f"Cluster {i+1}" for i in range(len(result.flat_clusters))]
                fig = px.pie(values=sizes, names=labels, title="Cluster size distribution")
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("Dendrogram")
            st.caption("Merges shown in order. Later merges happen at lower similarity.")
            merge_data = [
                {
                    "Step": i + 1,
                    "Group A": ", ".join(sorted(s.cluster_a)),
                    "Group B": ", ".join(sorted(s.cluster_b)),
                    "Similarity": s.similarity,
                    "Merged size": len(s.merged),
                }
                for i, s in enumerate(result.dendrogram.merges)
            ]
            st.dataframe(pd.DataFrame(merge_data), use_container_width=True, hide_index=True)

            eval_result = evaluate(result.flat_clusters, agg_countries)
            st.subheader("Evaluation against geographic reference")
            m1, m2, m3 = st.columns(3)
            m1.metric("Precision", f"{eval_result.precision:.4f}")
            m2.metric("Recall", f"{eval_result.recall:.4f}")
            m3.metric("F-value", f"{eval_result.f_value:.4f}")
            st.caption(
                f"{eval_result.correctly_grouped} correctly grouped pairs out of "
                f"{eval_result.total_grouped} grouped / {eval_result.total_reference_pairs} reference"
            )


with tab3:
    st.header("K-Means Clustering")
    st.caption("Partitional clustering using medoids. Specify k, then run multiple restarts.")

    matrix, countries = _get_matrix()

    if matrix is None:
        st.info("Build the similarity matrix first (Matrix tab).")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            km_k = st.slider("Number of clusters (k)", min_value=2, max_value=min(10, len(countries)), value=5)
        with col2:
            km_runs = st.slider("Random restarts", min_value=1, max_value=20, value=5)
        with col3:
            km_max_iter = st.slider("Max iterations", min_value=10, max_value=500, value=100)

        if st.button("Run K-Means Clustering", type="primary"):
            with st.spinner(f"Running k-means with {km_runs} restarts..."):
                try:
                    km_result = kmeans(matrix, countries, k=km_k, max_iterations=km_max_iter, n_runs=km_runs)
                    st.session_state["km_result"] = km_result
                    st.session_state["km_countries"] = countries
                except Exception as e:
                    st.error(f"Clustering error: {e}")

        if "km_result" in st.session_state:
            result = st.session_state["km_result"]
            km_countries = st.session_state["km_countries"]

            st.subheader(f"Results — {result.k} clusters · converged in {result.iterations_used} iterations")
            st.caption(f"Total intra-cluster similarity: {result.intra_cluster_similarity:.4f}")

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
                labels = [f"Cluster {i+1} [{result.medoids[i]}]" for i in range(len(result.clusters))]
                fig = px.pie(values=sizes, names=labels, title="Cluster size distribution")
                st.plotly_chart(fig, use_container_width=True)

            eval_result = evaluate(result.clusters, km_countries)
            st.subheader("Evaluation against geographic reference")
            m1, m2, m3 = st.columns(3)
            m1.metric("Precision", f"{eval_result.precision:.4f}")
            m2.metric("Recall", f"{eval_result.recall:.4f}")
            m3.metric("F-value", f"{eval_result.f_value:.4f}")
            st.caption(
                f"{eval_result.correctly_grouped} correctly grouped pairs out of "
                f"{eval_result.total_grouped} grouped / {eval_result.total_reference_pairs} reference"
            )


with tab4:
    st.header("Algorithm Comparison & Evaluation")
    st.caption("Run both algorithms with the same k to compare results side by side.")

    matrix, countries = _get_matrix()

    if matrix is None:
        st.info("Build the similarity matrix first (Matrix tab).")
    else:
        compare_k = st.slider("k for comparison", min_value=2, max_value=min(10, len(countries)), value=5, key="compare_k")
        compare_runs = st.slider("K-means restarts", min_value=1, max_value=20, value=5, key="compare_runs")

        if st.button("Run Both Algorithms", type="primary"):
            with st.spinner("Running agglomerative..."):
                agg = agglomerative(matrix, countries, k=compare_k)
            with st.spinner("Running k-means..."):
                km = kmeans(matrix, countries, k=compare_k, n_runs=compare_runs)

            agg_eval = evaluate(agg.flat_clusters, countries)
            km_eval = evaluate(km.clusters, countries)

            st.subheader("Cluster Assignments")
            col_agg, col_km = st.columns(2)
            with col_agg:
                st.markdown("**Agglomerative**")
                st.dataframe(_cluster_table(agg.flat_clusters), use_container_width=True, hide_index=True)
            with col_km:
                st.markdown("**K-Means**")
                st.dataframe(_cluster_table(km.clusters, km.medoids), use_container_width=True, hide_index=True)

            st.subheader("Evaluation Metrics")
            metrics_df = pd.DataFrame({
                "Metric": ["Precision", "Recall", "F-value",
                           "Correctly grouped pairs", "Total grouped pairs"],
                "Agglomerative": [
                    agg_eval.precision, agg_eval.recall, agg_eval.f_value,
                    agg_eval.correctly_grouped, agg_eval.total_grouped,
                ],
                "K-Means": [
                    km_eval.precision, km_eval.recall, km_eval.f_value,
                    km_eval.correctly_grouped, km_eval.total_grouped,
                ],
            })
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)

            fig = go.Figure()
            fig.add_trace(go.Bar(
                name="Agglomerative",
                x=["Precision", "Recall", "F-value"],
                y=[agg_eval.precision, agg_eval.recall, agg_eval.f_value],
            ))
            fig.add_trace(go.Bar(
                name="K-Means",
                x=["Precision", "Recall", "F-value"],
                y=[km_eval.precision, km_eval.recall, km_eval.f_value],
            ))
            fig.update_layout(
                barmode="group",
                title=f"Precision / Recall / F-value Comparison (k={compare_k})",
                yaxis_range=[0, 1],
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Reference Clustering")
            st.caption("Geographic groups used as ground truth for evaluation.")
            ref_data = []
            for i, group in enumerate(REFERENCE_CLUSTERS, 1):
                present = [c for c in group if c in countries]
                if present:
                    for c in sorted(present):
                        ref_data.append({"Group": i, "Country": c})
            st.dataframe(pd.DataFrame(ref_data), use_container_width=True, hide_index=True)