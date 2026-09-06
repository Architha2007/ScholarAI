"""Retrieval evaluation metrics dashboard for ScholarAI.

Reads benchmark and ablation study results live from data/evaluation/results/
and renders interactive visual comparisons in Streamlit.
"""

import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

RESULTS_FILES = {
    "faiss_baseline": "faiss_baseline.json",
    "hybrid_benchmark": "hybrid_benchmark.json",
    "reranking_benchmark": "reranking_benchmark.json",
    "ablation_study": "ablation_study.json",
}


def load_single_result(file_path: str) -> Optional[Dict[str, Any]]:
    """Loads a single benchmark JSON result file live from disk.

    Args:
        file_path: Path to the JSON file.

    Returns:
        Dict if file exists and is valid JSON, else None.
    """
    if not os.path.exists(file_path):
        return None

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"Error reading {file_path}: {e}")
        return None


def load_evaluation_results(
    results_dir: str = "data/evaluation/results",
) -> Dict[str, Optional[Dict[str, Any]]]:
    """Loads all saved evaluation result files from disk.

    Args:
        results_dir: Path to directory storing result JSON files.

    Returns:
        Dict mapping result key to loaded JSON dict (or None if missing).
    """
    results: Dict[str, Optional[Dict[str, Any]]] = {}
    for key, filename in RESULTS_FILES.items():
        path = os.path.join(results_dir, filename)
        results[key] = load_single_result(path)
    return results


def get_stage_comparison_data(
    results: Dict[str, Optional[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Formats summary metrics from FAISS, Hybrid, and Reranking results.

    Args:
        results: Dictionary containing loaded JSON result objects.

    Returns:
        List of dictionaries formatted for comparative dataframes/tables.
    """
    stages = [
        ("M6.2 FAISS Baseline (Dense)", "faiss_baseline"),
        ("M6.3 Hybrid Benchmark (FAISS + BM25)", "hybrid_benchmark"),
        ("M6.4 Reranking Benchmark (Cross-Encoder)", "reranking_benchmark"),
    ]

    table_rows = []
    for stage_label, key in stages:
        data = results.get(key)
        if data and "metrics" in data:
            m = data["metrics"]
            k = data.get("k", 5)
            p_val = m.get(f"precision@{k}", m.get("precision@5", 0.0))
            r_val = m.get(f"recall@{k}", m.get("recall@5", 0.0))
            mrr = m.get("mrr", 0.0)
            avg_lat = m.get("avg_latency_ms", 0.0)
            med_lat = m.get("median_latency_ms", 0.0)
        else:
            p_val, r_val, mrr, avg_lat, med_lat = None, None, None, None, None

        table_rows.append({
            "Stage": stage_label,
            "Precision@5": p_val,
            "Recall@5": r_val,
            "MRR": mrr,
            "Avg Latency (ms)": avg_lat,
            "Median Latency (ms)": med_lat,
        })

    return table_rows


def render_metrics_dashboard(
    results_dir: str = "data/evaluation/results",
) -> None:
    """Renders the Streamlit Retrieval Evaluation Dashboard page."""
    st.markdown(
        '<p class="hero-title">📊 Retrieval Evaluation Dashboard</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="hero-subtitle">'
        "Live performance metrics across retrieval stages (FAISS Baseline, "
        "Hybrid RRF, Cross-Encoder Reranking) and controlled ablation studies."
        "</p>",
        unsafe_allow_html=True,
    )

    results = load_evaluation_results(results_dir=results_dir)
    stage_rows = get_stage_comparison_data(results)

    # Top KPI Cards
    hybrid_data = results.get("hybrid_benchmark")
    baseline_data = results.get("faiss_baseline")

    if hybrid_data and baseline_data:
        h_m = hybrid_data.get("metrics", {})
        b_m = baseline_data.get("metrics", {})

        mrr_diff = h_m.get("mrr", 0.0) - b_m.get("mrr", 0.0)
        pct = mrr_diff / max(b_m.get("mrr", 1), 0.001) * 100
        mrr_delta_str = f"{mrr_diff:+.4f} (+{pct:.1f}%)"

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Hybrid Recall@5", f"{h_m.get('recall@5', 0.0):.4f}")
        with col2:
            st.metric(
                "Hybrid MRR", f"{h_m.get('mrr', 0.0):.4f}", delta=mrr_delta_str
            )
        with col3:
            avg_lat = h_m.get("avg_latency_ms", 0.0)
            st.metric("Hybrid Avg Latency", f"{avg_lat:.1f} ms")
        with col4:
            st.metric(
                "Queries Evaluated", f"{hybrid_data.get('num_queries', 0)}"
            )

    st.divider()

    # Stage Comparison Charts & Tables
    st.subheader("📈 Stage-by-Stage Performance Comparison")

    df_stages = pd.DataFrame(stage_rows)
    df_clean = df_stages.dropna(subset=["Precision@5"])

    if not df_clean.empty:
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("#### 🎯 Quality (Precision@5, Recall@5, MRR)")
            chart_df = df_clean.set_index("Stage")[
                ["Precision@5", "Recall@5", "MRR"]
            ]
            st.bar_chart(chart_df)

        with col_right:
            st.markdown("#### ⚡ Latency (ms)")
            lat_df = df_clean.set_index("Stage")[
                ["Avg Latency (ms)", "Median Latency (ms)"]
            ]
            st.bar_chart(lat_df)

        st.markdown("#### 📋 Raw Metrics Table")
        formatted_df = df_stages.style.format(
            {
                "Precision@5": "{:.4f}",
                "Recall@5": "{:.4f}",
                "MRR": "{:.4f}",
                "Avg Latency (ms)": "{:.2f}",
                "Median Latency (ms)": "{:.2f}",
            },
            na_rep="N/A",
        )
        st.dataframe(formatted_df, use_container_width=True)
    else:
        st.info("No benchmark results found.")

    st.divider()

    # Ablation Study Section
    st.subheader("🔬 Retrieval Strategy Ablation Study")
    ablation_data = results.get("ablation_study")

    if ablation_data and "comparison_matrix" in ablation_data:
        matrix = ablation_data["comparison_matrix"]
        ablation_rows = []
        for key, info in matrix.items():
            name = info.get("name", key)
            p_val = info.get("precision@5", info.get("precision@k"))
            r_val = info.get("recall@5", info.get("recall@k"))
            mrr = info.get("mrr")
            avg_lat = info.get("avg_latency_ms")
            med_lat = info.get("median_latency_ms")
            ablation_rows.append({
                "Strategy": name,
                "Precision@5": p_val,
                "Recall@5": r_val,
                "MRR": mrr,
                "Avg Latency (ms)": avg_lat,
                "Median Latency (ms)": med_lat,
            })

        df_ablation = pd.DataFrame(ablation_rows)
        st.dataframe(
            df_ablation.style.format({
                "Precision@5": "{:.4f}",
                "Recall@5": "{:.4f}",
                "MRR": "{:.4f}",
                "Avg Latency (ms)": "{:.2f}",
                "Median Latency (ms)": "{:.2f}",
            }),
            use_container_width=True,
        )

        with st.expander("💡 Ablation Study Insights & Key Takeaways"):
            st.markdown(
                "- **BM25 Keyword Retrieval:** Highest MRR (`0.9500`) on exact"
                " formula queries at sub-millisecond speed (`0.43 ms`).\n"
                "- **FAISS Dense Retrieval:** Provides 100% **Recall@5"
                " (`1.0000`)**, capturing semantic paraphrases.\n"
                "- **Hybrid RRF (FAISS + BM25):** Combines 1.0000 Recall with"
                " 0.9350 MRR at negligible overhead (~493 ms), serving as"
                " optimal production strategy.\n"
                "- **Cross-Encoder Reranking:** Adds heavy latency (~2.9s)"
                " while open-web weights offer lower precision on scientific"
                " formulas."
            )
    else:
        st.warning("Ablation study results not found.")

    st.divider()

    # Per-Query Inspector
    st.subheader("🔍 Per-Query Result Inspector")
    available_b = {
        "FAISS Baseline (M6.2)": results.get("faiss_baseline"),
        "Hybrid Benchmark (M6.3)": results.get("hybrid_benchmark"),
        "Reranking Benchmark (M6.4)": results.get("reranking_benchmark"),
    }
    valid_b = {
        k: v for k, v in available_b.items() if v and "per_query" in v
    }

    if valid_b:
        selected_b_name = st.selectbox(
            "Select Benchmark Run", list(valid_b.keys())
        )
        b_data = valid_b[selected_b_name]
        per_query_list = b_data.get("per_query", [])

        query_labels = [
            f"{q['id']}: {q['question'][:70]}..." for q in per_query_list
        ]
        selected_idx = st.selectbox(
            "Select Query to Inspect",
            range(len(query_labels)),
            format_func=lambda i: query_labels[i],
        )

        if per_query_list:
            q_info = per_query_list[selected_idx]
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric(
                    "Precision@5", f"{q_info.get('precision@5', 0.0):.4f}"
                )
            with c2:
                st.metric("Recall@5", f"{q_info.get('recall@5', 0.0):.4f}")
            with c3:
                st.metric(
                    "Reciprocal Rank",
                    f"{q_info.get('reciprocal_rank', 0.0):.4f}",
                )

            st.write(f"**Question:** {q_info.get('question')}")
            st.write(
                "**Ground-Truth Relevant Chunk(s):** "
                f"`{q_info.get('relevant_chunk_ids')}`"
            )
            st.write(
                "**Top-5 Retrieved Chunk(s):** "
                f"`{q_info.get('retrieved_chunk_ids')}`"
            )
