import os
import sys
import streamlit as st
import pandas as pd

try:
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
except NameError:
    PROJECT_ROOT = os.path.abspath(os.getcwd())

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.engine import load_and_vectorize, evaluate_idea, fetch_google_trends

st.set_page_config(
    page_title="researchLens — Research Hypothesis Evaluator",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------------------
# 1. REACTIVE STATE & ATOMIC CALLBACKS
# ------------------------------------------------------------------------------
if "hypothesis_text" not in st.session_state:
    st.session_state["hypothesis_text"] = ""

def set_preset_topic(topic_str: str):
    """Callback executed before rerun to safely update the text area."""
    st.session_state["hypothesis_text"] = topic_str

def clear_input():
    """Callback to safely clear user input without widget instantiation collisions."""
    st.session_state["hypothesis_text"] = ""

# ------------------------------------------------------------------------------
# 2. CACHED ENGINE INITIALIZATION
# ------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Indexing literature corpus into TF-IDF vector space...")
def get_engine():
    data_path = os.path.join(PROJECT_ROOT, "data", "papers.csv")
    df, vectorizer, tfidf_matrix = load_and_vectorize(data_path)
    return df, vectorizer, tfidf_matrix

df_corpus, vectorizer, tfidf_matrix = get_engine()

# ------------------------------------------------------------------------------
# 3. SIDEBAR CONFIGURATION
# ------------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/search-in-list.png", width=64)
    st.title("researchLens")
    st.caption("AI-Powered Research Triage & Evaluation Engine")
    st.divider()

    st.subheader("Corpus Statistics")
    st.metric("Indexed Publications", f"{len(df_corpus)} Papers")
    st.metric("Vocabulary Features", f"{tfidf_matrix.shape[1]:,} n-grams")
    st.markdown("**Domain:** Bangla NLP & Misinformation")
    st.markdown("**Coverage:** 2020 – 2025")
    st.divider()
    top_k_val = st.slider("Matches to Retrieve", min_value=1, max_value=10, value=5)

# ------------------------------------------------------------------------------
# 4. MAIN DASHBOARD INTERFACE
# ------------------------------------------------------------------------------
st.title("🔬 researchLens: Idea Novelty & Feasibility Evaluator")
st.markdown(
    "Evaluate research hypotheses against an indexed academic corpus. "
    "Novelty is scored via **inverse cosine distance**; Feasibility evaluates dataset and benchmark availability."
)

st.markdown("##### Quick-Load Preset Test Cases:")
col_s1, col_s2, col_s3 = st.columns(3)

with col_s1:
    st.button("📋 Replication (Low Novelty)", use_container_width=True, on_click=set_preset_topic,
              args=("Explainable AI for Bangla fake news detection using LIME and SHAP",))
with col_s2:
    st.button("🔀 Synthesis (Moderate Novelty)", use_container_width=True, on_click=set_preset_topic,
              args=("Multimodal deep learning combining audio transcripts and social graphs for rumor tracking",))
with col_s3:
    st.button("🚀 High Novelty (Under-explored)", use_container_width=True, on_click=set_preset_topic,
              args=("Quantum computing approaches to encrypted satellite communications",))

hypothesis_input = st.text_area(
    "Enter research hypothesis, method, and domain details:",
    key="hypothesis_text",
    height=110,
    placeholder="e.g., Cross-domain stance detection in Bangla news comments to verify disputed political statements..."
)

col_act1, col_act2 = st.columns([4, 1])
with col_act1:
    evaluate_clicked = st.button("Evaluate Hypothesis", type="primary", use_container_width=True)
with col_act2:
    st.button("Clear Input", use_container_width=True, on_click=clear_input)

# ------------------------------------------------------------------------------
# 5. EXECUTION & CONDITIONAL RENDERING
# ------------------------------------------------------------------------------
if evaluate_clicked:
    if not hypothesis_input.strip():
        st.warning("Please enter or select a research hypothesis before evaluating.")
    else:
        with st.spinner("Validating and scoring hypothesis against vector space..."):
            result = evaluate_idea(hypothesis_input, df_corpus, vectorizer, tfidf_matrix, top_k=top_k_val)

        if not result.get("is_valid", True):
            st.error(f"⚠️ **Invalid Hypothesis Input:** {result['error_message']}")
            st.info("💡 **Example of a valid hypothesis:**\n\n*\"Explainable AI for Bangla fake news detection using LIME and SHAP to interpret transformer representations.\"*")
        else:
            top_matches = result["top_matches"]
            
            # An idea has valid matches ONLY if top_matches is not empty and similarity >= 0.15
            has_matches = not top_matches.empty and float(top_matches["similarity"].max()) >= 0.15
            max_sim = float(top_matches["similarity"].max()) if has_matches else 0.0

            st.divider()
            st.subheader("Quantitative Evaluation")

            kpi_1, kpi_2, kpi_3 = st.columns(3)
            novelty_score = 100.0 if not has_matches else result["novelty_score"]
            novelty_status = "100% Unexplored in Corpus" if not has_matches else result["novelty_label"]

            with kpi_1:
                st.metric(
                    label="Novelty Score",
                    value=f"{novelty_score}%",
                    delta="Out-of-Corpus Domain" if not has_matches else ("Unique Potential" if novelty_score >= 35 else "Heavy Overlap"),
                    delta_color="normal" if novelty_score >= 35 else "inverse"
                )
                st.progress(novelty_score / 100.0)
                st.caption(f"Status: **{novelty_status}**")

            with kpi_2:
                feasibility_score = 0.0 if not has_matches else result["feasibility_score"]
                st.metric(
                    label="Feasibility Score",
                    value=f"{feasibility_score}%",
                    delta="No In-Corpus Benchmarks" if not has_matches else "Data & Benchmark Availability",
                    delta_color="off" if not has_matches else "normal"
                )
                st.progress(feasibility_score / 100.0)
                st.caption("No relevant domain benchmarks found in local corpus." if not has_matches else "Measured from explicit datasets and baseline methods in matching papers.")

            with kpi_3:
                st.metric(
                    label="Max Literature Overlap",
                    value=f"{max_sim:.4f}",
                    delta="0.0% Relevant Overlap" if not has_matches else f"{(max_sim * 100):.1f}% Lexical Alignment",
                    delta_color="normal" if not has_matches else ("inverse" if max_sim >= 0.70 else "normal")
                )
                st.progress(min(1.0, max_sim) if has_matches else 0.0)
                st.caption("No relevant papers detected in the index." if not has_matches else "Cosine similarity of closest published paper.")

            # ------------------------------------------------------------------
            # BRANCH A: OFF-TOPIC / NO REAL MATCHES -> SHOW GOOGLE TRENDS ONLY
            # ------------------------------------------------------------------
            if not has_matches:
                st.divider()
                st.warning("⚠️ **Out-of-Domain Topic Detected:** No relevant research papers found in the indexed corpus for this hypothesis.")
                st.markdown(
                    "Because this query has zero substantial alignment with the indexed literature, "
                    "the system evaluates **Google Trends** to analyze global public interest and search volume over the last 5 years."
                )
                
                st.subheader("📈 Real-World Demand: Google Trends Search Volume (5 Years)")
                with st.spinner("Fetching global interest trajectories..."):
                    trends_df = fetch_google_trends(hypothesis_input)
                
                st.line_chart(trends_df, use_container_width=True)
                st.caption(f"Longitudinal search interest tracking demand for the keywords extracted from: *\"{hypothesis_input}\"*")

            # ------------------------------------------------------------------
            # BRANCH B: IN-DOMAIN MATCHES FOUND -> SHOW LITERATURE & PAPERS
            # ------------------------------------------------------------------
            else:
                st.divider()
                st.subheader("📈 Literature Alignment & Corpus Trends")

                graph_col1, graph_col2 = st.columns(2)

                with graph_col1:
                    st.markdown("##### Cosine Similarity by Matched Publication")
                    chart_df = top_matches[["title", "similarity"]].copy()
                    chart_df["short_label"] = [
                        f"#{row['id']} " + (row['title'][:25] + "..." if len(row['title']) > 25 else row['title'])
                        for _, row in top_matches.iterrows()
                    ]
                    st.bar_chart(data=chart_df.set_index("short_label")["similarity"], use_container_width=True)

                with graph_col2:
                    st.markdown("##### Academic Publication Timeline")
                    timeline_data = top_matches["year"].value_counts().sort_index().rename("Paper Count")
                    st.line_chart(timeline_data, use_container_width=True)

                st.divider()
                st.subheader(f"Top {len(top_matches)} Related Literature Matches")

                display_df = top_matches[["id", "year", "similarity", "title", "source_link", "method", "limitations"]].copy()
                display_df.columns = ["ID", "Year", "Cosine Sim", "Title", "Direct Link", "Method Used", "Key Limitations"]
                display_df["Cosine Sim"] = display_df["Cosine Sim"].map(lambda x: f"{x:.4f}")

                st.dataframe(
                    display_df,
                    column_config={
                        "Direct Link": st.column_config.LinkColumn(
                            "Direct Link",
                            help="Click to open verified paper",
                            display_text="Open Paper ↗"
                        )
                    },
                    use_container_width=True,
                    hide_index=True
                )

                st.subheader("Detailed Paper Breakdown")
                for _, row in top_matches.iterrows():
                    with st.expander(f"Paper #{row['id']} ({row['year']}) — {row['title']} [Similarity: {row['similarity']:.4f}]"):
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.markdown(f"**Methodology:** {row['method']}")
                            st.markdown(f"**Direct URL:** [{row['source_link']}]({row['source_link']})")
                        with col_b:
                            st.markdown(f"**Known Limitations:** {row['limitations']}")