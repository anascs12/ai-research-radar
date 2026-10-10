"""Semantic search over the AI Research Radar corpus.

Run with:  uv run streamlit run app/streamlit_app.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from radar import nlp

DATA = Path(__file__).resolve().parents[1] / "data" / "processed"

st.set_page_config(page_title="AI Research Radar · Semantic Search", page_icon="🛰️", layout="wide")


@st.cache_resource
def model():
    return nlp.load_model()


@st.cache_data
def corpus():
    papers = pd.read_parquet(DATA / "papers_with_topics.parquet")
    vectors = np.load(DATA / "embeddings.npy")
    return papers, vectors


papers, vectors = corpus()
st.title("🛰️ AI Research Radar")
st.caption(f"Semantic search over {len(papers):,} AI papers from arXiv "
           f"({papers['month'].min()} to {papers['month'].max()}), embedded with {nlp.MODEL_NAME.split('/')[-1]}.")

with st.sidebar:
    st.header("Filters")
    topics = sorted(papers["topic_name"].unique())
    chosen = st.multiselect("Topics", topics, placeholder="All topics")
    months = sorted(papers["month"].unique())
    start, end = st.select_slider("Months", options=months, value=(months[0], months[-1]))
    top_k = st.slider("Results", 5, 30, 10)
    compare = st.toggle("Compare with keyword search", value=False)

examples = ["making language models admit when they don't know something",
            "robots learning tasks by watching videos",
            "cheaper ways to fine-tune big models on a single GPU"]
query = st.text_input("Describe what you're looking for in plain English", placeholder=examples[0])
st.caption("Try: " + " · ".join(f"*{e}*" for e in examples))

if query:
    mask = papers["month"].between(start, end)
    if chosen:
        mask &= papers["topic_name"].isin(chosen)
    subset, subset_vectors = papers[mask.to_numpy()], vectors[mask.to_numpy()]
    if subset.empty:
        st.warning("No papers match the filters.")
    else:
        results = nlp.semantic_search(query, subset_vectors, subset, model=model(), top_k=top_k)
        left, right = st.columns([3, 2]) if compare else (st.container(), None)
        with left:
            st.subheader("Semantic search")
            for _, row in results.iterrows():
                st.markdown(f"**[{row['title']}](https://arxiv.org/abs/{row['arxiv_id']})**  \n"
                            f"`{row['topic_name']}` · {row['month']} · similarity {row['similarity']:.2f}")
                with st.expander("Abstract"):
                    st.write(row["abstract"])
        if compare:
            with right:
                st.subheader("Keyword search")
                keyword = nlp.keyword_search(query, subset, top_k=top_k)
                st.caption(f"{len(keyword)} papers contain every query word.")
                for _, row in keyword.iterrows():
                    st.markdown(f"- [{row['title']}](https://arxiv.org/abs/{row['arxiv_id']})")

        share = results["topic_name"].value_counts().rename_axis("topic").reset_index(name="results")
        st.plotly_chart(px.bar(share, x="results", y="topic", orientation="h", title="Topics among the results",
                               template="plotly_white").update_layout(height=300, yaxis_title=""),
                        width="stretch")
