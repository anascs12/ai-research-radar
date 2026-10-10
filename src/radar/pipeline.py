"""Run the analysis: embeddings -> topics -> 2D map -> trend tables."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from . import nlp

ROOT = Path(".")
PROCESSED = ROOT / "data" / "processed"
N_TOPICS = 25  # silhouette is nearly flat for k=15..35; 25 kept every topic coherent (see notebook)


def topic_names(keywords: dict[int, list[str]]) -> dict[int, str]:
    """Human-written names from data/topic_names.json, falling back to the top keywords.

    Topic names were written by reading each topic's c-TF-IDF keywords and a sample
    of its paper titles; the mapping is checked against the keywords at load time.
    """
    path = ROOT / "data" / "topic_names.json"
    named = json.loads(path.read_text()) if path.exists() else {}
    names = {}
    for topic, words in keywords.items():
        entry = named.get(str(topic))
        if entry and entry.get("keywords", [])[:3] == words[:3]:
            names[topic] = entry["name"]
        else:
            names[topic] = " / ".join(words[:3])
    return names


def run(k: int = N_TOPICS) -> None:
    papers = pd.read_parquet(PROCESSED / "papers.parquet")
    print(f"{len(papers):,} papers")
    vectors = nlp.cached_embeddings(papers, PROCESSED / "embeddings.npy")
    print("embeddings", vectors.shape)

    labels = nlp.cluster(vectors, k)
    keywords = nlp.topic_keywords(nlp.document_text(papers), labels)
    names = topic_names(keywords)
    papers["topic"] = labels
    papers["topic_name"] = papers["topic"].map(names)

    reduced = PCA(n_components=50, random_state=42).fit_transform(vectors)
    xy = TSNE(n_components=2, perplexity=40, init="pca", random_state=42, metric="cosine").fit_transform(reduced)
    papers["x"], papers["y"] = xy[:, 0], xy[:, 1]

    papers.to_parquet(PROCESSED / "papers_with_topics.parquet", index=False)
    topics = pd.DataFrame({
        "topic": list(keywords), "name": [names[t] for t in keywords],
        "keywords": [", ".join(keywords[t]) for t in keywords],
        "papers": [int((labels == t).sum()) for t in keywords],
    }).sort_values("papers", ascending=False)
    topics.to_csv(PROCESSED / "topics.csv", index=False)
    print(topics.to_string(index=False))


if __name__ == "__main__":
    run()
