"""Embeddings, topic discovery, trend detection and semantic search."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from sklearn.metrics import silhouette_score

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384-d, fast on CPU

# Words that appear in almost every AI abstract and say nothing about the topic.
DOMAIN_STOP_WORDS = {
    "model", "models", "method", "methods", "approach", "approaches", "paper", "propose", "proposed",
    "results", "show", "shows", "based", "using", "use", "used", "new", "novel", "performance",
    "task", "tasks", "data", "existing", "framework", "work", "study", "large", "language", "llms",
    "llm", "learning", "achieve", "achieves", "demonstrate", "experiments", "state", "art",
    "significantly", "however", "address", "challenge", "challenges", "problem", "problems", "set",
    "different", "various", "key", "also", "introduce", "present", "effective", "effectively",
    "improve", "improves", "improved", "code", "available", "github", "com", "https",
}
STOP_WORDS = list(ENGLISH_STOP_WORDS | DOMAIN_STOP_WORDS)


def document_text(papers: pd.DataFrame) -> list[str]:
    return (papers["title"] + ". " + papers["abstract"]).tolist()


def load_model():
    from sentence_transformers import SentenceTransformer  # heavy import, only when needed

    return SentenceTransformer(MODEL_NAME, device="cpu")


def embed(texts: list[str], model=None, batch_size: int = 64) -> np.ndarray:
    """Unit-normalised embeddings, so a dot product equals cosine similarity."""
    model = model or load_model()
    vectors = model.encode(texts, batch_size=batch_size, show_progress_bar=False,
                           normalize_embeddings=True, convert_to_numpy=True)
    return vectors.astype(np.float32)


def cached_embeddings(papers: pd.DataFrame, path: Path) -> np.ndarray:
    """Embed once and reuse; the cache is invalidated if the paper list changes."""
    ids_path = path.with_suffix(".ids.txt")
    ids = "\n".join(papers["arxiv_id"])
    if path.exists() and ids_path.exists() and ids_path.read_text() == ids:
        return np.load(path)
    vectors = embed(document_text(papers))
    np.save(path, vectors)
    ids_path.write_text(ids)
    return vectors


def choose_k(vectors: np.ndarray, candidates=(15, 20, 25, 30, 35), sample: int = 4000, seed: int = 42) -> pd.DataFrame:
    """Silhouette score for each candidate k, computed on a random sample for speed."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(vectors), size=min(sample, len(vectors)), replace=False)
    rows = []
    for k in candidates:
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(vectors)
        rows.append({"k": k, "silhouette": silhouette_score(vectors[idx], labels[idx], metric="cosine")})
    return pd.DataFrame(rows)


def cluster(vectors: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    # On unit vectors, Euclidean k-means is equivalent to clustering by cosine similarity.
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(vectors)


def topic_keywords(texts: list[str], labels: np.ndarray, top_n: int = 8, min_count: int = 5) -> dict[int, list[str]]:
    """Class-based TF-IDF (the BERTopic idea): words frequent in one topic but rare across topics.

    Rare terms are dropped by total count across the corpus. (``min_df`` would be
    wrong here: each topic is one joined document, so min_df=5 would require a term
    to appear in five topics and discard exactly the distinctive ones.)
    """
    frame = pd.DataFrame({"text": texts, "topic": labels})
    joined = frame.groupby("topic")["text"].apply(" ".join)
    vectorizer = CountVectorizer(stop_words=STOP_WORDS, ngram_range=(1, 2),
                                 token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]+\b")
    counts = vectorizer.fit_transform(joined).toarray().astype(float)
    keep = counts.sum(axis=0) >= min_count
    counts = counts[:, keep]
    tf = counts / counts.sum(axis=1, keepdims=True)
    avg_words = counts.sum() / len(joined)
    idf = np.log(1 + avg_words / counts.sum(axis=0))
    scores = tf * idf
    terms = vectorizer.get_feature_names_out()[keep]
    return {topic: [terms[i] for i in scores[row].argsort()[::-1][:top_n]]
            for row, topic in enumerate(joined.index)}


def topic_shares(papers: pd.DataFrame, column: str = "topic") -> pd.DataFrame:
    """Share of each month's papers that belong to each topic (rows: months)."""
    counts = papers.groupby(["month", column]).size().unstack(fill_value=0)
    return counts.div(counts.sum(axis=1), axis=0)


def trend_table(shares: pd.DataFrame, edge_months: int = 6) -> pd.DataFrame:
    """Slope of monthly share (percentage points per year) and early-vs-late change."""
    x = np.arange(len(shares)) / 12.0
    slopes = {c: np.polyfit(x, shares[c].to_numpy() * 100, 1)[0] for c in shares.columns}
    early = shares.head(edge_months).mean() * 100
    late = shares.tail(edge_months).mean() * 100
    return pd.DataFrame({
        "share_first_6m_pct": early, "share_last_6m_pct": late,
        "change_pp": late - early, "slope_pp_per_year": pd.Series(slopes),
        "growth_ratio": late / early.replace(0, np.nan),
    }).sort_values("slope_pp_per_year", ascending=False)


KEYWORDS = {
    "AI agents": r"\bagent(?:s|ic)?\b",
    "Reasoning": r"\breasoning\b",
    "Retrieval-augmented generation": r"\bretrieval[- ]augmented\b|\bRAG\b",
    "Reinforcement learning": r"\breinforcement learning\b|\bRLHF\b|\bGRPO\b",
    "Chain-of-thought": r"\bchain[- ]of[- ]thought\b|\bCoT\b",
    "Multimodal / vision-language": r"\bmulti-?modal\b|\bvision[- ]language\b|\bVLMs?\b",
    "Hallucination": r"\bhallucinat\w*",
    "Diffusion models": r"\bdiffusion\b",
    "LoRA / fine-tuning efficiency": r"\bLoRA\b|\bparameter[- ]efficient\b",
    "Mixture of experts": r"\bmixture[- ]of[- ]experts\b|\bMoE\b",
    "Test-time compute": r"\btest[- ]time\b",
}


def keyword_trends(papers: pd.DataFrame, keywords: dict[str, str] = KEYWORDS) -> pd.DataFrame:
    """Share of each month's papers whose title or abstract matches each pattern (case-insensitive)."""
    text = papers["title"] + " " + papers["abstract"]
    hits = pd.DataFrame({
        name: text.str.contains(pattern, flags=re.IGNORECASE, regex=True)
        for name, pattern in keywords.items()
    })
    hits["month"] = papers["month"].to_numpy()
    return hits.groupby("month").mean() * 100


def semantic_search(query: str, vectors: np.ndarray, papers: pd.DataFrame, model=None, top_k: int = 10) -> pd.DataFrame:
    query_vector = embed([query], model=model)[0]
    scores = vectors @ query_vector
    best = np.argsort(scores)[::-1][:top_k]
    return papers.iloc[best].assign(similarity=scores[best])


def keyword_search(query: str, papers: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    """Baseline: papers whose title or abstract contains every query word."""
    words = [w for w in re.findall(r"\w+", query.lower()) if w not in ENGLISH_STOP_WORDS]
    text = (papers["title"] + " " + papers["abstract"]).str.lower()
    mask = np.logical_and.reduce([text.str.contains(rf"\b{re.escape(w)}\b") for w in words]) if words else False
    return papers[mask].head(top_k)
