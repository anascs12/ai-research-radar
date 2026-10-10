"""Generate notebooks/ai_research_radar.ipynb; interpretation cells come from scripts/findings.md."""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell


def findings(section: str) -> str:
    """Interpretation text lives in scripts/findings.md under '## <section>' headings."""
    path = ROOT / "scripts" / "findings.md"
    if not path.exists():
        return "_Interpretation pending._"
    text = path.read_text()
    marker = f"## {section}\n"
    if marker not in text:
        return "_Interpretation pending._"
    return text.split(marker, 1)[1].split("\n## ", 1)[0].strip()


cells = [
    md("""# AI Research Radar: Mapping the AI Research Landscape with Sentence Embeddings

**Author:** Anas Khan · Data Science student

Thousands of AI papers appear on arXiv every month, so nobody can read them all. This project uses **sentence embeddings**, the same technology behind semantic search and retrieval-augmented generation (RAG), to answer:

1. What are the main research topics in AI right now?
2. Which topics are **rising** and which are **fading** (January 2024 to September 2026)?
3. How often do key ideas such as *agents*, *reasoning* and *RAG* appear over time?
4. Does **semantic search** find relevant papers that keyword search misses?

**Method in one line:** arXiv API → title + abstract → `all-MiniLM-L6-v2` embeddings (384 dimensions, CPU) → k-means topics → class-based TF-IDF keywords → monthly topic shares and trend slopes → t-SNE map → semantic search app."""),
    code("""import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from radar import nlp, dashboard as D

ROOT = Path("..").resolve()
FIG = ROOT / "reports" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook")
def save(fig, name): fig.savefig(FIG / f"{name}.png", dpi=150, bbox_inches="tight")

papers = pd.read_parquet(ROOT / "data" / "processed" / "papers_with_topics.parquet")
vectors = np.load(ROOT / "data" / "processed" / "embeddings.npy")
topics = pd.read_csv(ROOT / "data" / "processed" / "topics.csv")
print(f"{len(papers):,} papers · {papers['month'].min()} to {papers['month'].max()} · embeddings {vectors.shape}")"""),
    md("## 1. The corpus"),
    code("""per_month = papers.groupby("month").size()
print(f"Papers per month: min {per_month.min()}, median {int(per_month.median())}, max {per_month.max()}")
print("Primary categories (top 8):")
print((papers["primary_category"].value_counts(normalize=True).head(8) * 100).round(1).to_string())"""),
    md(findings("corpus")),
    md("## 2. How many topics? Silhouette scores"),
    code("""k_scores = nlp.choose_k(vectors, candidates=(15, 20, 25, 30, 35))
k_scores.round(4)"""),
    md(findings("k")),
    md("## 3. The topics"),
    code("""pd.set_option("display.max_colwidth", 90)
topics.assign(share_pct=(topics["papers"] / topics["papers"].sum() * 100).round(1))[["name", "papers", "share_pct", "keywords"]]"""),
    code("""fig, ax = plt.subplots(figsize=(13, 10))
palette = sns.color_palette("husl", papers["topic"].nunique())
from adjustText import adjust_text
labels = []
for i, (topic, g) in enumerate(papers.groupby("topic_name")):
    ax.scatter(g["x"], g["y"], s=3, alpha=.45, color=palette[i])
    labels.append(ax.text(g["x"].median(), g["y"].median(), topic, fontsize=8.5, weight="bold", ha="center",
                          bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=.8)))
ax.margins(x=0.12, y=0.04)
adjust_text(labels, ax=ax, expand=(1.15, 1.4), arrowprops=dict(arrowstyle="-", color="grey", lw=.6))
ax.set_axis_off()
ax.set_title("The AI research landscape (t-SNE of sentence embeddings; labels at topic medians)", fontsize=13)
save(fig, "01_topic_map")
plt.show()"""),
    md(findings("topics")),
    md("## 4. Rising and falling topics"),
    code("""shares = nlp.topic_shares(papers, "topic_name")
trends = nlp.trend_table(shares)
trends.round(2)"""),
    code("""top = pd.concat([trends.head(8), trends.tail(8)])
fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(top.index[::-1], top["slope_pp_per_year"][::-1],
        color=["#29A36A" if v > 0 else "#E4572E" for v in top["slope_pp_per_year"][::-1]])
ax.axvline(0, color="black", lw=.8)
ax.set(title="Trend in each topic's share of papers (percentage points per year)", xlabel="pp / year")
save(fig, "02_rising_falling")
plt.show()"""),
    code("""movers = list(trends.index[:4]) + list(trends.index[-3:])
fig, ax = plt.subplots(figsize=(12, 5.5))
(shares[movers] * 100).rolling(3, min_periods=1).mean().plot(ax=ax, lw=2.2)
ax.set(title="Monthly share of papers, biggest movers (3-month rolling average)", xlabel="", ylabel="% of papers")
ax.legend(fontsize=8.5, frameon=False, ncol=2)
save(fig, "03_movers")
plt.show()"""),
    md(findings("trends")),
    md("## 5. Key ideas over time"),
    code("""kw = nlp.keyword_trends(papers)
summary = pd.DataFrame({"first_6m_pct": kw.head(6).mean(), "last_6m_pct": kw.tail(6).mean()})
summary["change_pp"] = summary["last_6m_pct"] - summary["first_6m_pct"]
summary["ratio"] = summary["last_6m_pct"] / summary["first_6m_pct"]
summary.sort_values("change_pp", ascending=False).round(2)"""),
    code("""fig, ax = plt.subplots(figsize=(12, 6))
kw.rolling(3, min_periods=1).mean().plot(ax=ax, lw=2, color=sns.color_palette("tab20", kw.shape[1]))
ax.set(title="Share of AI papers mentioning each idea (3-month rolling average)", xlabel="", ylabel="% of papers")
ax.legend(fontsize=8.5, frameon=False, ncol=2)
save(fig, "04_keywords")
plt.show()"""),
    md(findings("keywords")),
    md("""## 6. Semantic search vs. keyword search

Keyword search only finds papers that contain the exact words typed. Semantic search compares **meanings**: the query is embedded with the same model, and papers are ranked by cosine similarity."""),
    code("""model = nlp.load_model()
queries = ["making language models admit when they don't know something",
           "robots learning tasks by watching videos",
           "cheaper ways to fine-tune big models on a single GPU"]
for q in queries:
    sem = nlp.semantic_search(q, vectors, papers, model=model, top_k=5)
    kws = nlp.keyword_search(q, papers, top_k=5)
    print(f"\\nQUERY: {q}")
    print(f"  keyword search: {len(nlp.keyword_search(q, papers, top_k=10**6))} papers contain every word")
    for _, r in sem.iterrows():
        print(f"  [{r['similarity']:.2f}] {r['title'][:110]}")"""),
    md(findings("search")),
    md("## 7. Interactive dashboard"),
    code("""import shutil
figs = [D.topic_map(papers), D.rising_topics(trends), D.topic_trends(shares.rolling(3, min_periods=1).mean(), movers),
        D.keyword_chart(kw)]
intro = (f"<p>{len(papers):,} AI papers from arXiv ({papers['month'].min()} to {papers['month'].max()}), "
         "embedded with a sentence-transformer and grouped into topics. Hover any dot to see the paper; "
         "click legend items to show or hide topics.</p>"
         "<p>Code, notebook and semantic search app: "
         "<a href='https://github.com/anascs12/ai-research-radar'>github.com/anascs12/ai-research-radar</a></p>")
out = D.write(figs, ROOT / "reports" / "dashboard.html", "AI Research Radar", intro)
(ROOT / "docs").mkdir(exist_ok=True)
shutil.copyfile(out, ROOT / "docs" / "index.html")
print("Dashboard:", out.relative_to(ROOT), "+ docs/index.html")"""),
    md("## 8. Conclusions and limitations\n\n" + findings("conclusions")),
]

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = ROOT / "notebooks" / "ai_research_radar.ipynb"
nbf.write(nb, out)
print("wrote", out)
