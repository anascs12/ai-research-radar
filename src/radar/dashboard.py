"""Static interactive dashboard (Plotly) for GitHub Pages."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

TEMPLATE = "plotly_white"


def topic_map(papers: pd.DataFrame):
    data = papers.assign(year=papers["published"].dt.year.astype(str))
    fig = px.scatter(
        data, x="x", y="y", color="topic_name", hover_name="title",
        hover_data={"x": False, "y": False, "topic_name": True, "month": True},
        render_mode="webgl", template=TEMPLATE, opacity=.65,
        color_discrete_sequence=px.colors.qualitative.Alphabet,  # 26 distinct colours for 25 topics
        labels={"topic_name": "Topic", "month": "Month"},
        title="The AI research landscape: every dot is a paper, placed by meaning (t-SNE of sentence embeddings)",
    )
    fig.update_traces(marker=dict(size=4))
    fig.update_layout(height=760, xaxis=dict(visible=False), yaxis=dict(visible=False),
                      legend=dict(font=dict(size=10), itemsizing="constant"))
    return fig


def rising_topics(trends: pd.DataFrame, n: int = 8):
    table = pd.concat([trends.head(n), trends.tail(n)]).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=table["slope_pp_per_year"], y=table.index, orientation="h",
        marker_color=["#29A36A" if v > 0 else "#E4572E" for v in table["slope_pp_per_year"]],
        customdata=table[["share_first_6m_pct", "share_last_6m_pct"]].round(1),
        hovertemplate="%{y}<br>%{x:+.2f} pp per year<br>share: %{customdata[0]}% → %{customdata[1]}%<extra></extra>",
    ))
    fig.update_layout(template=TEMPLATE, height=560, title="Fastest-rising and fastest-falling topics (share of papers, percentage points per year)",
                      xaxis_title="Trend in share of papers (pp / year)")
    return fig


def topic_trends(shares: pd.DataFrame, topics: list[str]):
    data = (shares[topics] * 100).reset_index().melt(id_vars="month", var_name="topic", value_name="share")
    fig = px.line(data, x="month", y="share", color="topic", template=TEMPLATE, markers=False,
                  labels={"share": "% of papers", "month": ""},
                  title="Monthly share of papers for the biggest movers (3-month rolling average)")
    fig.update_layout(height=500, legend_title_text="")
    return fig


def keyword_chart(keywords: pd.DataFrame):
    data = keywords.rolling(3, min_periods=1).mean().reset_index().melt(id_vars="month", var_name="term", value_name="share")
    fig = px.line(data, x="month", y="share", color="term", template=TEMPLATE,
                  labels={"share": "% of papers mentioning the term", "month": ""},
                  title="How often key ideas appear in AI papers (3-month rolling average)")
    fig.update_layout(height=520, legend_title_text="")
    return fig


def write(figures: list, path: Path, title: str, intro_html: str) -> Path:
    parts = [f.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False) for i, f in enumerate(figures)]
    cards = "\n".join(f'<section class="card">{p}</section>' for p in parts)
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; background: #F4F6FB; color: #1C2333; }}
  header {{ background: linear-gradient(120deg, #0F172A, #4C1D95); color: #fff; padding: 36px 24px; }}
  header h1 {{ margin: 0 0 8px; font-size: 28px; }}
  header p {{ margin: 4px 0; max-width: 920px; opacity: .92; line-height: 1.5; }}
  header a {{ color: #C4B5FD; }}
  main {{ max-width: 1240px; margin: 0 auto; padding: 24px 16px 48px; }}
  .card {{ background: #fff; border-radius: 14px; padding: 12px; margin-bottom: 22px;
          box-shadow: 0 2px 10px rgba(15, 23, 42, .08); overflow-x: auto; }}
  footer {{ text-align: center; font-size: 13px; color: #5A6478; padding: 0 16px 32px; }}
</style></head>
<body>
<header><h1>{title}</h1>{intro_html}</header>
<main>{cards}</main>
<footer>Paper metadata from the arXiv API. Thank you to arXiv for use of its open access interoperability. Analysis by Anas Khan.</footer>
</body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path
