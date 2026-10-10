from datetime import date

import numpy as np
import pandas as pd
import pytest

from radar.arxiv import Window, parse_feed, windows
from radar.nlp import keyword_search, keyword_trends, topic_keywords, topic_shares, trend_table

FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2501.01234v2</id>
    <published>2025-01-02T10:00:00Z</published>
    <title>Agents That   Use Tools</title>
    <summary>We study language
      agents that call external tools.</summary>
    <author><name>A</name></author><author><name>B</name></author>
    <arxiv:primary_category term="cs.CL"/>
    <category term="cs.CL"/><category term="cs.AI"/>
  </entry>
</feed>"""


def test_windows_cover_the_period_without_gaps_or_overlap():
    spans = windows(date(2024, 1, 1), date(2024, 1, 12), days=5)
    assert [(w.start.day, w.end.day) for w in spans] == [(1, 5), (6, 10), (11, 12)]


def test_window_query_combines_categories_and_dates():
    query = Window(date(2025, 1, 1), date(2025, 1, 5)).query
    assert "cat:cs.CL OR cat:cs.LG OR cat:cs.AI" in query
    assert "submittedDate:[202501010000 TO 202501052359]" in query


def test_parse_feed_normalises_whitespace_and_strips_version():
    [paper] = parse_feed(FEED)
    assert paper["arxiv_id"] == "2501.01234"
    assert paper["title"] == "Agents That Use Tools"
    assert paper["abstract"] == "We study language agents that call external tools."
    assert paper["primary_category"] == "cs.CL"
    assert paper["categories"] == "cs.CL cs.AI"
    assert paper["n_authors"] == 2


def test_topic_keywords_prefers_words_distinctive_to_a_topic():
    texts = ["protein folding structure prediction"] * 6 + ["speech recognition audio transcription"] * 6
    labels = np.array([0] * 6 + [1] * 6)
    keywords = topic_keywords(texts, labels, top_n=3)
    assert "protein" in " ".join(keywords[0]) and "protein" not in " ".join(keywords[1])
    assert "speech" in " ".join(keywords[1])


def test_trend_table_finds_the_rising_topic():
    shares = pd.DataFrame({"rising": np.linspace(0.1, 0.5, 24), "falling": np.linspace(0.5, 0.1, 24)},
                          index=[f"m{i:02d}" for i in range(24)])
    table = trend_table(shares)
    assert table.index[0] == "rising"
    assert table.loc["rising", "slope_pp_per_year"] == pytest.approx(20.87, rel=0.01)


def test_topic_shares_rows_sum_to_one():
    papers = pd.DataFrame({"month": ["2025-01"] * 3 + ["2025-02"] * 2, "topic": [0, 0, 1, 1, 1]})
    shares = topic_shares(papers)
    assert np.allclose(shares.sum(axis=1), 1.0)
    assert shares.loc["2025-01", 0] == pytest.approx(2 / 3)


def test_keyword_trends_match_case_insensitively_on_word_boundaries():
    papers = pd.DataFrame({
        "month": ["2025-01", "2025-01"],
        "title": ["Agentic RAG", "Paragraph ranking"],  # 'rag' inside 'Paragraph' must not count
        "abstract": ["", ""],
    })
    trends = keyword_trends(papers, {"RAG": r"\bRAG\b", "Agents": r"\bagent(?:s|ic)?\b"})
    assert trends.loc["2025-01", "RAG"] == 50.0
    assert trends.loc["2025-01", "Agents"] == 50.0


def test_keyword_search_requires_every_word():
    papers = pd.DataFrame({"title": ["robot learning from video", "robot arms"], "abstract": ["", ""]})
    assert keyword_search("robot video", papers)["title"].tolist() == ["robot learning from video"]
