"""Download a time-stratified sample of AI papers from the arXiv API.

The arXiv API returns Atom XML and asks clients to wait 3 seconds between
requests (https://info.arxiv.org/help/api/tou.html). Responses are cached per
window in data/raw/, so re-running never repeats a request.

Sampling: the study period is cut into 5-day windows and the first ``per_window``
papers (by submission time) are taken from each. This spreads the sample evenly
over time, which is what trend analysis needs, but it is not a uniform random
sample within each window (see README limitations).
"""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

API_URL = "https://export.arxiv.org/api/query"
CATEGORIES = ("cs.CL", "cs.LG", "cs.AI")
NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
USER_AGENT = "ai-research-radar/1.0 (data science portfolio project)"


@dataclass(frozen=True)
class Window:
    start: date
    end: date

    @property
    def query(self) -> str:
        cats = " OR ".join(f"cat:{c}" for c in CATEGORIES)
        return f"({cats}) AND submittedDate:[{self.start:%Y%m%d}0000 TO {self.end:%Y%m%d}2359]"


def windows(start: date, end: date, days: int = 5) -> list[Window]:
    """Consecutive, non-overlapping windows covering [start, end]."""
    out, current = [], start
    while current <= end:
        stop = min(current + timedelta(days=days - 1), end)
        out.append(Window(current, stop))
        current = stop + timedelta(days=1)
    return out


def parse_feed(xml_text: str) -> list[dict]:
    """Extract paper records from an arXiv Atom feed."""
    root = ET.fromstring(xml_text)
    papers = []
    for entry in root.findall("a:entry", NS):
        def text(tag: str) -> str:
            node = entry.find(tag, NS)
            return " ".join((node.text or "").split()) if node is not None else ""

        arxiv_id = text("a:id").rsplit("/abs/", 1)[-1]
        primary = entry.find("arxiv:primary_category", NS)
        papers.append({
            "arxiv_id": arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id,
            "title": text("a:title"),
            "abstract": text("a:summary"),
            "published": text("a:published"),
            "primary_category": primary.get("term") if primary is not None else None,
            "categories": " ".join(c.get("term") for c in entry.findall("a:category", NS)),
            "n_authors": len(entry.findall("a:author", NS)),
        })
    return papers


class ArxivClient:
    def __init__(self, cache_dir: Path | str = "data/raw", pause_s: float = 3.1, retries: int = 5):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.pause_s, self.retries = pause_s, retries
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT

    def fetch_window(self, window: Window, per_window: int) -> list[dict]:
        cache = self.cache_dir / f"{window.start:%Y%m%d}_{per_window}.xml"
        if not cache.exists():
            params = {"search_query": window.query, "start": 0, "max_results": per_window,
                      "sortBy": "submittedDate", "sortOrder": "ascending"}
            for attempt in range(self.retries):
                response = self.session.get(API_URL, params=params, timeout=90)
                if response.status_code in (429, 500, 502, 503):
                    time.sleep(self.pause_s * (2 ** attempt))  # back off politely
                    continue
                response.raise_for_status()
                if "<entry>" not in response.text and attempt < self.retries - 1:
                    time.sleep(self.pause_s * (2 ** attempt))  # arXiv sometimes returns an empty page
                    continue
                cache.write_text(response.text, encoding="utf-8")
                break
            else:
                raise RuntimeError(f"arXiv kept failing for window starting {window.start}")
            time.sleep(self.pause_s)
        return parse_feed(cache.read_text(encoding="utf-8"))


def build_corpus(start: date, end: date, per_window: int = 60, root: Path = Path(".")) -> pd.DataFrame:
    client = ArxivClient(root / "data" / "raw")
    spans = windows(start, end)
    rows = []
    for i, window in enumerate(spans, 1):
        rows.extend(client.fetch_window(window, per_window))
        if i % 20 == 0 or i == len(spans):
            print(f"  {i}/{len(spans)} windows · {len(rows):,} papers so far", flush=True)
    papers = pd.DataFrame(rows).drop_duplicates("arxiv_id")
    papers["published"] = pd.to_datetime(papers["published"], utc=True)
    papers["month"] = papers["published"].dt.strftime("%Y-%m")
    papers = papers[papers["abstract"].str.len() > 100].sort_values("published").reset_index(drop=True)
    out = root / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    papers.to_parquet(out / "papers.parquet", index=False)
    return papers


if __name__ == "__main__":
    corpus = build_corpus(date(2024, 1, 1), date(2026, 9, 30))
    print(f"Saved {len(corpus):,} papers, {corpus['month'].min()} to {corpus['month'].max()}")
