## corpus
The sample has **12,060 papers**, 300–420 per month. That steady count comes from the sampling design (60 papers per 5-day window), not from arXiv's real volume, which is far larger and growing. So this analysis measures **what share of AI research goes to each topic**, not how many papers exist. Most papers have cs.LG (30%) or cs.CL (20%) as their primary category; the rest are papers cross-listed into the three sampled categories from computer vision, security, robotics and others.

## k
Silhouette scores are low (about 0.055–0.059) and almost flat across k = 15 to 35. That is normal for text: research topics overlap and blend into each other instead of forming separate islands, so no k is clearly "correct". I used **k = 25** because it was the most granular setting where every topic still read as one coherent research area when I checked its keywords and a sample of titles. Larger k mostly split big areas, such as LLM training, into fragments that were hard to tell apart.

## topics
The 25 topics cover the field well, from **LLM training and alignment** (6.4% of papers), **multimodal models** and **reasoning** to **AI for science**, **medical imaging** and **recommender systems**. Topic names were written by hand after reading each topic's class-based TF-IDF keywords and sample titles. On the map, related areas sit next to each other: medical imaging is close to clinical AI, and agents are close to reasoning and code. Some topics are broader than their names suggest; "Applied ML on signals & sensors", for instance, mixes EEG, battery and other sensor-prediction work.

## trends
- **LLM agents & code generation is the clear breakout topic.** Its share of papers rose from **2.1% in the first six months to 9.8% in the last six (4.6×)**. Growth was slow through mid-2025 and then accelerated sharply.
- **Reasoning** (3.2% → 6.1%) and **efficient inference & AI hardware** (2.2% → 5.6%, covering quantization, KV caches and serving) are the next fastest risers. This is consistent with the field's 2025–26 focus on reasoning models and the cost of running them.
- Falling shares: **graph neural networks** (4.3% → 1.8%), **federated learning** (3.0% → 1.1%), **neural network theory** (5.1% → 2.3%) and **multilingual/classic NLP** (7.1% → 4.4%). Falling share does not mean these areas are dying, only that they are growing more slowly than the rest of AI.
- The **retrieval/RAG topic's share fell** (5.8% → 4.1%) even though papers *mentioning* RAG doubled (see the next section). One plausible explanation is that retrieval is increasingly a component inside agent systems rather than a research topic of its own. That is an interpretation, not something this data proves.

## keywords
- Papers mentioning **agents** rose from **6.6% to 22.9%** of the sample (3.5×). By the end of the period, roughly one AI paper in five mentions agents.
- **Reasoning** mentions tripled (5.7% → 16.9%); **mixture of experts** (0.6% → 2.2%, 3.7×) and **test-time compute** (0.8% → 2.4%, 2.9×) grew fastest from small bases.
- **Hallucination** stayed nearly flat (1.8% → 2.0%), and **chain-of-thought** grew only modestly. Ideas that became standard practice tend to be mentioned less explicitly over time.
- Keyword matching is a blunt instrument (for example, "agent" also matches reinforcement-learning agents), so these numbers show direction and scale, not exact counts.

## search
For all three everyday-language queries, **keyword search found zero papers**, because no abstract contains every word of "making language models admit when they don't know something". Semantic search returned clearly relevant results for the first query (papers on LLM self-knowledge and hallucination) and partly relevant results for the others: *Video2Policy* and *Visual IRL* match "robots learning by watching videos", but some neighbours are only loosely related. For the fine-tuning query, most matches were about cheap **inference** rather than **fine-tuning**, which shows the limit of a small 22M-parameter embedding model. A larger embedding model, hybrid keyword + semantic ranking, or a cross-encoder re-ranker would be the next improvements; this is the same retrieval step a RAG system depends on.

## conclusions
**Findings**
1. **Agents are the defining trend of 2025–26 AI research**: their topic share grew 4.6× and mentions 3.5×, with the acceleration starting in the second half of 2025.
2. **Reasoning and efficiency rose together**: as models think longer, making inference cheaper became a research priority (quantization, KV cache, mixture of experts, test-time compute).
3. Graph neural networks, federated learning and classic NLP lost share as attention moved to LLM-based systems.
4. Embeddings make research searchable by meaning: semantic search found relevant papers where keyword search returned nothing.

**Limitations**
- **Sample, not census:** 60 papers per 5-day window, taken in submission order, so within a window the earliest submissions are over-represented. Shares are reliable for trends; absolute counts are not.
- Only papers in cs.CL, cs.LG or cs.AI were sampled; computer-vision-only or robotics-only papers are under-represented.
- k-means assigns each paper to exactly one topic, but many papers span several (for example, a multimodal agent).
- Topic names are my interpretation of keywords and titles.
- `all-MiniLM-L6-v2` is small and fast; larger embedding models would capture finer distinctions.
