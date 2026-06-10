# High-Level Design (HLD)

## 1. Purpose & scope

A system that ranks résumés against a job description using RAG techniques —
semantic retrieval over embeddings, augmented by keyword signals — and returns
explainable, scored matches. This HLD describes the system from the outside in:
actors, subsystems, data, and operational characteristics. Implementation-level
detail lives in [LLD.md](LLD.md).

## 2. Context

```
        ┌──────────────┐        job description / résumés        ┌───────────────┐
        │  Recruiter /  │ ─────────────────────────────────────▶ │  Profile-      │
        │  Hiring tool  │ ◀───────── ranked matches (JSON) ────── │  Matching      │
        └──────────────┘                                          │  System        │
                                                                  └──────┬────────┘
                                            embeddings (local/managed)   │
                                                  ▲                       ▼
                            ┌────────────────────────────┐     ┌──────────────────┐
                            │ Embedding provider          │     │  Vector database │
                            │ (HF / OpenAI / Cohere)      │     │  (ChromaDB)      │
                            └────────────────────────────┘     └──────────────────┘
```

**Primary actor:** a recruiter or an upstream ATS/hiring tool.
**External systems:** embedding backend, vector database. Both are optional —
the system runs standalone via offline fallbacks.

## 3. Subsystems

| # | Subsystem | Responsibility | Key modules |
|---|-----------|----------------|-------------|
| 1 | **Ingestion** | Read résumé files; normalise text | `ingestion/loaders.py` |
| 2 | **Chunking** | Split into section-tagged, size-bounded chunks | `ingestion/chunking.py` |
| 3 | **Metadata extraction** | Pull Name, Skills, Experience, Education | `extraction/metadata.py` |
| 4 | **Embedding** | Vectorise text via a pluggable provider | `embeddings/*` |
| 5 | **Vector store** | Persist vectors + metadata; ANN search + filter | `vectorstore/*` |
| 6 | **Indexing pipeline (Part A)** | Orchestrate 1–5 | `rag/pipeline.py` (`ResumeRAG`) |
| 7 | **JD parsing** | Derive critical skills + must-have requirements | `matching/requirements.py` |
| 8 | **Hybrid retrieval** | Dense + sparse search, resume-level aggregation | `matching/hybrid_search.py` |
| 9 | **Scoring** | 0-100 score, reasoning, excerpts | `matching/scoring.py` |
| 10 | **Matching engine (Part B)** | Orchestrate 7–9; enforce filters | `matching/engine.py` (`JobMatcher`) |
| 11 | **Evaluation** | Precision@K, Recall@K, MRR, latency | `evaluation/metrics.py` |
| 12 | **Interfaces** | CLIs, Python API | `cli/*`, `resume_rag.py`, `job_matcher.py` |

## 4. Primary use cases

### UC-1 — Index a résumé corpus (Part A)
1. Recruiter points the system at a directory of résumés.
2. Each file is loaded, chunked by section, and its metadata extracted.
3. Chunks are embedded and upserted (with metadata) into the vector store.
4. System returns an indexing summary (counts, provider, elapsed time).

### UC-2 — Match a job description (Part B)
1. Recruiter submits a JD (file or text).
2. System parses critical skills and must-have requirements.
3. Hybrid search retrieves and ranks candidate résumés.
4. Candidates failing must-haves are filtered out.
5. Remaining candidates are scored (0-100) with reasoning.
6. System returns the top-K matches as JSON.

### UC-3 — Evaluate retrieval quality
1. Given labelled ground truth, run all JDs through UC-2.
2. Compute Precision@K, Recall@K, MRR, and latency.
3. Emit a report (`outputs/evaluation.json`).

## 5. Data model (logical)

```
Resume ──1:N── ResumeChunk         ResumeMetadata (Name, Skills[], ExpYears, Education[])
   └───────── has ───────────────────────────┘
JobDescription ──1:N── MustHaveRequirement,  critical_skills[]
MatchResponse ──1:N── MatchResult (score, matched_skills[], excerpts[], reasoning)
```

Full field-level schemas: [LLD.md §2](LLD.md).

## 6. Technology choices

| Concern | Choice | Rationale |
|---------|--------|-----------|
| Language | Python 3.10+ | Ecosystem for embeddings/NLP |
| Config | pydantic-settings | Typed, env-driven, validated |
| Embeddings | sentence-transformers (default) | Local, free, strong quality |
| Vector DB | ChromaDB | Local, persistent, zero-ops |
| Sparse search | rank-bm25 | Lightweight, well-understood |
| Models/validation | Pydantic v2 | Strict contracts, easy serialisation |
| Tests | pytest | Standard, fast |

## 7. Non-functional requirements

| NFR | Target / approach |
|-----|-------------------|
| **Performance** | Sub-10 ms/query on the sample corpus (in-memory); HNSW for larger corpora |
| **Scalability** | Stateless engines; vector DB holds state; horizontally scalable behind an API |
| **Reliability** | Resilient ingestion (skip bad files); graceful backend fallback |
| **Security** | Secrets only via env; `.env` git-ignored; no PII logged at INFO |
| **Observability** | Structured logs; per-query latency and sub-scores in `--full` output |
| **Portability** | Pure-Python core; runs offline with zero external services |
| **Maintainability** | Modular, typed, tested; clear extension seams |

## 8. Deployment view

- **Local / CLI:** install, generate data, index, match.
- **Library:** import `ResumeRAG` / `JobMatcher` into another service.
- **Service (future):** wrap `JobMatcher.match` in a FastAPI endpoint; ChromaDB
  as a sidecar or managed Pinecone/Weaviate; embedding via managed API for
  elastic throughput.

## 9. Risks & mitigations

| Risk | Mitigation |
|------|------------|
| Heuristic extraction misses fields | Section-scoped parsing + taxonomy; LLM-extractor seam for upgrade |
| Embedding/vendor outage | Provider abstraction + offline fallback |
| Ranking bias / opacity | Transparent weighted score + reasoning; configurable weights |
| Large corpora exceed memory | ChromaDB persistence + ANN; pagination of candidate pool |
