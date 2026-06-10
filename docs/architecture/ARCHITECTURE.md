# Architecture

## 1. Overview

The system is a two-stage Retrieval-Augmented Generation (RAG) pipeline for
matching résumés to job descriptions:

- **Part A (Indexing)** transforms a corpus of résumés into a searchable vector
  index enriched with structured metadata.
- **Part B (Matching)** turns a job description into a ranked, explainable list
  of candidate matches by retrieving against that index.

It is built around **ports and adapters (hexagonal architecture)**: stable
domain logic depends on narrow interfaces (`EmbeddingProvider`, `VectorStore`),
while concrete backends (HuggingFace, OpenAI, Cohere, ChromaDB, in-memory) are
interchangeable adapters resolved by factories from configuration.

## 2. Design principles

| Principle | How it's applied |
|-----------|------------------|
| **Separation of concerns** | Each stage (load, chunk, extract, embed, store, search, score) is an isolated module with one responsibility. |
| **Dependency inversion** | Core pipelines depend on `EmbeddingProvider` / `VectorStore` ABCs, never on a specific vendor. |
| **Graceful degradation** | Missing optional deps/keys → automatic fallback to offline backends, never a hard crash. |
| **12-factor config** | All tunables via env vars / `.env`, resolved once into an immutable `Settings`. |
| **Explainability** | Every match carries sub-scores, matched sections, excerpts, and natural-language reasoning. |
| **Testability** | Deterministic offline backends make the entire pipeline unit-testable without network or GPUs. |

## 3. Component diagram

```
                                   ┌────────────────────────┐
                                   │       config.py        │  env / .env → Settings
                                   └───────────┬────────────┘
                                               │ (injected)
        PART A — ResumeRAG                     ▼
 ┌───────────────────────────────────────────────────────────────────────────┐
 │  loaders ──▶ SectionAwareChunker ──▶ MetadataExtractor                      │
 │     │                │                       │                              │
 │  raw text        ResumeChunk[]          ResumeMetadata                      │
 │                       │                       │                             │
 │                       ▼                       ▼                             │
 │              EmbeddingProvider ─────▶ VectorRecord{vec, doc, metadata}      │
 │                  (factory)                    │                             │
 │                                               ▼                             │
 │                                        VectorStore.upsert()                 │
 └───────────────────────────────────────────────────────────────────────────┘
                                               │  (shared embedder + store)
        PART B — JobMatcher                     ▼
 ┌───────────────────────────────────────────────────────────────────────────┐
 │  parse_job_description ──▶ JobDescription{critical_skills, must_have}       │
 │            │                                                                │
 │            ▼                                                                │
 │  HybridSearcher: dense (VectorStore.query) ⊕ sparse (BM25)  ──▶ Candidates  │
 │            │                                                                │
 │            ▼                                                                │
 │  passes_must_haves (hard filter)                                            │
 │            │                                                                │
 │            ▼                                                                │
 │  MatchScorer: 0-100 score + matched_skills + excerpts + reasoning           │
 │            │                                                                │
 │            ▼                                                                │
 │  MatchResponse  ──▶  to_spec_dict()  (assignment JSON contract)             │
 └───────────────────────────────────────────────────────────────────────────┘
```

## 4. Key architectural decisions (ADRs, condensed)

### ADR-1: Provider abstraction for embeddings
**Decision:** define `EmbeddingProvider` ABC; resolve concrete provider via a
factory.
**Why:** the brief permits OpenAI/Cohere/HuggingFace. Abstracting them avoids
lock-in, enables cost/latency/quality trade-offs per environment, and lets tests
run on a deterministic offline provider.

### ADR-2: ChromaDB as default vector store, behind a `VectorStore` port
**Decision:** ChromaDB (persistent, local, zero-ops) is the default; an
in-memory store is the fallback.
**Why:** local-first developer experience and no managed-service credentials,
while Pinecone/Weaviate remain drop-in via the same interface.

### ADR-3: Section-aware chunking
**Decision:** chunk on detected résumé sections, sub-splitting only oversized
sections with overlap.
**Why:** preserves semantic coherence (a "Skills" block stays whole), improves
retrieval precision, and enables *section-level* match reasoning.

### ADR-4: Hybrid retrieval via over-fetch + re-rank
**Decision:** dense recall fetches a large candidate pool; BM25 re-ranks within
it, boosting critical-skill terms; signals are min-max fused.
**Why:** captures both semantic similarity and exact skill keywords without
maintaining a second full-corpus index, and is backend-agnostic.

### ADR-5: Transparent, rule-based scoring (not a black box)
**Decision:** the 0-100 score is a documented weighted blend of semantic
relevance, skill coverage, and experience adequacy.
**Why:** hiring decisions demand auditability; weights are configurable and the
reasoning string explains each contribution.

## 5. Data flow & contracts

- **Embeddings** are L2-normalised → cosine similarity = dot product.
- **Vector-store metadata** is scalar-only; list fields (skills, education) are
  serialised as `|`-delimited strings (`ResumeMetadata.as_store_metadata`).
- **Output contract** is pinned by `MatchResult` / `MatchResponse.to_spec_dict()`
  and guarded by `tests/test_end_to_end.py`.

## 6. Scalability & extension points

| Concern | Current | Path to scale |
|---------|---------|---------------|
| Corpus size | ChromaDB HNSW (local) | Pinecone/Weaviate adapter; sharded collections |
| Embedding throughput | Batched encode | GPU batch / managed embedding API |
| Ranking quality | Hybrid + heuristic score | Cross-encoder re-ranker; LLM rationale tier |
| Extraction accuracy | Rule-based | LLM extractor behind `MetadataExtractor` seam |
| Serving | CLI / library | Wrap `JobMatcher` in FastAPI; async batch matching |

## 7. Cross-cutting concerns

- **Configuration:** `config.py` (pydantic-settings).
- **Logging:** `logging_config.py` (structured, third-party noise suppressed).
- **Error handling:** loaders skip bad files; factories fall back; scoring clamps.
- **Reproducibility:** deterministic data generation, hashing embedder, pinned
  "current year" for experience inference.

See [HLD.md](HLD.md) and [LLD.md](LLD.md) for higher- and lower-level detail.
