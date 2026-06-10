# Assignment Specification & Requirement Coverage

This document reproduces the original brief and maps **every requirement** to
its concrete implementation, demonstrating full coverage.

---

## Brief

### Learning Objectives
- Implement document chunking and embedding
- Build vector databases
- Create retrieval pipelines
- Understand semantic search

### Part A: RAG System Setup (50%) — `resume_rag.py`
**Document Processing Pipeline**
- Load resumes using file system tools
- Chunk documents intelligently (preserve sections like Education, Experience)
- Generate embeddings using OpenAI/Cohere/HuggingFace models
- Store in vector database (ChromaDB, Pinecone, or Weaviate)

**Metadata Extraction**
- Extract key fields: Name, Skills, Experience Years, Education
- Store metadata alongside embeddings for filtering

### Part B: Job Matching Engine (50%) — `job_matcher.py`
**Semantic Search**
- Accept job description as input
- Convert JD to embedding
- Retrieve top-K similar resumes (K=10)
- Implement hybrid search (semantic + keyword for critical skills)

**Ranking & Scoring**
- Score matches (0-100 scale)
- Provide match reasoning (which sections matched)
- Filter by must-have requirements (e.g., "5+ years Python")

**Output Format**
```json
{ "job_description": "...", "top_matches": [ { "candidate_name": "John Doe",
  "resume_path": "resumes/john_doe.pdf", "match_score": 92,
  "matched_skills": ["Python", "Machine Learning"], "relevant_excerpts": ["..."],
  "reasoning": "Strong match for ML experience..." } ] }
```

### Submission Guidelines — Deliverables
- Complete RAG implementation
- Dataset: 30+ diverse resumes, 5+ job descriptions
- Jupyter notebook with experimentation and analysis
- Performance metrics: retrieval accuracy, latency
- Demo video (3-4 minutes)

---

## Requirement Coverage Matrix

### Part A

| # | Requirement | Implementation | Status |
|---|-------------|----------------|--------|
| A1 | Load resumes from filesystem | `ingestion/loaders.py` (`.txt/.md/.pdf/.docx`, resilient batch load) | ✅ |
| A2 | Intelligent, section-preserving chunking | `ingestion/chunking.py` — `SectionAwareChunker` detects headings, keeps sections intact, overlap-splits oversized ones | ✅ |
| A3 | Embeddings via OpenAI/Cohere/HuggingFace | `embeddings/providers.py` + `factory.py` (HuggingFace default; OpenAI & Cohere; offline hashing fallback) | ✅ |
| A4 | Store in a vector database | `vectorstore/chroma_store.py` (ChromaDB, persistent) + in-memory fallback | ✅ |
| A5 | Extract Name, Skills, Experience Years, Education | `extraction/metadata.py` — `MetadataExtractor` | ✅ |
| A6 | Store metadata alongside embeddings for filtering | `rag/pipeline.py` writes `ResumeMetadata.as_store_metadata()` into each `VectorRecord.metadata` | ✅ |

### Part B

| # | Requirement | Implementation | Status |
|---|-------------|----------------|--------|
| B1 | Accept job description as input | `job_matcher.py` (`--jd-file` / `--jd-text`), `JobMatcher.match()` | ✅ |
| B2 | Convert JD to embedding | `matching/hybrid_search.py` embeds `job.search_text` | ✅ |
| B3 | Retrieve top-K similar resumes (K=10) | `MatchingSettings.top_k = 10`; aggregated to resume level | ✅ |
| B4 | Hybrid search (semantic + keyword for critical skills) | `matching/hybrid_search.py` — dense + BM25 fusion with critical-skill term boosting | ✅ |
| B5 | Score matches (0-100) | `matching/scoring.py` — composite of semantic + skill coverage + experience | ✅ |
| B6 | Match reasoning (which sections matched) | `MatchScorer._build_reasoning()` + `matched_sections` | ✅ |
| B7 | Filter by must-have requirements ("5+ years Python") | `matching/requirements.py` — `MustHaveRequirement.parse` + `passes_must_haves` | ✅ |
| B8 | Exact output schema | `models/match.py` — `MatchResponse.to_spec_dict()` | ✅ |

### Submission

| # | Requirement | Implementation | Status |
|---|-------------|----------------|--------|
| S1 | Complete RAG implementation | `src/profile_matching/**` (15 modules) | ✅ |
| S2 | 30+ resumes, 5+ job descriptions | `scripts/generate_sample_data.py` → **32 resumes, 6 JDs** in `data/` | ✅ |
| S3 | Jupyter notebook (experimentation + analysis) | `notebooks/experimentation.ipynb` | ✅ |
| S4 | Performance metrics (retrieval accuracy, latency) | `evaluation/metrics.py` + `scripts/run_evaluation.py` (Precision@K, Recall@K, MRR, latency) | ✅ |
| S5 | Demo video (3-4 min) | See `docs/DEMO_SCRIPT.md` for the recording walkthrough (video is recorded separately) | ⏳ author-recorded |

> **Note on vector DB choice:** the brief allows ChromaDB, Pinecone, or Weaviate.
> ChromaDB was chosen as the default (local, zero-ops, persistent). The
> `VectorStore` abstraction (`vectorstore/base.py`) makes Pinecone/Weaviate
> drop-in alternatives — implement the interface and register it in the factory.
