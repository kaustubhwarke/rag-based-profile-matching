# Implementation Plan

This plan documents how the system was built and the order of execution. All
phases are **complete**; the checklist doubles as a delivery audit.

## Phase 0 — Project scaffolding ✅
- [x] `pyproject.toml` (PEP 621, optional-deps groups, console scripts, tooling)
- [x] `requirements.txt`, `.env.example`, hardened `.gitignore`
- [x] Package layout under `src/profile_matching/`
- [x] Typed configuration (`config.py`) + structured logging (`logging_config.py`)

## Phase 1 — Domain models ✅
- [x] `models/resume.py` — `SectionType`, `ResumeMetadata`, `ResumeChunk`, `Resume`
- [x] `models/job.py` — `MustHaveRequirement` (+ parser), `JobDescription`
- [x] `models/match.py` — `MatchResult`, `MatchResponse` (+ `to_spec_dict`)

## Phase 2 — Part A: ingestion & indexing ✅
- [x] Filesystem loaders (`.txt/.md/.pdf/.docx`) with resilient batch loading
- [x] `SectionAwareChunker` (heading detection, overlap sub-splitting)
- [x] `MetadataExtractor` (Name, Skills, Experience Years, Education)
- [x] Embedding providers + factory (HuggingFace/OpenAI/Cohere/hashing)
- [x] Vector stores + factory (ChromaDB / in-memory)
- [x] `ResumeRAG` pipeline tying it together; metadata stored with vectors

## Phase 3 — Part B: matching engine ✅
- [x] JD parsing → critical skills + must-have requirements
- [x] `HybridSearcher` (dense + BM25 fusion, critical-skill boosting)
- [x] Must-have filtering (`passes_must_haves`)
- [x] `MatchScorer` (0-100 composite score, reasoning, excerpts, matched sections)
- [x] `JobMatcher` orchestration → `MatchResponse` (exact JSON schema)

## Phase 4 — Interfaces ✅
- [x] `resume_rag.py` (index / inspect / search / stats)
- [x] `job_matcher.py` (file/text input, top-k, must-have toggle, output, full)
- [x] Packaged CLIs + console-script entry points

## Phase 5 — Dataset & evaluation ✅
- [x] `scripts/generate_sample_data.py` → 32 résumés, 6 JDs, ground truth
- [x] `evaluation/metrics.py` → Precision@K, Recall@K, MRR, latency
- [x] `scripts/run_evaluation.py` → `outputs/evaluation.json`

## Phase 6 — Notebook & docs ✅
- [x] `notebooks/experimentation.ipynb` (demo + metrics + weight sweep + charts)
- [x] `README.md`; `docs/architecture/{ARCHITECTURE,HLD,LLD}.md`
- [x] `docs/{IMPLEMENTATION_PLAN,PROJECT_STRUCTURE,ASSIGNMENT,DEMO_SCRIPT}.md`

## Phase 7 — Testing & quality ✅
- [x] 21 pytest tests (unit + integration), offline-deterministic
- [x] Ruff / Black / Mypy configuration
- [x] End-to-end smoke run validated (32 résumés indexed, matches produced)

---

## Build order rationale

The build proceeds **contracts-first**: configuration and domain models are
fixed before behaviour, so every subsequent module codes against stable types.
Within each part, dependencies flow one direction (ingestion → embedding →
store → retrieval → scoring), which keeps modules independently testable and
mirrors the runtime data flow.

## Effort distribution (relative)

| Area | Share |
|------|-------|
| Part A (ingestion, chunking, extraction, embeddings, store) | ~40% |
| Part B (parsing, hybrid search, scoring, engine) | ~35% |
| Evaluation, dataset, notebook | ~15% |
| Docs, tests, tooling | ~10% |

## Future iterations (not in scope, designed-for)

1. **Cross-encoder re-ranker** over the top-K for sharper ordering.
2. **LLM rationale tier** — Claude-generated, citation-grounded explanations for
   the highest-value matches (the `reasoning` field is the seam).
3. **FastAPI service** wrapping `JobMatcher` with async batch matching.
4. **LLM-based metadata extraction** behind the `MetadataExtractor` interface.
5. **Pinecone/Weaviate adapters** implementing `VectorStore`.
