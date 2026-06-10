# Project Structure

Complete package and file hierarchy with the responsibility of each item.

```
rag-based-profile-matching/
├── resume_rag.py                  # Part A entry point (thin shim → cli.resume_rag_cli)
├── job_matcher.py                 # Part B entry point (thin shim → cli.job_matcher_cli)
├── pyproject.toml                 # PEP 621 packaging, deps, console scripts, tooling
├── requirements.txt               # Pinned dependency list (pip install -r)
├── .env.example                   # Documented environment configuration template
├── .gitignore                     # Python / secrets / artifacts / IDE ignores
│
├── README.md                      # Overview, quickstart, usage, metrics
│
├── docs/                          # All documentation (README stays at root)
│   ├── IMPLEMENTATION_PLAN.md     # Build phases, checklist, rationale
│   ├── PROJECT_STRUCTURE.md       # This file
│   ├── ASSIGNMENT.md              # Original brief + requirement-coverage matrix
│   ├── DEMO_SCRIPT.md             # 3-4 minute demo video walkthrough
│   └── architecture/              # Design docs
│       ├── ARCHITECTURE.md        # Architecture, principles, ADRs, extension points
│       ├── HLD.md                 # High-level design (subsystems, use cases, NFRs)
│       └── LLD.md                 # Low-level design (modules, algorithms, contracts)
│
├── data/                          # Sample dataset (generated, deterministic)
│   ├── resumes/                   # 32 diverse résumés (.txt)
│   ├── job_descriptions/          # 6 job descriptions (.txt)
│   └── ground_truth.json          # Relevance labels for evaluation
│
├── notebooks/
│   └── experimentation.ipynb      # Demo + metrics + hybrid-weight sweep + charts
│
├── scripts/
│   ├── generate_sample_data.py    # Generate résumés, JDs, and ground truth
│   └── run_evaluation.py          # Index + evaluate → outputs/evaluation.json
│
├── tests/                         # pytest suite (21 tests, offline-deterministic)
│   ├── conftest.py                # Fixtures (offline settings, sample résumé, indexed RAG)
│   ├── test_chunking.py
│   ├── test_metadata.py
│   ├── test_requirements.py
│   ├── test_embeddings_and_store.py
│   └── test_end_to_end.py
│
└── src/
    └── profile_matching/          # Installable package
        ├── __init__.py            # Public API: ResumeRAG, JobMatcher, get_settings
        ├── config.py              # Settings (pydantic-settings) + enums + get_settings()
        ├── logging_config.py      # configure_logging() / get_logger()
        │
        ├── models/                # Pydantic domain models (data contracts)
        │   ├── resume.py          # SectionType, ResumeMetadata, ResumeChunk, Resume
        │   ├── job.py             # MustHaveRequirement (+parser), JobDescription
        │   └── match.py           # MatchResult, MatchResponse (+ to_spec_dict)
        │
        ├── ingestion/             # ── Part A: document processing ──
        │   ├── loaders.py         # load_document/load_documents (.txt/.md/.pdf/.docx)
        │   └── chunking.py        # SectionAwareChunker (section-preserving chunking)
        │
        ├── extraction/            # ── Part A: metadata extraction ──
        │   └── metadata.py        # MetadataExtractor (Name, Skills, Experience, Education)
        │
        ├── embeddings/            # ── Part A: embeddings (pluggable) ──
        │   ├── base.py            # EmbeddingProvider ABC
        │   ├── providers.py       # HuggingFace / OpenAI / Cohere / Hashing(offline)
        │   └── factory.py         # build_embedding_provider() + fallback
        │
        ├── vectorstore/           # ── Part A: vector database (pluggable) ──
        │   ├── base.py            # VectorStore ABC, VectorRecord, SearchHit
        │   ├── chroma_store.py    # ChromaVectorStore (persistent)
        │   ├── memory_store.py    # InMemoryVectorStore (fallback / tests)
        │   └── factory.py         # build_vector_store() + fallback
        │
        ├── rag/                   # ── Part A: orchestration ──
        │   └── pipeline.py        # ResumeRAG (load→chunk→extract→embed→store→search)
        │
        ├── matching/              # ── Part B: job matching engine ──
        │   ├── requirements.py    # parse_job_description(), passes_must_haves()
        │   ├── hybrid_search.py   # HybridSearcher (dense + BM25 fusion)
        │   ├── scoring.py         # MatchScorer (0-100 score + reasoning)
        │   └── engine.py          # JobMatcher (orchestrates Part B)
        │
        ├── evaluation/            # ── Metrics ──
        │   └── metrics.py         # evaluate_matcher(): Precision@K, Recall@K, MRR, latency
        │
        ├── cli/                   # ── Interfaces ──
        │   ├── resume_rag_cli.py  # `resume-rag` console script
        │   └── job_matcher_cli.py # `job-matcher` console script
        │
        └── utils/                 # ── Shared helpers ──
            ├── text.py            # normalisation, tokenisation, truncation
            └── skills.py          # skill taxonomy + extraction/normalisation
```

## Layered dependency direction

```
cli  ─▶  rag.ResumeRAG / matching.JobMatcher
            │
            ├─▶ ingestion ─▶ utils
            ├─▶ extraction ─▶ utils.skills
            ├─▶ embeddings (base ◀ providers ◀ factory)
            ├─▶ vectorstore (base ◀ stores ◀ factory)
            └─▶ models   (depended on by everything; depends on nothing)
config & logging_config are cross-cutting (imported widely, import little).
```

Dependencies always point inward toward `models`/`config`; no cycles. Backends
(`embeddings`, `vectorstore`) are reached only through their ABCs + factories,
so swapping a provider touches exactly one module.

## Generated / git-ignored (not committed)

```
.vectorstore/        # ChromaDB persistence (regenerated by `resume_rag.py index`)
outputs/             # evaluation.json, match results
*.egg-info/          # build metadata
.venv/ __pycache__/  # environment / bytecode
```

## File counts

| Category | Count |
|----------|-------|
| Python modules (`src/`) | 33 |
| Tests | 5 files / 21 tests |
| Documentation | 7 markdown files |
| Sample résumés | 32 |
| Job descriptions | 6 |
