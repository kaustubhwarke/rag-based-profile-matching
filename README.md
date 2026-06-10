# RAG-Based Profile Matching

> Enterprise-grade Retrieval-Augmented Generation system that matches résumés to
> job descriptions using section-aware chunking, pluggable embeddings, a vector
> database, and hybrid (semantic + keyword) search with explainable scoring.

[![python](https://img.shields.io/badge/python-3.10%2B-blue)](#)
[![tests](https://img.shields.io/badge/tests-21%20passing-brightgreen)](#testing)
[![license](https://img.shields.io/badge/license-MIT-green)](#license)

This repository implements the assignment in two parts:

| Part | Deliverable | Entry point |
| ---- | ----------- | ----------- |
| **A — RAG System Setup** | Document processing, chunking, embeddings, vector store, metadata extraction | [`resume_rag.py`](resume_rag.py) |
| **B — Job Matching Engine** | Semantic + hybrid search, ranking/scoring, must-have filtering, JSON output | [`job_matcher.py`](job_matcher.py) |

Full design docs: [ARCHITECTURE](docs/architecture/ARCHITECTURE.md) ·
[HLD](docs/architecture/HLD.md) · [LLD](docs/architecture/LLD.md) ·
[IMPLEMENTATION_PLAN](docs/IMPLEMENTATION_PLAN.md) ·
[PROJECT_STRUCTURE](docs/PROJECT_STRUCTURE.md) ·
[ASSIGNMENT spec & coverage](docs/ASSIGNMENT.md)

---

## Highlights

- **Section-aware chunking** — preserves Experience / Education / Skills / Projects
  sections so retrieval and reasoning operate on coherent units.
- **Pluggable embeddings** — HuggingFace (default), OpenAI, Cohere, plus a
  deterministic **offline hashing fallback** so the whole system runs with *zero*
  external dependencies or API keys.
- **Pluggable vector store** — ChromaDB (persistent) with an in-memory fallback.
- **Hybrid search** — dense semantic retrieval fused with BM25 keyword scoring,
  with critical-skill term boosting.
- **Explainable scoring** — transparent 0-100 score (semantic + skill coverage +
  experience) with per-section match reasoning and relevant excerpts.
- **Hard requirement filtering** — e.g. *"5+ years Python"* gates candidates
  before ranking.
- **Production posture** — typed config (pydantic-settings), structured logging,
  graceful degradation, 21 unit/integration tests, evaluation harness, CLIs.

## Architecture at a glance

```
            PART A — INDEXING                         PART B — MATCHING
 ┌───────────────────────────────────┐   ┌──────────────────────────────────────┐
 │ resumes/  → Loader → Section-aware │   │ JD → Parser (critical skills +        │
 │            chunker → Metadata      │   │      must-haves)                      │
 │            extractor               │   │   → Hybrid search (dense + BM25)      │
 │          → Embeddings → Vector DB ─┼───┼─▶ → Must-have filter                  │
 │            (+ metadata)            │   │   → 0-100 scoring + reasoning         │
 └───────────────────────────────────┘   │   → top-K MatchResponse (JSON)        │
                                          └──────────────────────────────────────┘
```

See [ARCHITECTURE.md](docs/architecture/ARCHITECTURE.md) for the full picture.

## Quickstart

```bash
# 1. (optional) create a virtualenv
python -m venv .venv && . .venv/Scripts/activate      # Windows
# python -m venv .venv && source .venv/bin/activate   # macOS/Linux

# 2. install — minimal (offline) core
pip install -e .
#    …or full production stack (HuggingFace + ChromaDB + loaders)
pip install -e ".[all]"

# 3. generate the sample dataset (32 resumes, 6 JDs, ground truth)
python scripts/generate_sample_data.py

# 4. fastest way to see it end-to-end (Part A + Part B in one process; works offline)
python scripts/demo.py

# …or run the parts separately:
# 4a. Part A — index the resumes
python resume_rag.py index --resume-dir data/resumes
# 4b. Part B — match a job description
python job_matcher.py --jd-file data/job_descriptions/jd_01_ml_engineer.txt --top-k 10
```

> The separate Part A / Part B commands persist the index between runs only with
> ChromaDB (`pip install -e ".[all]"`). With the offline in-memory fallback, use
> `scripts/demo.py` (or the notebook), which indexes and matches in one process.

> **Running fully offline:** the default config targets HuggingFace + ChromaDB.
> If those aren't installed, the system automatically falls back to the offline
> `hashing` embedder + in-memory store (you'll see a one-line warning). To make
> that explicit and persistent-free for demos/tests:
>
> ```bash
> # PowerShell
> $env:PM_EMBEDDING__PROVIDER='hashing'; $env:PM_VECTORSTORE__BACKEND='memory'
> # bash
> export PM_EMBEDDING__PROVIDER=hashing PM_VECTORSTORE__BACKEND=memory
> ```
>
> Note: with the in-memory store, indexing and matching must run in the **same
> process** (use the notebook or `scripts/run_evaluation.py`). Cross-process CLI
> use requires the persistent ChromaDB backend.

## Usage

### Part A — `resume_rag.py`

```bash
python resume_rag.py index   --resume-dir data/resumes [--reset]
python resume_rag.py inspect data/resumes/0001_priya_sharma.txt   # show metadata + chunks
python resume_rag.py search  "senior python engineer with aws"     # raw chunk search
python resume_rag.py stats
```

### Part B — `job_matcher.py`

```bash
python job_matcher.py --jd-file <path> [--top-k 10] [--no-must-haves] [--output out.json] [--full]
python job_matcher.py --jd-text "Senior ML engineer, 5+ years Python, PyTorch, AWS"
```

Output matches the required schema exactly:

```json
{
  "job_description": "...",
  "top_matches": [
    {
      "candidate_name": "John Doe",
      "resume_path": "resumes/john_doe.pdf",
      "match_score": 92,
      "matched_skills": ["Python", "Machine Learning"],
      "relevant_excerpts": ["..."],
      "reasoning": "Strong match for ML experience..."
    }
  ]
}
```

`--full` additionally emits diagnostics (semantic/keyword sub-scores, matched
sections, latency) for observability.

### Python API

```python
from profile_matching import ResumeRAG, JobMatcher

rag = ResumeRAG()
rag.index_directory("data/resumes")

matcher = JobMatcher(rag=rag)
response = matcher.match_file("data/job_descriptions/jd_01_ml_engineer.txt", top_k=10)
print(response.to_spec_dict())
```

## Configuration

All settings are environment-driven (prefix `PM_`, nested via `__`); see
[`.env.example`](.env.example). Key options:

| Variable | Default | Purpose |
| -------- | ------- | ------- |
| `PM_EMBEDDING__PROVIDER` | `huggingface` | `huggingface` / `openai` / `cohere` / `hashing` |
| `PM_EMBEDDING__MODEL` | `all-MiniLM-L6-v2` | Embedding model |
| `PM_VECTORSTORE__BACKEND` | `chroma` | `chroma` / `memory` |
| `PM_MATCHING__TOP_K` | `10` | Number of matches |
| `PM_MATCHING__SEMANTIC_WEIGHT` / `__KEYWORD_WEIGHT` | `0.7` / `0.3` | Hybrid fusion |

## Evaluation & performance metrics

```bash
python scripts/run_evaluation.py      # writes outputs/evaluation.json
```

On the bundled dataset (offline `hashing` backend):

| Metric | Value |
| ------ | ----- |
| Recall@10 | **1.00** (all relevant candidates surfaced) |
| Precision@10 | **0.35** (at the `#relevant / K` ceiling for every JD) |
| MRR | **0.72** |
| Mean latency | **~6 ms/query** |

The [notebook](notebooks/experimentation.ipynb) reproduces these and includes a
hybrid-weight sweep and charts. Real `sentence-transformers` embeddings improve
ranking quality further.

## Testing

```bash
# A pre-existing broken third-party pytest plugin may be present in some envs;
# disabling plugin autoload keeps the run clean.
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src pytest -q
```

21 tests cover chunking, metadata extraction, requirement parsing/filtering,
embeddings, the vector store, scoring, and the full index→match pipeline.

## Project structure

See [PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) for the complete file/package
hierarchy and responsibilities.

## License

MIT — see [`pyproject.toml`](pyproject.toml).
