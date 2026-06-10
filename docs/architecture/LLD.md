# Low-Level Design (LLD)

Implementation-level detail: modules, classes, signatures, algorithms, and the
data contracts that bind them. Pairs with [HLD.md](HLD.md).

## 1. Module map

```
profile_matching/
  config.py                     Settings (pydantic-settings), enums, get_settings()
  logging_config.py             configure_logging(), get_logger()
  models/        resume.py      SectionType, ResumeMetadata, ResumeChunk, Resume
                 job.py         MustHaveRequirement, JobDescription
                 match.py       MatchResult, MatchResponse
  ingestion/     loaders.py     load_document(), load_documents(), LoadedDocument
                 chunking.py    SectionAwareChunker
  extraction/    metadata.py    MetadataExtractor
  embeddings/    base.py        EmbeddingProvider (ABC)
                 providers.py   HuggingFace/OpenAI/Cohere/Hashing
                 factory.py     build_embedding_provider()
  vectorstore/   base.py        VectorStore (ABC), VectorRecord, SearchHit
                 memory_store.py InMemoryVectorStore
                 chroma_store.py ChromaVectorStore
                 factory.py     build_vector_store()
  rag/           pipeline.py    ResumeRAG          (Part A)
  matching/      requirements.py parse_job_description(), passes_must_haves()
                 hybrid_search.py HybridSearcher, CandidateAggregate, ChunkMatch
                 scoring.py     MatchScorer
                 engine.py      JobMatcher          (Part B)
  evaluation/    metrics.py     evaluate_matcher(), EvaluationReport
  cli/           resume_rag_cli.py, job_matcher_cli.py
  utils/         text.py, skills.py
```

## 2. Data contracts

### ResumeMetadata
```python
name: str; email: str|None; phone: str|None
skills: list[str]; experience_years: float
education: list[str]; highest_degree: str|None; titles: list[str]
as_store_metadata() -> dict[str, scalar]   # lists → "a | b | c"
```

### ResumeChunk
```python
chunk_id: str; resume_id: str; section: SectionType
section_heading: str|None; text: str; order: int
embedding_text() -> str                    # "[Heading]\n<text>"
```

### MatchResult  (assignment output contract — names are fixed)
```python
candidate_name: str; resume_path: str; match_score: int (0..100)
matched_skills: list[str]; relevant_excerpts: list[str]; reasoning: str
# diagnostics: semantic_score, keyword_score, skill_coverage,
#              experience_years, matched_sections
```

### MatchResponse
```python
job_description: str; top_matches: list[MatchResult]
job_title; total_candidates_considered; latency_ms
to_spec_dict() -> dict   # minimal payload, exactly the required schema
```

## 3. Algorithms

### 3.1 Section-aware chunking (`SectionAwareChunker.chunk`)
1. Split text into lines; classify each line via `_match_heading` (alias lookup,
   ≤5 words, punctuation-insensitive) to partition into `(section, heading, body)`.
2. For each section body:
   - if `len(body) ≤ max_chunk_chars` → one chunk;
   - else split on blank lines into paragraphs, greedily packing into ≤
     `max_chunk_chars` buffers, carrying `chunk_overlap_chars` of trailing context;
   - a single oversized paragraph → fixed window split (`step = max - overlap`).
3. Emit `ResumeChunk`s with monotonically increasing `order` and stable
   `chunk_id = "{resume_id}::chunk-{order}"`.

**Complexity:** O(n) in characters. **Config:** `ChunkingSettings`.

### 3.2 Metadata extraction (`MetadataExtractor.extract`)
- **Name:** first non-empty line that is not contact info, 2–4 alphabetic words.
- **Email/Phone:** regex; phone validated to ≥10 digits.
- **Skills:** taxonomy scan (`utils/skills.extract_skills`) over the Skills
  section first, then the whole document; canonicalised + de-duplicated.
- **Experience years:** `max(` explicit "N years" matches, longest inferred
  date-range tenure `)`; "Present" → pinned current year (2026) for determinism.
- **Education / highest degree:** degree-pattern scan; ranked by `_DEGREE_RANK`.

### 3.3 Skill detection (`utils/skills`)
Alias→canonical dictionary compiled into one regex with word-ish boundaries that
respect `+`/`#`/`.` (so `c++`, `c#`, `node.js` match correctly); longest aliases
matched first to avoid partial shadowing.

### 3.4 Embeddings
All providers return **L2-normalised** float32 of shape `(n, dim)`.
- **HuggingFace:** `SentenceTransformer.encode(normalize_embeddings=True)`.
- **OpenAI/Cohere:** batched API calls, then normalise (Cohere uses
  `search_document` vs `search_query` input types).
- **Hashing fallback:** hashing trick (md5 → index + sign), sublinear term
  weighting, then normalise. Deterministic, offline.

### 3.5 Vector store
- **Interface:** `upsert(records)`, `query(embedding, top_k, where)`, `count()`,
  `reset()`. Similarity is **cosine** in `[-1, 1]`.
- **InMemory:** NumPy matrix; `scores = matrix @ q`; optional exact metadata
  equality filter; exact top-K via sort.
- **Chroma:** persistent client, `hnsw:space=cosine`; squared-L2 distance `d`
  converted to cosine via `cos = 1 - d/2` for unit vectors; supports `where`.

### 3.6 JD parsing (`parse_job_description`)
- **Title:** `Title:`/`Role:` line, else first non-empty line.
- **Critical skills:** taxonomy scan over the whole JD.
- **Must-haves:** per line, if it contains a requirement cue
  (`must-have|required|minimum|at least|mandatory`) **or** a "N years" phrase,
  parse via `MustHaveRequirement.parse` (extract `min_years` and/or canonical
  `skill`); de-duplicate by `(skill, min_years)`.

### 3.7 Hybrid search (`HybridSearcher.search`)
1. **Dense recall:** embed `job.search_text`; `store.query(top_k=candidate_pool)`.
2. **Sparse:** tokenise candidate documents; build `BM25Okapi`; query =
   JD tokens **+** each critical-skill token repeated ×2 (boost).
3. **Fuse per chunk:** min-max normalise semantic (clipped ≥0) and BM25 scores;
   `hybrid = w_sem·sem + w_kw·kw`.
4. **Aggregate to resume:** group chunks by `resume_id`; resume scores =
   max over its chunks; retain `best_chunks()` and `matched_sections()`.
5. Return candidates sorted by `hybrid_score` desc.

### 3.8 Must-have filtering (`passes_must_haves`)
For each requirement: global min-years → compare `experience_years`; skill
requirement → membership in candidate's skill set (and optional scoped years).
Returns `(passes, unmet_reasons)`.

### 3.9 Scoring (`MatchScorer.score`)
```
skill_coverage   = |matched_critical_skills| / |critical_skills|
experience_score = min(1, exp / required_years)        if required_years > 0
                 = min(1, exp / 8)                      otherwise (mild prior)
composite        = 0.55·hybrid + 0.30·skill_coverage + 0.15·experience_score
match_score      = round(clip(composite, 0, 1) · 100)
```
`reasoning` is assembled from semantic band, matched skills + coverage %,
experience verdict vs. the bar, and the matched sections.

### 3.10 Matching orchestration (`JobMatcher.match`)
parse → `HybridSearcher.search` → `passes_must_haves` gate →
`MatchScorer.score` each → sort by `match_score` desc → top-K →
`MatchResponse` (with `latency_ms`, `total_candidates_considered`).

## 4. Configuration keys (selected)

| Setting | Env | Default |
|---------|-----|---------|
| embedding.provider | `PM_EMBEDDING__PROVIDER` | huggingface |
| embedding.model | `PM_EMBEDDING__MODEL` | all-MiniLM-L6-v2 |
| vectorstore.backend | `PM_VECTORSTORE__BACKEND` | chroma |
| chunking.max_chunk_chars | `PM_CHUNKING__MAX_CHUNK_CHARS` | 1200 |
| matching.top_k | `PM_MATCHING__TOP_K` | 10 |
| matching.semantic_weight / keyword_weight | `PM_MATCHING__…` | 0.7 / 0.3 |
| matching.score_* weights | `PM_MATCHING__SCORE_*` | 0.55 / 0.30 / 0.15 |

## 5. Error handling & edge cases

| Case | Behaviour |
|------|-----------|
| Unsupported file type | `ValueError` (single) / skip + warn (batch) |
| PDF/DOCX dep missing | Actionable `RuntimeError` only when such a file is read |
| Embedding/vector backend unavailable | Factory logs warning, falls back to offline |
| Empty corpus / no chunks | `query` returns `[]`; matcher returns empty `top_matches` |
| No critical skills in JD | `skill_coverage = 0`; score driven by semantic + experience |
| Negative cosine (hashing) | Clipped to 0 before fusion |

## 6. Testing strategy

| Test file | Covers |
|-----------|--------|
| `test_chunking.py` | section detection, ordering, oversized split, embed prefix |
| `test_metadata.py` | name/email/phone/skills/experience/degree, scalar metadata |
| `test_requirements.py` | requirement parsing, JD parsing, must-have filtering |
| `test_embeddings_and_store.py` | normalisation, similarity ordering, upsert/query/filter |
| `test_end_to_end.py` | index→match, exact output schema, hard-filter exclusion |

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src pytest -q`
