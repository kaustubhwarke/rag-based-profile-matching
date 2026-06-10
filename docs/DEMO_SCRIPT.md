# Demo Video Walkthrough (3–4 minutes)

A scene-by-scene script for recording the required demo video. Each scene lists
what to show, the command to run, and what to say.

> Recommended setup before recording:
> ```bash
> pip install -e ".[all]"            # or core: pip install -e .
> python scripts/generate_sample_data.py
> ```

---

## Scene 1 — Intro & architecture (0:00–0:30)
**Show:** `README.md` top + the architecture diagram in `docs/architecture/ARCHITECTURE.md`.
**Say:** "This is an enterprise RAG system that matches résumés to job
descriptions. Part A indexes résumés into a vector store with extracted
metadata; Part B runs hybrid semantic + keyword search and returns scored,
explainable matches."

## Scene 2 — Part A: indexing (0:30–1:15)
**Run:**
```bash
python resume_rag.py index --resume-dir data/resumes
```
**Say:** "We load 32 résumés, chunk them by section — Experience, Education,
Skills — extract metadata like name, skills, and years of experience, embed each
chunk, and store everything in ChromaDB."
**Then show metadata extraction on one file:**
```bash
python resume_rag.py inspect data/resumes/0001_wei_patel.txt
```

## Scene 3 — Part B: matching (1:15–2:30)
**Run:**
```bash
python job_matcher.py --jd-file data/job_descriptions/jd_01_ml_engineer.txt --top-k 5
```
**Say:** "We submit an ML Engineer job description. The engine parses critical
skills and must-have requirements — note the '5+ years Python' gate — runs hybrid
search, filters candidates who don't meet hard requirements, and scores the rest
0–100." **Point at one result's** `matched_skills`, `relevant_excerpts`, and
`reasoning`. **Then show the must-have filter and diagnostics:**
```bash
python job_matcher.py --jd-text "Backend role, 30+ years Java required" --top-k 5
python job_matcher.py --jd-file data/job_descriptions/jd_02_backend_engineer.txt --full --top-k 3
```

## Scene 4 — Metrics & notebook (2:30–3:30)
**Run:**
```bash
python scripts/run_evaluation.py
```
**Say:** "We evaluate retrieval quality against labelled ground truth.
Recall@10 is 1.0 — every relevant candidate is surfaced — with MRR around 0.7 and
about 6 ms per query." **Then open** `notebooks/experimentation.ipynb`, scroll to
the charts and the hybrid-weight sweep.

## Scene 5 — Wrap-up (3:30–4:00)
**Show:** `docs/PROJECT_STRUCTURE.md` and the passing test run:
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src pytest -q
```
**Say:** "The system is modular and production-ready: pluggable embeddings and
vector stores, graceful offline fallbacks, 21 passing tests, and full design
documentation. Thanks for watching."

---

### Recording tips
- Increase terminal font size; use a clean, wide window.
- Pre-run commands once so models/data are cached (no first-run download lag).
- For a fully offline recording, set
  `PM_EMBEDDING__PROVIDER=hashing` and `PM_VECTORSTORE__BACKEND=memory`
  (use the notebook for the index→match flow since the in-memory store is
  per-process).
