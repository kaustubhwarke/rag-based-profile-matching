from __future__ import annotations

from profile_matching.matching.engine import JobMatcher
from profile_matching.models.match import MatchResponse

JD_ML = """Title: Senior Machine Learning Engineer
Build NLP and LLM systems with PyTorch on AWS.
Must-have requirements:
- 5+ years Python
- Machine Learning and NLP
"""


def test_pipeline_indexes_and_matches(indexed_rag):
    matcher = JobMatcher(rag=indexed_rag)
    resp = matcher.match(JD_ML, title="Senior ML Engineer", top_k=5)

    assert isinstance(resp, MatchResponse)
    assert resp.top_matches, "expected at least one match"

    top = resp.top_matches[0]
    assert top.candidate_name == "Jane Doe"
    assert 0 <= top.match_score <= 100
    assert "Python" in top.matched_skills
    assert top.relevant_excerpts
    assert top.reasoning


def test_spec_dict_has_required_keys(indexed_rag):
    matcher = JobMatcher(rag=indexed_rag)
    payload = matcher.match(JD_ML, top_k=3).to_spec_dict()

    assert set(payload) == {"job_description", "top_matches"}
    match = payload["top_matches"][0]
    assert set(match) == {
        "candidate_name",
        "resume_path",
        "match_score",
        "matched_skills",
        "relevant_excerpts",
        "reasoning",
    }


def test_must_have_filter_excludes_underqualified(indexed_rag):
    matcher = JobMatcher(rag=indexed_rag)
    strict = "Must-have requirements:\n- 30+ years Python\n"
    resp = matcher.match(strict, top_k=5, enforce_must_haves=True)
    assert resp.top_matches == []  # nobody has 30 years
