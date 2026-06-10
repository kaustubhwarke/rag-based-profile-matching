from __future__ import annotations

from profile_matching.matching.requirements import (
    parse_job_description,
    passes_must_haves,
)
from profile_matching.models.job import MustHaveRequirement

JD = """Title: Senior ML Engineer
Build NLP and LLM systems.
Must-have requirements:
- 5+ years Python
- Experience with AWS
"""


def test_parse_must_have_years_and_skill():
    req = MustHaveRequirement.parse("5+ years Python")
    assert req.min_years == 5.0
    assert req.skill and "python" in req.skill.lower()


def test_parse_job_description_extracts_title_and_skills():
    job = parse_job_description(JD, job_id="jd1")
    assert "ML Engineer" in job.title
    assert "Python" in job.critical_skills
    assert "AWS" in job.critical_skills
    assert any(r.min_years == 5.0 for r in job.must_have)


def test_passes_must_haves_filters_on_years():
    job = parse_job_description(JD, job_id="jd1")
    junior = {"skills_lower": "python | aws", "experience_years": 2.0}
    senior = {"skills_lower": "python | aws", "experience_years": 6.0}
    assert passes_must_haves(junior, job)[0] is False
    assert passes_must_haves(senior, job)[0] is True


def test_passes_must_haves_filters_on_missing_skill():
    job = parse_job_description("Must-have: Kubernetes required", job_id="jd2")
    cand = {"skills_lower": "python | java", "experience_years": 10.0}
    ok, unmet = passes_must_haves(cand, job)
    assert ok is False
    assert unmet
