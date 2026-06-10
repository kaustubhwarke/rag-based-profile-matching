from __future__ import annotations

from profile_matching.extraction import MetadataExtractor
from tests.conftest import SAMPLE_RESUME


def test_extracts_name():
    meta = MetadataExtractor().extract(SAMPLE_RESUME)
    assert meta.name == "Jane Doe"


def test_extracts_email_and_phone():
    meta = MetadataExtractor().extract(SAMPLE_RESUME)
    assert meta.email == "jane.doe@example.com"
    assert meta.phone is not None


def test_extracts_skills_canonicalised():
    meta = MetadataExtractor().extract(SAMPLE_RESUME)
    assert "Python" in meta.skills
    assert "PyTorch" in meta.skills
    assert "AWS" in meta.skills
    assert "Machine Learning" in meta.skills


def test_extracts_experience_years():
    meta = MetadataExtractor().extract(SAMPLE_RESUME)
    # "7+ years" explicit, and 2019-Present (7 yrs to 2026) inferred.
    assert meta.experience_years >= 7.0


def test_extracts_highest_degree():
    meta = MetadataExtractor().extract(SAMPLE_RESUME)
    assert meta.highest_degree == "M.S."
    assert any("Computer Science" in e for e in meta.education)


def test_store_metadata_is_scalar():
    meta = MetadataExtractor().extract(SAMPLE_RESUME)
    flat = meta.as_store_metadata()
    assert all(isinstance(v, (str, int, float, bool)) for v in flat.values())
    assert "python" in flat["skills_lower"]
