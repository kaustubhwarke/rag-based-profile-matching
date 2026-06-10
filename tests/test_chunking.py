from __future__ import annotations

from profile_matching.config import ChunkingSettings
from profile_matching.ingestion.chunking import SectionAwareChunker
from profile_matching.models.resume import SectionType

RESUME = """John Smith
john@example.com

SUMMARY
Backend engineer.

EXPERIENCE
Senior Engineer at Acme (2018 - 2023)
Built microservices.

SKILLS
Python, Java, Kafka

EDUCATION
B.Tech in Computer Science
"""


def test_sections_detected():
    chunks = SectionAwareChunker().chunk("john-smith", RESUME)
    sections = {c.section for c in chunks}
    assert SectionType.EXPERIENCE in sections
    assert SectionType.SKILLS in sections
    assert SectionType.EDUCATION in sections


def test_chunk_ids_unique_and_ordered():
    chunks = SectionAwareChunker().chunk("john-smith", RESUME)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert [c.order for c in chunks] == list(range(len(chunks)))


def test_oversized_section_is_split_with_overlap():
    big = "EXPERIENCE\n" + ("word " * 2000)
    settings = ChunkingSettings(max_chunk_chars=300, chunk_overlap_chars=50)
    chunks = SectionAwareChunker(settings).chunk("r1", big)
    exp_chunks = [c for c in chunks if c.section == SectionType.EXPERIENCE]
    assert len(exp_chunks) > 1
    assert all(len(c.text) <= 350 for c in exp_chunks)


def test_embedding_text_includes_section_prefix():
    chunks = SectionAwareChunker().chunk("john-smith", RESUME)
    skills_chunk = next(c for c in chunks if c.section == SectionType.SKILLS)
    assert skills_chunk.embedding_text().startswith("[")
