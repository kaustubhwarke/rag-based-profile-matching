"""Section-aware resume chunking.

Resumes are semi-structured: they contain recognisable sections (Experience,
Education, Skills, ...). Naive fixed-size chunking destroys this structure and
hurts retrieval quality. This chunker:

  1. Detects section headings via a curated, alias-rich pattern.
  2. Keeps each section intact as the primary chunk unit.
  3. Sub-splits only sections that exceed ``max_chunk_chars``, preserving
     paragraph/line boundaries and applying overlap to retain context.

The result is a list of :class:`ResumeChunk`, each tagged with its section type
— enabling section-level match reasoning downstream.
"""

from __future__ import annotations

import re

from profile_matching.config import ChunkingSettings
from profile_matching.models.resume import ResumeChunk, SectionType

# Heading aliases -> canonical section type. Matched on lines that look like
# headings (short, often upper-case, optionally followed by a colon).
_SECTION_ALIASES: dict[SectionType, tuple[str, ...]] = {
    SectionType.SUMMARY: ("summary", "objective", "profile", "about", "professional summary"),
    SectionType.EXPERIENCE: (
        "experience",
        "work experience",
        "professional experience",
        "employment",
        "employment history",
        "work history",
        "career history",
    ),
    SectionType.EDUCATION: ("education", "academic background", "academics", "qualifications"),
    SectionType.SKILLS: (
        "skills",
        "technical skills",
        "core competencies",
        "competencies",
        "technologies",
        "tech stack",
    ),
    SectionType.PROJECTS: ("projects", "key projects", "selected projects", "personal projects"),
    SectionType.CERTIFICATIONS: ("certifications", "certificates", "licenses"),
    SectionType.PUBLICATIONS: ("publications", "papers", "research"),
    SectionType.AWARDS: ("awards", "honors", "honours", "achievements"),
    SectionType.CONTACT: ("contact", "contact information", "personal details"),
}

_ALIAS_LOOKUP: dict[str, SectionType] = {
    alias: section for section, aliases in _SECTION_ALIASES.items() for alias in aliases
}
_MAX_HEADING_WORDS = 5


def _match_heading(line: str) -> tuple[SectionType, str] | None:
    """Return (section, heading) if ``line`` is a recognised section heading."""

    stripped = line.strip().rstrip(":").strip()
    if not stripped or len(stripped.split()) > _MAX_HEADING_WORDS:
        return None
    key = re.sub(r"[^a-z ]", "", stripped.lower()).strip()
    section = _ALIAS_LOOKUP.get(key)
    if section is not None:
        return section, stripped
    return None


class SectionAwareChunker:
    """Splits resume text into section-tagged, size-bounded chunks."""

    def __init__(self, settings: ChunkingSettings | None = None) -> None:
        self.settings = settings or ChunkingSettings()

    def chunk(self, resume_id: str, text: str) -> list[ResumeChunk]:
        sections = self._split_into_sections(text)
        chunks: list[ResumeChunk] = []
        order = 0
        for section, heading, body in sections:
            body = body.strip()
            if len(body) < self.settings.min_chunk_chars and not chunks:
                # Keep very short leading blocks (often contact info) anyway.
                pass
            for piece in self._split_oversized(body):
                if not piece.strip():
                    continue
                chunks.append(
                    ResumeChunk(
                        chunk_id=f"{resume_id}::chunk-{order}",
                        resume_id=resume_id,
                        section=section,
                        section_heading=heading,
                        text=piece.strip(),
                        order=order,
                    )
                )
                order += 1
        return chunks

    def _split_into_sections(
        self, text: str
    ) -> list[tuple[SectionType, str | None, str]]:
        """Partition text into (section, heading, body) tuples by headings."""

        lines = text.split("\n")
        sections: list[tuple[SectionType, str | None, list[str]]] = [
            (SectionType.CONTACT, None, [])
        ]
        for line in lines:
            heading = _match_heading(line)
            if heading is not None:
                section_type, heading_text = heading
                sections.append((section_type, heading_text, []))
            else:
                sections[-1][2].append(line)
        return [(s, h, "\n".join(b)) for s, h, b in sections]

    def _split_oversized(self, body: str) -> list[str]:
        """Split a section body that exceeds the size budget, with overlap."""

        max_chars = self.settings.max_chunk_chars
        if len(body) <= max_chars:
            return [body]

        overlap = self.settings.chunk_overlap_chars
        paragraphs = re.split(r"\n\s*\n", body)
        pieces: list[str] = []
        buffer = ""
        for para in paragraphs:
            candidate = f"{buffer}\n\n{para}".strip() if buffer else para
            if len(candidate) <= max_chars:
                buffer = candidate
                continue
            if buffer:
                pieces.append(buffer)
                tail = buffer[-overlap:] if overlap else ""
                buffer = f"{tail}\n\n{para}".strip()
            else:
                # Single paragraph larger than the budget — hard window split.
                pieces.extend(self._window_split(para, max_chars, overlap))
                buffer = ""
        if buffer:
            pieces.append(buffer)
        return pieces

    @staticmethod
    def _window_split(text: str, max_chars: int, overlap: int) -> list[str]:
        step = max(1, max_chars - overlap)
        return [text[i : i + max_chars] for i in range(0, len(text), step)]
