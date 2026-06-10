"""Rule-based metadata extraction.

Extracts the key fields required by the assignment — Name, Skills, Experience
Years, Education — using deterministic heuristics (regex + section parsing +
the skill taxonomy). This keeps extraction fast, free, and fully offline; the
design exposes a single :class:`MetadataExtractor` seam that could be swapped
for an LLM-based extractor in a higher-cost/higher-accuracy tier.
"""

from __future__ import annotations

import re

from profile_matching.models.resume import ResumeMetadata, SectionType
from profile_matching.utils.skills import extract_skills
from profile_matching.utils.text import normalise_whitespace

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(r"(?:(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?)?\d{3}[\s.-]?\d{4})")

# "5+ years", "7 yrs of experience", "over 3 years"
_YEARS_RE = re.compile(
    r"(\d{1,2}(?:\.\d)?)\s*\+?\s*(?:years?|yrs?)(?:\s+of)?(?:\s+(?:experience|exp))?",
    re.IGNORECASE,
)
# Date ranges like "2018 - 2023" / "2019 to Present" used to infer tenure.
_DATE_RANGE_RE = re.compile(
    r"(19|20)\d{2}\s*(?:-|–|—|to)\s*((?:19|20)\d{2}|present|current|now)",
    re.IGNORECASE,
)
_CURRENT_YEAR = 2026  # pinned for deterministic, reproducible extraction

_DEGREE_PATTERNS = [
    (r"\bph\.?\s*d\.?\b|\bdoctorate\b", "PhD"),
    (r"\bm\.?\s*tech\b|\bmaster of technology\b", "M.Tech"),
    (r"\bm\.?\s*s\.?\b|\bmaster of science\b|\bm\.?\s*sc\b", "M.S."),
    (r"\bm\.?\s*b\.?\s*a\.?\b", "MBA"),
    (r"\bmaster'?s?\b|\bm\.?\s*e\.?\b", "Master's"),
    (r"\bb\.?\s*tech\b|\bbachelor of technology\b", "B.Tech"),
    (r"\bb\.?\s*s\.?\b|\bbachelor of science\b|\bb\.?\s*sc\b", "B.S."),
    (r"\bb\.?\s*e\.?\b|\bbachelor of engineering\b", "B.E."),
    (r"\bbachelor'?s?\b|\bb\.?\s*a\.?\b", "Bachelor's"),
    (r"\bdiploma\b", "Diploma"),
]
# Ordering for "highest degree" selection (higher index == higher).
_DEGREE_RANK = ["Diploma", "Bachelor's", "B.A.", "B.S.", "B.E.", "B.Tech",
                "Master's", "MBA", "M.S.", "M.Tech", "PhD"]


class MetadataExtractor:
    """Extracts :class:`ResumeMetadata` from raw text and parsed sections."""

    def extract(
        self,
        raw_text: str,
        sections: dict[SectionType, str] | None = None,
    ) -> ResumeMetadata:
        sections = sections or {}
        return ResumeMetadata(
            name=self._extract_name(raw_text),
            email=self._first(_EMAIL_RE, raw_text),
            phone=self._extract_phone(raw_text),
            skills=self._extract_skills(raw_text, sections),
            experience_years=self._extract_experience_years(raw_text),
            education=self._extract_education(raw_text, sections),
            highest_degree=self._extract_highest_degree(raw_text),
            titles=self._extract_titles(raw_text),
        )

    # ------------------------------------------------------------------ name
    @staticmethod
    def _extract_name(text: str) -> str:
        """Heuristic: the name is usually the first non-empty, non-contact line."""

        for line in text.split("\n"):
            candidate = line.strip()
            if not candidate:
                continue
            if _EMAIL_RE.search(candidate) or _PHONE_RE.search(candidate):
                continue
            if any(ch.isdigit() for ch in candidate):
                continue
            words = candidate.split()
            if 1 < len(words) <= 4 and all(w[:1].isalpha() for w in words):
                # Strip common header labels.
                if candidate.lower() not in {"resume", "curriculum vitae", "cv"}:
                    return candidate
        return "Unknown"

    @staticmethod
    def _first(pattern: re.Pattern[str], text: str) -> str | None:
        match = pattern.search(text)
        return match.group(0) if match else None

    @staticmethod
    def _extract_phone(text: str) -> str | None:
        match = _PHONE_RE.search(text)
        if not match:
            return None
        digits = re.sub(r"\D", "", match.group(0))
        return match.group(0).strip() if len(digits) >= 10 else None

    # ---------------------------------------------------------------- skills
    @staticmethod
    def _extract_skills(text: str, sections: dict[SectionType, str]) -> list[str]:
        # Prefer the dedicated skills section, then augment from the full text.
        skills = extract_skills(sections.get(SectionType.SKILLS, ""))
        for skill in extract_skills(text):
            if skill not in skills:
                skills.append(skill)
        return skills

    # ------------------------------------------------------------ experience
    @classmethod
    def _extract_experience_years(cls, text: str) -> float:
        """Take the max of (a) explicit "N years" statements and (b) tenure
        inferred from the longest employment date range."""

        explicit = [float(m.group(1)) for m in _YEARS_RE.finditer(text)]
        explicit_max = max(explicit) if explicit else 0.0

        inferred = 0.0
        for match in _DATE_RANGE_RE.finditer(text):
            start = int(match.group(0)[:4])
            end_token = match.group(2).lower()
            end = _CURRENT_YEAR if end_token in {"present", "current", "now"} else int(end_token)
            inferred = max(inferred, float(max(0, end - start)))

        return round(max(explicit_max, inferred), 1)

    # ------------------------------------------------------------- education
    @staticmethod
    def _extract_education(text: str, sections: dict[SectionType, str]) -> list[str]:
        block = sections.get(SectionType.EDUCATION) or text
        entries: list[str] = []
        for line in block.split("\n"):
            line = line.strip()
            if not line:
                continue
            if any(re.search(p, line, re.IGNORECASE) for p, _ in _DEGREE_PATTERNS):
                normalised = normalise_whitespace(line)
                if normalised not in entries:
                    entries.append(normalised)
        return entries[:5]

    @classmethod
    def _extract_highest_degree(cls, text: str) -> str | None:
        found = [
            label
            for pattern, label in _DEGREE_PATTERNS
            if re.search(pattern, text, re.IGNORECASE)
        ]
        if not found:
            return None
        return max(found, key=lambda d: _DEGREE_RANK.index(d) if d in _DEGREE_RANK else -1)

    # ---------------------------------------------------------------- titles
    @staticmethod
    def _extract_titles(text: str) -> list[str]:
        title_kw = (
            r"(senior|lead|principal|staff|junior|sr\.?|jr\.?)?\s*"
            r"(software|data|machine learning|ml|backend|front[- ]?end|full[- ]?stack|"
            r"devops|cloud|platform|security|qa|site reliability)?\s*"
            r"(engineer|developer|scientist|architect|analyst|manager|consultant)"
        )
        titles: list[str] = []
        for match in re.finditer(title_kw, text, re.IGNORECASE):
            title = normalise_whitespace(match.group(0)).title()
            if title and title not in titles:
                titles.append(title)
        return titles[:5]
