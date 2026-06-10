"""Job-description parsing and must-have requirement filtering.

  * :func:`parse_job_description` turns raw JD text into a structured
    :class:`JobDescription`, deriving critical skills (via the skill taxonomy)
    and hard "must-have" requirements (explicit lines and "N+ years X" phrases).
  * :func:`passes_must_haves` evaluates a candidate's stored metadata against
    those requirements — the gate applied *before* ranking.
"""

from __future__ import annotations

import re

from profile_matching.models.job import JobDescription, MustHaveRequirement
from profile_matching.utils.skills import extract_skills, normalise_skill
from profile_matching.vectorstore.base import Metadata

# Lines under these headers (or bullet lines containing these cues) are treated
# as hard requirements.
_REQUIRED_CUES = re.compile(
    r"\b(must[- ]have|required|requirement|minimum|at least|mandatory)\b", re.IGNORECASE
)
_TITLE_RE = re.compile(r"^\s*(?:title|role|position)\s*[:\-]\s*(.+)$", re.IGNORECASE | re.MULTILINE)


def _decode_skill_meta(meta: Metadata) -> set[str]:
    raw = str(meta.get("skills_lower") or meta.get("skills") or "")
    return {s.strip().lower() for s in raw.split("|") if s.strip()}


def parse_job_description(
    text: str,
    job_id: str,
    title: str | None = None,
) -> JobDescription:
    """Parse raw JD text into a structured :class:`JobDescription`."""

    resolved_title = title
    if not resolved_title:
        title_match = _TITLE_RE.search(text)
        if title_match:
            resolved_title = title_match.group(1).strip()
        else:
            # Fall back to the first non-empty line.
            for line in text.split("\n"):
                if line.strip():
                    resolved_title = line.strip()
                    break

    critical_skills = extract_skills(text)

    must_have: list[MustHaveRequirement] = []
    seen: set[str] = set()
    for raw_line in text.split("\n"):
        line = raw_line.strip(" \t-•*")
        if not line:
            continue
        is_required_line = bool(_REQUIRED_CUES.search(line))
        has_years = bool(re.search(r"\d+\s*\+?\s*(?:years?|yrs?)", line, re.IGNORECASE))
        if is_required_line or has_years:
            req = MustHaveRequirement.parse(line)
            # Normalise the parsed skill against the taxonomy when possible.
            if req.skill:
                canonical = normalise_skill(req.skill) or _first_taxonomy_skill(req.skill)
                req.skill = canonical
            key = f"{req.skill}|{req.min_years}"
            if (req.skill or req.min_years) and key not in seen:
                seen.add(key)
                must_have.append(req)

    return JobDescription(
        job_id=job_id,
        title=resolved_title or "Untitled Role",
        text=text,
        must_have=must_have,
        critical_skills=critical_skills,
    )


def _first_taxonomy_skill(text: str) -> str | None:
    skills = extract_skills(text)
    return skills[0] if skills else None


def passes_must_haves(meta: Metadata, job: JobDescription) -> tuple[bool, list[str]]:
    """Return (passes, unmet_reasons) for a candidate against a JD's must-haves."""

    candidate_skills = _decode_skill_meta(meta)
    experience_years = float(meta.get("experience_years") or 0.0)
    unmet: list[str] = []

    for req in job.must_have:
        if req.min_years is not None and req.skill is None:
            # Global minimum experience.
            if experience_years + 1e-6 < req.min_years:
                unmet.append(f"requires {req.min_years:g}+ years (has {experience_years:g})")
        elif req.skill is not None:
            has_skill = req.skill.lower() in candidate_skills
            if not has_skill:
                unmet.append(f"missing required skill '{req.skill}'")
            elif req.min_years is not None and experience_years + 1e-6 < req.min_years:
                unmet.append(
                    f"requires {req.min_years:g}+ years {req.skill} (has {experience_years:g})"
                )

    return (len(unmet) == 0, unmet)
