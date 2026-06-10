"""Match scoring and explanation.

Converts a :class:`CandidateAggregate` into the assignment's required
:class:`MatchResult`, producing:

  * a 0-100 ``match_score`` — a transparent weighted blend of
        - semantic relevance (hybrid retrieval signal),
        - skill coverage (overlap with the JD's critical skills), and
        - experience adequacy (vs. the JD's required years);
  * ``matched_skills`` — JD critical skills present in the candidate;
  * ``relevant_excerpts`` — the highest-scoring chunk snippets;
  * ``reasoning`` — a human-readable explanation citing which sections matched.
"""

from __future__ import annotations

from profile_matching.config import MatchingSettings
from profile_matching.matching.hybrid_search import CandidateAggregate
from profile_matching.models.job import JobDescription
from profile_matching.models.match import MatchResult
from profile_matching.utils.text import truncate


def _decode_list(value: object) -> list[str]:
    return [s.strip() for s in str(value or "").split("|") if s.strip()]


def _required_years(job: JobDescription) -> float:
    years = [r.min_years for r in job.must_have if r.min_years is not None]
    return max(years) if years else 0.0


class MatchScorer:
    """Computes composite scores and reasoning for ranked candidates."""

    def __init__(self, settings: MatchingSettings | None = None) -> None:
        self.settings = settings or MatchingSettings()

    def score(self, candidate: CandidateAggregate, job: JobDescription) -> MatchResult:
        meta = candidate.metadata
        candidate_skills = _decode_list(meta.get("skills"))
        candidate_skills_lower = {s.lower() for s in candidate_skills}

        # --- Skill coverage -------------------------------------------------
        critical = job.critical_skills or []
        matched_skills = [s for s in critical if s.lower() in candidate_skills_lower]
        skill_coverage = (len(matched_skills) / len(critical)) if critical else 0.0

        # --- Experience adequacy -------------------------------------------
        experience_years = float(meta.get("experience_years") or 0.0)
        required_years = _required_years(job)
        if required_years <= 0:
            experience_score = min(1.0, experience_years / 8.0)  # mild prior
        else:
            experience_score = min(1.0, experience_years / required_years)

        # --- Composite 0-100 ------------------------------------------------
        s = self.settings
        composite = (
            s.score_semantic_weight * candidate.hybrid_score
            + s.score_skill_weight * skill_coverage
            + s.score_experience_weight * experience_score
        )
        match_score = int(round(max(0.0, min(1.0, composite)) * 100))

        # --- Excerpts & reasoning ------------------------------------------
        best = candidate.best_chunks(limit=3)
        excerpts = [truncate(c.text, 280) for c in best]
        reasoning = self._build_reasoning(
            name=str(meta.get("name", "Candidate")),
            matched_skills=matched_skills,
            skill_coverage=skill_coverage,
            experience_years=experience_years,
            required_years=required_years,
            matched_sections=candidate.matched_sections(),
            semantic=candidate.semantic_score,
        )

        return MatchResult(
            candidate_name=str(meta.get("name", "Unknown")),
            resume_path=str(meta.get("source_path", "")),
            match_score=match_score,
            matched_skills=matched_skills,
            relevant_excerpts=excerpts,
            reasoning=reasoning,
            semantic_score=round(candidate.semantic_score, 4),
            keyword_score=round(candidate.keyword_score, 4),
            skill_coverage=round(skill_coverage, 4),
            experience_years=experience_years,
            matched_sections=candidate.matched_sections(),
        )

    @staticmethod
    def _build_reasoning(
        *,
        name: str,
        matched_skills: list[str],
        skill_coverage: float,
        experience_years: float,
        required_years: float,
        matched_sections: list[str],
        semantic: float,
    ) -> str:
        parts: list[str] = []

        if semantic >= 0.6:
            parts.append("Strong semantic alignment with the role")
        elif semantic >= 0.35:
            parts.append("Moderate semantic alignment with the role")
        else:
            parts.append("Limited semantic alignment with the role")

        if matched_skills:
            shown = ", ".join(matched_skills[:6])
            parts.append(
                f"matches {len(matched_skills)} key skill(s) ({shown}) "
                f"— {skill_coverage:.0%} of critical skills"
            )
        else:
            parts.append("no critical skills explicitly matched")

        if required_years > 0:
            verdict = "meets" if experience_years >= required_years else "below"
            parts.append(
                f"{experience_years:g} yrs experience ({verdict} the {required_years:g}+ yr bar)"
            )
        elif experience_years > 0:
            parts.append(f"{experience_years:g} yrs of experience")

        if matched_sections:
            parts.append("evidence in: " + ", ".join(matched_sections[:4]))

        return f"{name}: " + "; ".join(parts) + "."
