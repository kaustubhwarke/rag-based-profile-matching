"""Skill taxonomy and normalisation.

A curated dictionary of canonical skills with common aliases. Used by both
metadata extraction (to detect skills in resumes) and matching (to detect
critical skills in job descriptions and compute skill overlap).

The taxonomy is intentionally data-driven so it can be extended without code
changes; in a larger deployment this would be backed by an ontology service.
"""

from __future__ import annotations

import re

# canonical -> set of aliases (all matched case-insensitively, word-boundary)
SKILL_ALIASES: dict[str, set[str]] = {
    "Python": {"python", "py"},
    "Java": {"java"},
    "JavaScript": {"javascript", "js", "ecmascript"},
    "TypeScript": {"typescript", "ts"},
    "C++": {"c++", "cpp"},
    "C#": {"c#", "csharp"},
    "Go": {"golang", "go"},
    "Rust": {"rust"},
    "Ruby": {"ruby"},
    "PHP": {"php"},
    "Scala": {"scala"},
    "Kotlin": {"kotlin"},
    "Swift": {"swift"},
    "R": {"rlang"},
    "SQL": {"sql", "t-sql", "pl/sql"},
    "NoSQL": {"nosql"},
    "React": {"react", "react.js", "reactjs"},
    "Angular": {"angular", "angular.js", "angularjs"},
    "Vue": {"vue", "vue.js", "vuejs"},
    "Node.js": {"node", "node.js", "nodejs"},
    "Django": {"django"},
    "Flask": {"flask"},
    "FastAPI": {"fastapi"},
    "Spring": {"spring", "spring boot", "springboot"},
    ".NET": {".net", "dotnet", "asp.net"},
    "Express": {"express", "express.js"},
    "Machine Learning": {"machine learning", "ml"},
    "Deep Learning": {"deep learning", "dl"},
    "NLP": {"nlp", "natural language processing"},
    "Computer Vision": {"computer vision", "cv", "opencv"},
    "TensorFlow": {"tensorflow", "tf"},
    "PyTorch": {"pytorch", "torch"},
    "scikit-learn": {"scikit-learn", "sklearn", "scikit learn"},
    "Keras": {"keras"},
    "Pandas": {"pandas"},
    "NumPy": {"numpy"},
    "Spark": {"spark", "pyspark", "apache spark"},
    "Hadoop": {"hadoop"},
    "Kafka": {"kafka", "apache kafka"},
    "Airflow": {"airflow", "apache airflow"},
    "AWS": {"aws", "amazon web services"},
    "Azure": {"azure", "microsoft azure"},
    "GCP": {"gcp", "google cloud", "google cloud platform"},
    "Docker": {"docker"},
    "Kubernetes": {"kubernetes", "k8s"},
    "Terraform": {"terraform"},
    "Ansible": {"ansible"},
    "Jenkins": {"jenkins"},
    "CI/CD": {"ci/cd", "cicd", "continuous integration"},
    "Git": {"git", "github", "gitlab"},
    "Linux": {"linux", "unix"},
    "PostgreSQL": {"postgresql", "postgres"},
    "MySQL": {"mysql"},
    "MongoDB": {"mongodb", "mongo"},
    "Redis": {"redis"},
    "Elasticsearch": {"elasticsearch", "elastic search"},
    "Snowflake": {"snowflake"},
    "Tableau": {"tableau"},
    "Power BI": {"power bi", "powerbi"},
    "GraphQL": {"graphql"},
    "REST": {"rest", "restful", "rest api"},
    "Microservices": {"microservices", "microservice"},
    "Agile": {"agile", "scrum", "kanban"},
    "DevOps": {"devops"},
    "Data Engineering": {"data engineering", "etl", "elt"},
    "Data Science": {"data science"},
    "LLM": {"llm", "large language model", "gpt", "rag"},
}

# Pre-compile alias -> canonical and regex patterns for fast scanning.
_ALIAS_TO_CANONICAL: dict[str, str] = {}
for _canonical, _aliases in SKILL_ALIASES.items():
    for _alias in _aliases:
        _ALIAS_TO_CANONICAL[_alias] = _canonical

# Longer aliases first so "machine learning" matches before "ml" fragments.
_SORTED_ALIASES = sorted(_ALIAS_TO_CANONICAL, key=len, reverse=True)
_SKILL_PATTERN = re.compile(
    r"(?<![\w+#.])(" + "|".join(re.escape(a) for a in _SORTED_ALIASES) + r")(?![\w+#])",
    re.IGNORECASE,
)


def extract_skills(text: str) -> list[str]:
    """Return canonical skills detected in ``text`` (deduped, original order)."""

    found: list[str] = []
    seen: set[str] = set()
    for match in _SKILL_PATTERN.finditer(text):
        canonical = _ALIAS_TO_CANONICAL[match.group(1).lower()]
        if canonical not in seen:
            seen.add(canonical)
            found.append(canonical)
    return found


def normalise_skill(token: str) -> str | None:
    """Map a free-text skill token to its canonical form, if recognised."""

    return _ALIAS_TO_CANONICAL.get(token.strip().lower())
