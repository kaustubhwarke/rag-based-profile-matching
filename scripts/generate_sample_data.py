#!/usr/bin/env python
"""Generate a diverse, deterministic synthetic dataset.

Produces 30+ resumes and 6 job descriptions as ``.txt`` files so the pipeline
is demonstrable end-to-end with zero external data. Output is deterministic
(fixed seed) for reproducible experiments and tests.

    python scripts/generate_sample_data.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
RESUME_DIR = ROOT / "data" / "resumes"
JD_DIR = ROOT / "data" / "job_descriptions"

FIRST_NAMES = [
    "Priya", "Arjun", "Mei", "Carlos", "Aisha", "Liam", "Sofia", "Wei", "Noah",
    "Fatima", "Diego", "Hannah", "Omar", "Yuki", "Elena", "Raj", "Chloe",
    "Ahmed", "Isabella", "Kenji", "Olivia", "Mateo", "Sara", "Daniel", "Ananya",
    "Lucas", "Grace", "Ivan", "Nadia", "Tom", "Lia", "Victor",
]
LAST_NAMES = [
    "Sharma", "Patel", "Chen", "Garcia", "Khan", "Smith", "Rossi", "Wang",
    "Johnson", "Ali", "Lopez", "Muller", "Hassan", "Tanaka", "Petrov", "Nair",
    "Dubois", "Farah", "Conti", "Sato", "Brown", "Silva", "Haddad", "Cohen",
    "Iyer", "Martin", "Lee", "Novak", "Kaur", "Wilson", "Costa", "Reyes",
]

ROLE_TEMPLATES = [
    {
        "title": "Machine Learning Engineer",
        "summary": "ML engineer specialising in NLP and large language model systems, building production RAG and recommendation pipelines.",
        "skills": ["Python", "PyTorch", "TensorFlow", "NLP", "LLM", "scikit-learn", "AWS", "Docker", "Kubernetes", "SQL"],
        "degree": "M.S. in Computer Science",
        "projects": ["Built a retrieval-augmented generation pipeline serving 2M queries/day.",
                     "Fine-tuned transformer models reducing intent-classification error by 18%."],
    },
    {
        "title": "Data Scientist",
        "summary": "Data scientist focused on predictive modelling, experimentation, and statistical inference for product analytics.",
        "skills": ["Python", "Pandas", "NumPy", "scikit-learn", "SQL", "Machine Learning", "Tableau", "Spark", "R"],
        "degree": "PhD in Statistics",
        "projects": ["Designed an A/B testing framework adopted across 4 product teams.",
                     "Developed churn-prediction models improving retention by 12%."],
    },
    {
        "title": "Senior Backend Engineer",
        "summary": "Backend engineer designing scalable microservices and high-throughput APIs.",
        "skills": ["Java", "Spring", "Python", "PostgreSQL", "Kafka", "Redis", "Microservices", "REST", "Docker", "Kubernetes"],
        "degree": "B.Tech in Computer Engineering",
        "projects": ["Re-architected a monolith into 30+ microservices on Kubernetes.",
                     "Built an event-driven payments service processing $50M/month."],
    },
    {
        "title": "Frontend Engineer",
        "summary": "Frontend engineer crafting accessible, performant single-page applications.",
        "skills": ["JavaScript", "TypeScript", "React", "Vue", "Node.js", "GraphQL", "REST", "Git"],
        "degree": "B.S. in Computer Science",
        "projects": ["Led migration of a legacy app to React with a 40% load-time improvement.",
                     "Built a reusable component library used by 8 teams."],
    },
    {
        "title": "DevOps Engineer",
        "summary": "DevOps engineer automating CI/CD and infrastructure-as-code across multi-cloud environments.",
        "skills": ["AWS", "Azure", "Terraform", "Ansible", "Kubernetes", "Docker", "CI/CD", "Jenkins", "Linux", "Python"],
        "degree": "B.E. in Information Technology",
        "projects": ["Reduced deployment time from hours to minutes with GitOps pipelines.",
                     "Implemented IaC managing 200+ cloud resources via Terraform."],
    },
    {
        "title": "Data Engineer",
        "summary": "Data engineer building reliable batch and streaming data platforms.",
        "skills": ["Python", "Spark", "Hadoop", "Kafka", "Airflow", "SQL", "Snowflake", "AWS", "Data Engineering"],
        "degree": "M.Tech in Data Science",
        "projects": ["Built a streaming lakehouse ingesting 5TB/day with Spark and Kafka.",
                     "Migrated ETL workloads to Airflow improving SLA adherence to 99.5%."],
    },
    {
        "title": "Full Stack Developer",
        "summary": "Full-stack developer delivering end-to-end features from database to UI.",
        "skills": ["JavaScript", "TypeScript", "React", "Node.js", "Express", "Python", "Django", "PostgreSQL", "MongoDB", "REST"],
        "degree": "B.S. in Software Engineering",
        "projects": ["Shipped a SaaS billing module end-to-end used by 500+ customers.",
                     "Built REST and GraphQL APIs powering a React dashboard."],
    },
    {
        "title": "Cloud Solutions Architect",
        "summary": "Cloud architect designing secure, cost-efficient, highly available systems.",
        "skills": ["AWS", "GCP", "Azure", "Kubernetes", "Terraform", "Microservices", "DevOps", "Python", "Linux"],
        "degree": "M.S. in Computer Science",
        "projects": ["Designed a multi-region active-active architecture with 99.99% uptime.",
                     "Cut cloud spend 35% through rightsizing and savings plans."],
    },
    {
        "title": "Security Engineer",
        "summary": "Security engineer focused on application security, threat modelling, and cloud hardening.",
        "skills": ["Python", "Linux", "AWS", "Docker", "Kubernetes", "CI/CD", "DevOps", "REST"],
        "degree": "B.S. in Cybersecurity",
        "projects": ["Built an automated SAST/DAST pipeline catching 90% of vulns pre-release.",
                     "Led a zero-trust network segmentation rollout."],
    },
    {
        "title": "NLP Research Engineer",
        "summary": "Research engineer advancing language understanding with transformer architectures and RAG.",
        "skills": ["Python", "PyTorch", "NLP", "LLM", "Deep Learning", "TensorFlow", "Machine Learning", "Docker"],
        "degree": "PhD in Computer Science",
        "projects": ["Published 3 papers on retrieval-augmented generation.",
                     "Built a semantic search engine over 10M documents."],
    },
]

# Maps each role template title to a coarse "family" used as evaluation
# ground truth (a resume is relevant to a JD iff their families match).
ROLE_FAMILY = {
    "Machine Learning Engineer": "ml",
    "NLP Research Engineer": "ml",
    "Data Scientist": "data_science",
    "Senior Backend Engineer": "backend",
    "Frontend Engineer": "frontend",
    "DevOps Engineer": "devops",
    "Data Engineer": "data_engineering",
    "Full Stack Developer": "fullstack",
    "Cloud Solutions Architect": "cloud",
    "Security Engineer": "security",
}

COMPANIES = ["Acme Corp", "DataNova", "CloudReach", "FinEdge", "HealthAI", "RetailX",
             "Streamline", "QuantumSoft", "BrightLabs", "NimbusTech"]
CITIES = ["San Francisco", "Bangalore", "London", "Berlin", "Toronto", "Singapore",
          "Austin", "Dublin", "Sydney", "Amsterdam"]


def make_resume(rng: random.Random, idx: int, role: dict | None = None) -> tuple[str, str, str]:
    first = rng.choice(FIRST_NAMES)
    last = rng.choice(LAST_NAMES)
    name = f"{first} {last}"
    role = role or rng.choice(ROLE_TEMPLATES)
    seniority = rng.choice(["", "Senior ", "Lead ", "Staff "])
    total_years = rng.randint(2, 16)
    email = f"{first.lower()}.{last.lower()}@example.com"
    phone = f"+1 ({rng.randint(200,989)}) {rng.randint(200,989)}-{rng.randint(1000,9999)}"
    city = rng.choice(CITIES)

    # Build 2-3 experience entries with date ranges that roughly sum to tenure.
    end_year = 2026
    exp_lines = []
    remaining = total_years
    n_jobs = rng.randint(2, 3)
    for j in range(n_jobs):
        span = max(1, remaining // (n_jobs - j))
        start = end_year - span
        company = rng.choice(COMPANIES)
        title = f"{seniority if j == 0 else ''}{role['title']}".strip()
        end_label = "Present" if j == 0 else str(end_year)
        exp_lines.append(f"{title} — {company}, {city} ({start} - {end_label})")
        exp_lines.append(f"  - {rng.choice(role['projects'])}")
        exp_lines.append(f"  - Collaborated cross-functionally using {', '.join(rng.sample(role['skills'], 3))}.")
        end_year = start
        remaining -= span

    skills = role["skills"][:]
    rng.shuffle(skills)

    resume = f"""{name}
{email} | {phone} | {city}

SUMMARY
{role['summary']} {total_years}+ years of professional experience.

EXPERIENCE
{chr(10).join(exp_lines)}

SKILLS
{', '.join(skills)}

EDUCATION
{role['degree']} — University of {city} ({end_year - rng.randint(0,3)})

PROJECTS
- {role['projects'][0]}
- {role['projects'][1]}

CERTIFICATIONS
- {rng.choice(['AWS Certified Solutions Architect', 'Google Professional ML Engineer',
              'Certified Kubernetes Administrator', 'Azure Solutions Architect Expert'])}
"""
    slug = f"{idx:04d}_{first.lower()}_{last.lower()}"
    return slug, resume, ROLE_FAMILY[role["title"]]


JOB_DESCRIPTIONS = [
    ("jd_01_ml_engineer", "ml", """Title: Senior Machine Learning Engineer

We are hiring a Senior Machine Learning Engineer to build production NLP and LLM systems,
including retrieval-augmented generation (RAG) pipelines and semantic search.

Responsibilities:
- Design, train and deploy ML models with PyTorch and TensorFlow.
- Build scalable inference services on AWS using Docker and Kubernetes.

Must-have requirements:
- 5+ years Python
- Machine Learning and NLP experience
- Experience with AWS
- Strong knowledge of PyTorch or TensorFlow
"""),
    ("jd_02_backend_engineer", "backend", """Title: Senior Backend Engineer

Seeking a backend engineer to design scalable microservices and event-driven systems.

Must-have requirements:
- 4+ years Java
- Experience with Spring and Microservices
- PostgreSQL and Kafka
- Familiarity with Docker and Kubernetes
"""),
    ("jd_03_data_engineer", "data_engineering", """Title: Data Engineer

Build and operate large-scale batch and streaming data platforms.

Requirements:
- 3+ years Python
- Spark and Kafka experience required
- Airflow for orchestration
- SQL and Snowflake
- AWS cloud experience
"""),
    ("jd_04_frontend_engineer", "frontend", """Title: Frontend Engineer

Craft performant, accessible single-page applications.

Must-have requirements:
- 3+ years JavaScript
- Strong React and TypeScript skills
- Experience with GraphQL and REST APIs
"""),
    ("jd_05_devops_engineer", "devops", """Title: DevOps Engineer

Automate CI/CD and infrastructure across multi-cloud environments.

Requirements:
- 4+ years experience
- Terraform and Ansible
- Kubernetes and Docker
- AWS or Azure
- CI/CD with Jenkins
"""),
    ("jd_06_cloud_architect", "cloud", """Title: Cloud Solutions Architect

Design secure, highly available, cost-efficient cloud systems.

Must-have requirements:
- 6+ years experience
- AWS and Kubernetes
- Terraform and Microservices
- DevOps practices
"""),
]


def main() -> None:
    rng = random.Random(SEED)
    RESUME_DIR.mkdir(parents=True, exist_ok=True)
    JD_DIR.mkdir(parents=True, exist_ok=True)

    # Clean previously generated files so re-runs produce a deterministic set.
    for old in RESUME_DIR.glob("*.txt"):
        old.unlink()
    for old in JD_DIR.glob("*.txt"):
        old.unlink()

    n_resumes = 32
    resume_family: dict[str, str] = {}
    for i in range(1, n_resumes + 1):
        # Guarantee at least one resume per role family (first N), then random.
        forced = ROLE_TEMPLATES[i - 1] if i <= len(ROLE_TEMPLATES) else None
        slug, content, family = make_resume(rng, i, forced)
        (RESUME_DIR / f"{slug}.txt").write_text(content, encoding="utf-8")
        resume_family[slug] = family

    jd_family: dict[str, str] = {}
    for slug, family, content in JOB_DESCRIPTIONS:
        (JD_DIR / f"{slug}.txt").write_text(content, encoding="utf-8")
        jd_family[slug] = family

    # Ground truth for evaluation: relevant resumes per JD share its family.
    ground_truth = {
        "resume_family": resume_family,
        "jd_family": jd_family,
        "relevant_by_jd": {
            jd: sorted(r for r, fam in resume_family.items() if fam == jfam)
            for jd, jfam in jd_family.items()
        },
    }
    (ROOT / "data" / "ground_truth.json").write_text(
        json.dumps(ground_truth, indent=2), encoding="utf-8"
    )

    print(f"Generated {n_resumes} resumes -> {RESUME_DIR}")
    print(f"Generated {len(JOB_DESCRIPTIONS)} job descriptions -> {JD_DIR}")
    print(f"Wrote ground truth -> {ROOT / 'data' / 'ground_truth.json'}")


if __name__ == "__main__":
    main()
