"""CLI for Part A — build / inspect the resume RAG index."""

from __future__ import annotations

import argparse
import json
import sys

from profile_matching.config import get_settings
from profile_matching.logging_config import configure_logging
from profile_matching.rag.pipeline import ResumeRAG


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="resume-rag",
        description="Index resumes into the vector store (RAG Part A).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="Process and index a directory of resumes.")
    p_index.add_argument("--resume-dir", default=None, help="Directory of resume files.")
    p_index.add_argument("--reset", action="store_true", help="Clear the index first.")

    p_inspect = sub.add_parser("inspect", help="Show extracted metadata for one resume.")
    p_inspect.add_argument("path", help="Path to a single resume file.")

    p_search = sub.add_parser("search", help="Run a raw chunk-level semantic search.")
    p_search.add_argument("query", help="Free-text query.")
    p_search.add_argument("--top-k", type=int, default=5)

    sub.add_parser("stats", help="Show vector store statistics.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    configure_logging(settings.log_level)
    rag = ResumeRAG(settings)

    if args.command == "index":
        if args.reset:
            rag.reset()
        summary = rag.index_directory(args.resume_dir)
        print(json.dumps(summary, indent=2))

    elif args.command == "inspect":
        resume = rag.process_document(args.path)
        print(json.dumps(resume.metadata.model_dump(), indent=2))
        print(f"\nChunks: {resume.chunk_count}")
        for chunk in resume.chunks:
            print(f"  [{chunk.order:02d}] {chunk.section.value:14s} {len(chunk.text):4d} chars")

    elif args.command == "search":
        hits = rag.search_chunks(args.query, top_k=args.top_k)
        for hit in hits:
            print(f"{hit.score:.3f}  {hit.metadata.get('name')}  [{hit.metadata.get('section')}]")
            print(f"        {hit.document[:120]}…")

    elif args.command == "stats":
        print(json.dumps({"total_vectors": rag.store.count(),
                          "embedder": rag.embedder.name}, indent=2))

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
