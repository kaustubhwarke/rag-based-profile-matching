"""CLI for Part B — match a job description against the indexed corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from profile_matching.config import get_settings
from profile_matching.logging_config import configure_logging
from profile_matching.matching.engine import JobMatcher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="job-matcher",
        description="Match a job description to indexed resumes (Part B).",
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--jd-file", help="Path to a job-description file.")
    src.add_argument("--jd-text", help="Inline job-description text.")

    parser.add_argument("--title", default=None, help="Optional role title.")
    parser.add_argument("--top-k", type=int, default=None, help="Number of matches (default 10).")
    parser.add_argument(
        "--no-must-haves",
        action="store_true",
        help="Do not enforce hard must-have requirements.",
    )
    parser.add_argument("--output", default=None, help="Write JSON result to this path.")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Emit the full diagnostic payload instead of the minimal spec schema.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    configure_logging(settings.log_level)

    matcher = JobMatcher(settings=settings)

    if args.jd_file:
        response = matcher.match_file(
            args.jd_file,
            title=args.title,
            top_k=args.top_k,
            enforce_must_haves=not args.no_must_haves,
        )
    else:
        response = matcher.match(
            args.jd_text,
            title=args.title,
            top_k=args.top_k,
            enforce_must_haves=not args.no_must_haves,
        )

    payload = response.model_dump() if args.full else response.to_spec_dict()
    text = json.dumps(payload, indent=2)

    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"Wrote {len(response.top_matches)} matches to {args.output}")
    else:
        print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
