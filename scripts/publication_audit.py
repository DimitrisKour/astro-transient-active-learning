#!/usr/bin/env python3
"""Conservative pre-publication scan of tracked files and Git history.

Reports *paths and rule names only*, never matched credentials.
Use a fully fetched clone for --history; shallow clones cannot audit old blobs.
This is a heuristic audit, not a security or copyright guarantee.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

RULES = {
    "private-key-marker": re.compile(rb"-----BEGIN (?:[A-Z ]*PRIVATE KEY|OPENSSH PRIVATE KEY)-----"),
    "github-token": re.compile(rb"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"),
    "cloud-aws-access-key": re.compile(rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "openai-secret": re.compile(rb"\bsk-(?:proj-)?[A-Za-z0-9_-]{25,}"),
    "jwt-literal": re.compile(rb"\beyJ[A-Za-z0-9_-]{18,}\.eyJ[A-Za-z0-9_-]{18,}\.[A-Za-z0-9_-]{8,}"),
    "password-assignment": re.compile(
        rb"(?i)\b(?:password|api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret)"
        rb"""\s*[:=]\s*['"][A-Za-z0-9_./+=-]{16,}['"]"""
    ),
}
INFORMATIONAL = {
    "hpc-local-path": re.compile(rb"""/(?:projects|work)/(?:bcrv|hdd|nvme)/[^\s'"]{8,}"""),
    "personal-home-path": re.compile(rb"(?:/home/[A-Za-z0-9._-]+/|C:\\Users\\[A-Za-z0-9._-]+\\)"),
}
TEXT_SUFFIXES = {
    ".py", ".md", ".rst", ".txt", ".toml", ".json", ".yaml", ".yml", ".sh",
    ".sbatch", ".ini", ".cfg", ".csv", ".ipynb", ".cff", ".gitignore",
    ".svg", ".xml", ".html", ".sql", ".env",
}
TEXT_NAMES = {"LICENSE", "NOTICE", "Makefile", "Dockerfile", ".gitignore"}


def git(*args: str, input: bytes | None = None) -> bytes:
    return subprocess.run(
        ["git", *args], input=input, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=True,
    ).stdout


def text_like(path: str) -> bool:
    name = Path(path).name
    return Path(name).suffix.lower() in TEXT_SUFFIXES or name in TEXT_NAMES


def findings(content: bytes, *, include_paths: bool) -> list[str]:
    patterns = dict(RULES)
    if include_paths:
        patterns.update(INFORMATIONAL)
    return [name for name, pattern in patterns.items() if pattern.search(content)]


def scan_tracked(include_paths: bool) -> tuple[int, int]:
    count = 0
    flagged = 0
    for raw in git("ls-files", "-z").split(b"\x00"):
        if not raw:
            continue
        path = raw.decode("utf-8", errors="replace")
        if not text_like(path):
            continue
        file = Path(path)
        if not file.is_file():
            continue
        count += 1
        rules = findings(file.read_bytes(), include_paths=include_paths)
        if rules:
            flagged += bool(set(rules) & RULES.keys())
            print(f"TRACKED {path}: {', '.join(rules)}")
    return count, flagged


def scan_history(include_paths: bool) -> tuple[int, int]:
    if git("rev-parse", "--is-shallow-repository").strip() == b"true":
        raise RuntimeError("Cannot audit incomplete history: fetch all branches/tags and unshallow first")
    entries = git("rev-list", "--objects", "--all").splitlines()
    # `git cat-file --batch-check` distinguishes blobs from tree/commit objects,
    # without requiring the content of uninteresting Git objects.
    shas = [entry.split(b" ", 1)[0] for entry in entries if entry]
    types = git("cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)",
                input=b"\n".join(shas) + b"\n").splitlines()
    metadata = {line.split()[0]: (line.split()[1], int(line.split()[2]))
                for line in types if len(line.split()) == 3}
    # Keep one path per object. Identical content is inspected once.
    blobs: dict[bytes, str] = {}
    for entry in entries:
        parts = entry.split(b" ", 1)
        sha = parts[0]
        if len(parts) < 2:
            continue
        path = parts[1].decode("utf-8", errors="replace")
        if metadata.get(sha, (None,))[0] == b"blob" and text_like(path):
            blobs.setdefault(sha, path)
    count = 0
    flagged = 0
    for sha, path in blobs.items():
        # Skip oversized binaries masquerading as text, but report these for review.
        size = metadata[sha][1]
        if size > 15_000_000:
            print(f"REVIEW_OVERSIZE {path}: blob {sha.decode()[:12]} ({size} bytes)")
            continue
        content = git("cat-file", "blob", sha.decode())
        count += 1
        rules = findings(content, include_paths=include_paths)
        if rules:
            flagged += bool(set(rules) & RULES.keys())
            print(f"HISTORY {path}: blob {sha.decode()[:12]}: {', '.join(rules)}")
    return count, flagged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", action="store_true", help="audit all reachable Git history")
    parser.add_argument("--include-paths", action="store_true",
                        help="also report non-secret HPC/home paths for manual review")
    args = parser.parse_args()
    try:
        count, flagged = (scan_history(args.include_paths) if args.history
                          else scan_tracked(args.include_paths))
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Audit failed: {exc}", file=sys.stderr)
        return 2
    print(f"Scanned {count} text-like {'historical blobs' if args.history else 'tracked files'}; "
          f"{flagged} with potential credential material.")
    if flagged:
        print("Potential secret indicators require manual investigation and rotation if real.")
    print("Clean results do not prove publication safety, redistribution rights, or data clearance.")
    return 1 if flagged else 0


if __name__ == "__main__":
    raise SystemExit(main())
