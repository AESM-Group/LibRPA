#!/usr/bin/env python3
"""Decide whether a GitHub push or pull request needs CI build jobs."""

import fnmatch
import json
import os
from pathlib import PurePosixPath
import subprocess


def needs_build(path):
    """Return whether a change to this path requires CI build jobs."""
    if path.startswith("docs/") or path.endswith(".md"):
        return False
    file = PurePosixPath(path)
    if file.parent == PurePosixPath("utilities"):
        return not any(fnmatch.fnmatchcase(file.name, pattern) for pattern in (
            "check_*.py", "test_*.py", "convert_fortran_module_to_stub.py",
        ))
    return True


def build_required(event, *, cwd=None):
    if "pull_request" in event:
        pr = event["pull_request"]
        # Match GitHub's PR filter: compare the head with its merge base.
        revision = f'{pr["base"]["sha"]}...{pr["head"]["sha"]}'
    else:
        before, after = event["before"], event["after"]
        if before == "0" * 40:
            return True
        revision = f"{before}..{after}"
    # Include both names of renames, and preserve paths containing newlines.
    paths = subprocess.check_output(
        ["git", "diff", "--name-only", "--no-renames", "-z", revision, "--"],
        cwd=cwd,
    ).decode("utf-8", errors="surrogateescape").split("\0")
    return any(needs_build(path) for path in paths if path)


if __name__ == "__main__":
    with open(os.environ["GITHUB_EVENT_PATH"], encoding="utf-8") as stream:
        event = json.load(stream)
    print(f"build={str(build_required(event)).lower()}")
