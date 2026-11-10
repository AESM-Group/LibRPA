#!/usr/bin/env python3
"""Decide whether a GitHub push or pull request needs CI build jobs."""

import fnmatch
import json
import os
from pathlib import PurePosixPath
import subprocess
import sys
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


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


def github_json(path, *, environ):
    request = Request(
        f'{environ["GITHUB_API_URL"]}/repos/{environ["GITHUB_REPOSITORY"]}/{path}',
        headers={
            "Authorization": f'Bearer {environ["GITHUB_TOKEN"]}',
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def successful_oneapi_run(event, *, environ, get_json=github_json):
    """Find a successful build of this PR's exact merge commit."""
    pr = event["pull_request"]
    merge_sha = environ["GITHUB_SHA"]
    workflow_path = (
        f'{environ["GITHUB_REPOSITORY"]}/.github/workflows/ci-oneapi.yml@{merge_sha}'
    )
    page = 1
    while True:
        query = urlencode({
            "head_sha": pr["head"]["sha"], "event": "pull_request",
            "status": "success", "per_page": 100, "page": page,
        })
        runs = get_json(
            f"actions/workflows/scheduler.yml/runs?{query}", environ=environ,
        )["workflow_runs"]
        for run in runs:
            if (str(run["id"]) == environ["GITHUB_RUN_ID"]
                    or run["head_sha"] != pr["head"]["sha"]
                    or run["status"] != "completed"
                    or run["conclusion"] != "success"):
                continue
            # Fork PRs can have an empty pull_requests list. The reusable
            # workflow records the tested merge SHA and PR ref directly.
            if not any(
                workflow.get("path") == workflow_path
                and workflow.get("sha") == merge_sha
                and workflow.get("ref") == f'refs/pull/{pr["number"]}/merge'
                for workflow in run.get("referenced_workflows", [])
            ):
                continue
            job_page = 1
            while True:
                jobs = get_json(
                    f'actions/runs/{run["id"]}/jobs?filter=latest&per_page=100&page={job_page}',
                    environ=environ,
                )["jobs"]
                if any(job["name"] == "oneAPI / ONEAPI"
                       and job["status"] == "completed"
                       and job["conclusion"] == "success" for job in jobs):
                    return run["id"]
                if len(jobs) < 100:
                    break
                job_page += 1
        if len(runs) < 100:
            return None
        page += 1


def scheduled_build_required(event, *, environ, cwd=None, get_json=github_json):
    if not build_required(event, cwd=cwd):
        return False
    if "pull_request" in event and event.get("action") == "edited":
        try:
            run_id = successful_oneapi_run(event, environ=environ, get_json=get_json)
        except (URLError, TimeoutError, ValueError, KeyError) as error:
            print(f"Cannot verify an earlier oneAPI build; build again: {error}", file=sys.stderr)
        else:
            if run_id is not None:
                print(f"Reuse successful oneAPI build from run {run_id}.", file=sys.stderr)
                return False
    return True


if __name__ == "__main__":
    with open(os.environ["GITHUB_EVENT_PATH"], encoding="utf-8") as stream:
        event = json.load(stream)
    print(f"build={str(scheduled_build_required(event, environ=os.environ)).lower()}")
