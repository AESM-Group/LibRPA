#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Set, Tuple


__doc__ = """Give the minimum risk tier of a change from its diff.

The script applies the H conditions in "Risk tiers" of
docs/develop/develop_tips.md that the paths and the lines of the diff show.
The result is a minimum. A person or an agent examines the change for the
other H conditions and can raise the tier.

Without an H condition, the tier is L if the change has only documentation and
new tests, and M otherwise. A new test is an added test file, or added lines
with no removed lines in an existing test file.

Usage:
  python utilities/check_risk_tier.py --base origin/master
  python utilities/check_risk_tier.py --base HEAD^1 --body-env PR_BODY --github

The script always exits with 0 after a successful comparison. It exits with
another value if git fails, for example when the base revision does not exist.
"""

TIERS = ("L", "M", "H")

# H conditions that the diff does not show
UNCHECKED_CONDITIONS = (
    "a change to an installed file outside `include/`",
    "a change to an input, output, or restart file format",
    "a change to a default value",
    "a change to a convention",
)

TEST_DIRS = (
    "src/test/",
    "driver/test/",
    "regression_tests/testcases/",
    "regression_tests/refs/",
    "regression_tests/verification/",
    "regression_tests/testsuite.xml",
)

MAX_ROWS = 20

_INSTALL_PAT = re.compile(r"(?<![A-Za-z0-9_])install\s*\(", re.IGNORECASE)
_HUNK_PAT = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_TIER_PAT = re.compile(r"^[ \t*_]*(?:risk[ \t]+)?tier[ \t*_]*:[ \t*_]*([LMH])\b",
                       re.IGNORECASE | re.MULTILINE)


@dataclass
class FileChange:
    path: str
    # A, M, or D. Renames are a deletion and an addition.
    status: str
    # Line numbers in the base file and in the new file
    removed: List[int] = field(default_factory=list)
    added: List[int] = field(default_factory=list)
    binary: bool = False

    @property
    def removes_lines(self) -> bool:
        """True if the change removes or changes a line of the base file."""
        if self.status == "D":
            return True
        if self.status == "A":
            return False
        return self.binary or bool(self.removed)


@dataclass
class Result:
    tier: str
    # (path, H condition)
    h_conditions: List[Tuple[str, str]]
    # Files that make the tier M when there is no H condition
    code_files: List[str]


def parse_diff(name_status: str, unified: str) -> List[FileChange]:
    """Read the output of `git diff --name-status -z` and `git diff --unified=0`."""
    fields = name_status.split("\0")
    changes = {}
    order = []
    i = 0
    while i + 1 < len(fields):
        status, path = fields[i], fields[i + 1]
        i += 2
        if not status:
            break
        # T is a change of file type, which is a change of the file
        status = "M" if status[0] == "T" else status[0]
        changes[path] = FileChange(path, status)
        order.append(path)

    current: Optional[FileChange] = None
    for line in unified.splitlines():
        if line.startswith("diff --git "):
            # Without renames the header is "a/<path> b/<path>" with the same path twice
            pair = line[len("diff --git "):]
            current = changes.get(pair[2:2 + (len(pair) - 5) // 2])
        elif current is None:
            continue
        elif line.startswith("Binary files "):
            current.binary = True
        else:
            m = _HUNK_PAT.match(line)
            if m:
                old_start, old_count = int(m.group(1)), int(m.group(2) or "1")
                new_start, new_count = int(m.group(3)), int(m.group(4) or "1")
                current.removed.extend(range(old_start, old_start + old_count))
                current.added.extend(range(new_start, new_start + new_count))
    return [changes[path] for path in order]


def install_lines(text: str) -> Set[int]:
    """Give the line numbers that an install() command in a CMake file covers."""
    covered: Set[int] = set()
    line = 1
    start = None
    depth = 0
    in_string = False
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c == "\n":
            line += 1
        elif in_string:
            if c == "\\":
                if i + 1 < n and text[i + 1] == "\n":
                    line += 1
                i += 1
            elif c == '"':
                in_string = False
        elif c == "#":
            j = text.find("\n", i)
            i = n if j < 0 else j
            continue
        elif c == '"':
            in_string = True
        elif start is None:
            m = _INSTALL_PAT.match(text, i)
            if m:
                start = line
                depth = 1
                i = m.end()
                continue
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                covered.update(range(start, line + 1))
                start = None
        i += 1
    if start is not None:
        covered.update(range(start, line + 1))
    return covered


def is_cmake(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return name == "CMakeLists.txt" or name.endswith((".cmake", ".cmake.in"))


def is_test(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return path.startswith(TEST_DIRS) or (name.startswith("test_") and name.endswith(".py"))


def is_doc(path: str) -> bool:
    return path.startswith("docs/") or path.endswith(".md")


def h_condition(change: FileChange, read_old: Callable[[str], str],
                read_new: Callable[[str], str]) -> Optional[str]:
    """Give the H condition that the change meets, or None."""
    path = change.path
    if path.startswith(".github/") or path in ("REVIEW.md", "CLAUDE.md"):
        return "change to `.github/`, `REVIEW.md`, or `CLAUDE.md`"
    if path.startswith(("thirdparty/", "src/mpi/")):
        return "change to `thirdparty/` or `src/mpi/`"
    if path.startswith("regression_tests/refs/") and change.status != "A":
        return "change to an existing reference file"
    if path == "regression_tests/testsuite.xml" and change.removes_lines:
        return "removed or changed line in `testsuite.xml`"
    if path.startswith("regression_tests/backend/comparisons/") and not is_test(path):
        return "change to a comparator"
    if path.startswith(("include/", "binding/fortran/")) and change.removes_lines:
        return "removed or changed line in `include/` or `binding/fortran/`"
    if is_cmake(path):
        old_lines = install_lines(read_old(path)) if change.status != "A" else set()
        new_lines = install_lines(read_new(path)) if change.status != "D" else set()
        if old_lines.intersection(change.removed) or new_lines.intersection(change.added):
            return "added, removed, or changed `install(` command"
    return None


def classify(changes: Sequence[FileChange], read_old: Callable[[str], str],
             read_new: Callable[[str], str]) -> Result:
    h_conditions = []
    code_files = []
    for change in changes:
        condition = h_condition(change, read_old, read_new)
        if condition is not None:
            h_conditions.append((change.path, condition))
        elif is_doc(change.path):
            pass
        elif is_test(change.path) and not change.removes_lines:
            pass
        else:
            code_files.append(change.path)
    if h_conditions:
        tier = "H"
    elif code_files:
        tier = "M"
    else:
        tier = "L"
    return Result(tier, h_conditions, code_files)


def declared_tier(body: str) -> Optional[str]:
    """Give the tier in the `Tier:` line of a pull request description."""
    # Do not read the tier from the comments of the template
    body = re.sub(r"<!--.*?-->", "", body, flags=re.DOTALL)
    m = _TIER_PAT.search(body)
    return m.group(1).upper() if m else None


def report(result: Result, declared: Optional[str], body_given: bool) -> str:
    lines = ["### Risk tier", "", f"Minimum tier from the diff: **{result.tier}**", ""]
    if result.h_conditions:
        lines += ["| File | H condition |", "| :--- | :---------- |"]
        for path, condition in result.h_conditions[:MAX_ROWS]:
            lines.append(f"| `{path}` | {condition} |")
        if len(result.h_conditions) > MAX_ROWS:
            lines.append(f"| ... | {len(result.h_conditions) - MAX_ROWS} more files |")
        lines.append("")
    elif result.code_files:
        shown = ", ".join(f"`{path}`" for path in result.code_files[:MAX_ROWS])
        more = len(result.code_files) - MAX_ROWS
        if more > 0:
            shown += f", and {more} more files"
        lines += [f"Files other than documentation and new tests: {shown}", ""]

    if body_given:
        if declared is None:
            lines += ["The description has no `Tier:` line.", ""]
        elif TIERS.index(declared) < TIERS.index(result.tier):
            lines += [f"The description gives tier {declared}, which is lower than the minimum tier.", ""]
        else:
            lines += [f"The description gives tier {declared}.", ""]

    lines.append("The diff does not show these H conditions. Examine the change for them: "
                 + "; ".join(UNCHECKED_CONDITIONS) + ".")
    return "\n".join(lines) + "\n"


def git(args: Sequence[str]) -> str:
    out = subprocess.run(["git", "-c", "core.quotepath=off", *args],
                         check=True, capture_output=True)
    return out.stdout.decode("utf-8", errors="replace")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True,
                        help="base branch or revision; the comparison starts at its merge base with the head")
    parser.add_argument("--head", default="HEAD", help="revision to compare with the base")
    parser.add_argument("--body-env", help="environment variable with the pull request description")
    parser.add_argument("--github", action="store_true",
                        help="write the report to the job summary and give a warning annotation")
    args = parser.parse_args(argv)

    for rev in (args.base, args.head):
        try:
            git(["rev-parse", "--verify", "--quiet", rev + "^{commit}"])
        except subprocess.CalledProcessError:
            print(f"error: revision {rev} does not exist", file=sys.stderr)
            return 2
    try:
        # Compare with the merge base, so that later commits on the base branch are not in the diff
        base = git(["merge-base", args.base, args.head]).strip()
        diff_args = ["diff", "--no-renames", "--no-color", "--no-ext-diff", base, args.head]
        name_status = git([*diff_args[:1], "--name-status", "-z", *diff_args[1:]])
        unified = git([*diff_args, "--unified=0"])
    except subprocess.CalledProcessError as exc:
        print(f"error: git {' '.join(exc.cmd[3:])} failed:\n{exc.stderr.decode(errors='replace')}",
              file=sys.stderr)
        return 2

    changes = parse_diff(name_status, unified)
    result = classify(changes,
                      lambda path: git(["show", f"{base}:{path}"]),
                      lambda path: git(["show", f"{args.head}:{path}"]))

    body = os.environ.get(args.body_env, "") if args.body_env else None
    declared = declared_tier(body) if body is not None else None
    text = report(result, declared, body is not None)
    print(text, end="")

    if args.github:
        summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary:
            with open(summary, "a", encoding="utf-8") as f:
                f.write(text)
        if declared is not None and TIERS.index(declared) < TIERS.index(result.tier):
            print(f"::warning title=Risk tier::The description gives tier {declared}, "
                  f"but the diff needs at least tier {result.tier}. See the job summary.")
        elif body is not None and declared is None:
            print(f"::notice title=Risk tier::The description has no Tier line. "
                  f"The diff needs at least tier {result.tier}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
