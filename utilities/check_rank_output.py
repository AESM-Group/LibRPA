#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


__doc__ = """Check for stdout writes that can run on several MPI ranks.

Stdout from several ranks can land inside the tables that the regression tests
parse. The checker finds calls to lib_printf, printf, and cout under src/ and
driver/ that have no root check in the enclosing statements. Print text that is
the same on all ranks with lib_printf_root or printf_comm_root, and text that
differs by rank with lib_printf_coll or printf_comm_coll.

A write is also accepted if its level is LIBRPA_VERBOSE_CRITICAL, because an
error message from one rank must not wait for the other ranks.

The checker reads only the enclosing statements of a call, so it does not see a
root check in a caller. KNOWN_WRITES gives the number of such writes, and of the
remaining uncoordinated writes, that each file can have. Lower a number when you
remove a write. The check fails if a number is too low or too high.

It intentionally uses a lightweight scan of the source text, so it can run
without external dependencies.
"""


SCAN_DIRS = ("src", "driver")
SUFFIXES = {".cpp", ".h", ".hpp"}
EXCLUDED_PREFIXES = (
    "src/test/",
    "src/io/global_io.",
    # Not in driver/CMakeLists.txt
    "driver/task_qsgw.cpp",
    "driver/task_qsgwA.cpp",
    "driver/task_hf_band.cpp",
    "driver/task_scRPA.cpp",
    "driver/task_scRPA_band.cpp",
)

WRITE = re.compile(
    r"(?<![\w.>:])"
    r"(?:(?:librpa_int::)?(?:global::)?lib_printf\s*\(|(?:std::)?cout\s*<<|(?:std::)?printf\s*\()"
)
ROOT_CHECK = re.compile(
    r"is_root\s*\(\s*\)|(?<!intra_h\.)\bmyid\w*\s*==\s*0|\b0\s*==\s*[\w.>-]*myid"
)
CRITICAL = re.compile(r"\s*LIBRPA_VERBOSE_CRITICAL\b")

# Number of writes that the scan reports and that are known, for each file.
KNOWN_WRITES: Dict[str, int] = {
    "driver/inputfile.cpp": 2,
    "driver/tasks/qsgw.cpp": 1,
    "driver/tasks/sternheimer_rpa.cpp": 2,
    "driver/tasks/task_helper.cpp": 11,
    "driver/test/test_gw_occupation_output.cpp": 1,
    "src/api/global.cpp": 1,
    "src/core/chi0.cpp": 6,
    "src/core/dielecmodel.cpp": 18,
    "src/core/epsilon.cpp": 1,
    "src/core/exx.cpp": 2,
    "src/core/gw.cpp": 1,
    "src/core/meanfield.cpp": 2,
    "src/core/qsgw/task.cpp": 3,
    "src/core/sternheimer_rpa.cpp": 2,
    "src/core/timefreq.cpp": 9,
    "src/elpa/elpa_connector.cpp": 4,
    "src/gpu/la_connector.h": 2,
    "src/io/input_elsi.cpp": 1,
    "src/math/complexmatrix.cpp": 11,
    "src/math/complexmatrix.h": 4,
    "src/math/fitting.cpp": 4,
    "src/math/lapack_connector.h": 1,
    "src/math/matrix.cpp": 6,
    "src/math/matrix3.cpp": 3,
    "src/math/matrix_m.h": 10,
    "src/math/utils_matrix_m_mpi.h": 6,
    "src/math/vector3.h": 1,
    "src/mpi/base_blacs.cpp": 1,
    "src/utils/utils_cmake.cpp": 1,
}


@dataclass(frozen=True)
class Write:
    path: str
    line: int
    text: str


def strip_comments_and_strings(source: str) -> str:
    """Replace comments, literals, and preprocessor lines with spaces, and keep the newlines."""
    out = list(source)
    i, n = 0, len(source)

    def blank(begin: int, end: int) -> None:
        for k in range(begin, end):
            if out[k] != "\n":
                out[k] = " "

    while i < n:
        c = source[i]
        if source.startswith("//", i):
            end = source.find("\n", i)
            end = n if end < 0 else end
            blank(i, end)
            i = end
        elif source.startswith("/*", i):
            end = source.find("*/", i + 2)
            end = n if end < 0 else end + 2
            blank(i, end)
            i = end
        elif c in "\"'":
            end = i + 1
            while end < n and source[end] != c:
                end += 2 if source[end] == "\\" else 1
            end = min(end + 1, n)
            blank(i + 1, end - 1)
            i = end
        elif c == "#" and not source[:i].rpartition("\n")[2].strip():
            end = i
            while True:
                end = source.find("\n", end)
                if end < 0:
                    end = n
                    break
                if source[end - 1] != "\\":
                    break
                end += 1
            blank(i, end)
            i = end
        else:
            i += 1
    return "".join(out)


def find_unchecked_writes(path: str, source: str) -> List[Write]:
    """Return the writes with no root check in the statements that enclose them."""
    text = strip_comments_and_strings(source)
    source_lines = source.splitlines()
    writes = {m.start(): m for m in WRITE.finditer(text)}
    found: List[Write] = []
    # The header of each open block, such as "if (comm_h.is_root())".
    blocks: List[str] = []
    statement_start = 0
    paren_depth = 0

    for i, c in enumerate(text):
        match = writes.get(i)
        if match is not None:
            enclosing = blocks + [text[statement_start:i]]
            checked = any(ROOT_CHECK.search(header) for header in enclosing)
            critical = CRITICAL.match(source, match.end())
            if not checked and not critical:
                line = text.count("\n", 0, i) + 1
                found.append(Write(path, line, source_lines[line - 1].strip()))
        if c == "(":
            paren_depth += 1
        elif c == ")":
            paren_depth = max(0, paren_depth - 1)
        elif c == "{":
            blocks.append(text[statement_start:i])
            statement_start = i + 1
        elif c == "}":
            if blocks:
                blocks.pop()
            statement_start = i + 1
        elif c == ";" and paren_depth == 0:
            statement_start = i + 1
    return found


def scan(root: Path) -> Dict[str, List[Write]]:
    result: Dict[str, List[Write]] = {}
    for directory in SCAN_DIRS:
        for file in sorted((root / directory).rglob("*")):
            relative = file.relative_to(root).as_posix()
            if file.suffix not in SUFFIXES or relative.startswith(EXCLUDED_PREFIXES):
                continue
            writes = find_unchecked_writes(relative, file.read_text(errors="replace"))
            if writes:
                result[relative] = writes
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="root of the LibRPA source tree")
    parser.add_argument("--list", action="store_true", help="print all the writes that the scan finds")
    args = parser.parse_args()

    found = scan(args.root)
    if args.list:
        for writes in found.values():
            for write in writes:
                print(f"{write.path}:{write.line}: {write.text}")

    errors = 0
    for path in sorted(set(found) | set(KNOWN_WRITES)):
        writes = found.get(path, [])
        known = KNOWN_WRITES.get(path, 0)
        if len(writes) > known:
            errors += 1
            print(f"{path}: {len(writes)} unchecked stdout writes, {known} known:")
            for write in writes:
                print(f"  line {write.line}: {write.text}")
        elif len(writes) < known:
            errors += 1
            print(f"{path}: {len(writes)} unchecked stdout writes, {known} known. "
                  "Lower the number in KNOWN_WRITES.")
    if errors:
        print("See the description in utilities/check_rank_output.py.")
        return 1
    print("No new unchecked stdout writes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
