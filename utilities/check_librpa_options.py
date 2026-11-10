#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


__doc__ = """Check consistency of the LibRPA runtime option definitions.

The checker compares:

* include/librpa_options.h                 LibrpaOptions C struct
* binding/fortran/librpa_f03.f90           LibrpaOptions_c bind(c) type
* binding/fortran/librpa_f03.f90           LibrpaOptions high-level wrapper
* binding/fortran/librpa_f03.f90           sync_opts() coverage
* src/api/options.cpp                      librpa_init_options() coverage
* driver/driver.h                          DriverParams documentation
* docs/user_guide/runtime_parameters.yml   user-guide coverage

It intentionally uses lightweight parsers for the declaration patterns used by
these files, so it can run without external dependencies.
"""


@dataclass(frozen=True)
class Field:
    name: str
    kind: str
    raw_type: str
    path: Path
    line: int
    array: str = ""
    char_len: str = ""


@dataclass(frozen=True)
class SyncPair:
    f_name: str
    c_name: str
    path: Path
    line: int


@dataclass(frozen=True)
class DocField:
    name: str
    path: Path
    line: int
    doc: str = ""


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def loc(path: Path, line: int, root: Path) -> str:
    return f"{rel(path, root)}:{line}"


def norm_expr(expr: str) -> str:
    return re.sub(r"\s+", "", expr).lower()


def strip_c_comments(lines: Sequence[str]) -> List[str]:
    stripped: List[str] = []
    in_block = False

    for line in lines:
        i = 0
        out = []
        while i < len(line):
            if in_block:
                end = line.find("*/", i)
                if end == -1:
                    i = len(line)
                else:
                    in_block = False
                    i = end + 2
                continue

            if line.startswith("/*", i):
                in_block = True
                i += 2
                continue

            if line.startswith("//", i):
                break

            out.append(line[i])
            i += 1

        stripped.append("".join(out))

    return stripped


def strip_fortran_comment(line: str) -> str:
    if "!" not in line:
        return line
    return line.split("!", 1)[0]


def clean_c_doc_line(line: str) -> str:
    text = line.strip()
    for prefix in ("//!", "///", "/*!", "/**", "*/"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    if text.startswith("*"):
        text = text[1:].strip()
    return text


def iter_c_doc_lines(lines: Sequence[str], start: int) -> Tuple[List[str], int]:
    text = lines[start].lstrip()
    if text.startswith(("//!", "///")):
        out: List[str] = []
        i = start
        while i < len(lines) and lines[i].lstrip().startswith(("//!", "///")):
            out.append(clean_c_doc_line(lines[i]))
            i += 1
        return out, i

    out = [clean_c_doc_line(lines[start])]
    i = start + 1
    while i < len(lines):
        out.append(clean_c_doc_line(lines[i]))
        if "*/" in lines[i]:
            return out, i + 1
        i += 1
    return out, i


def parse_c_doc_field_name(line: str) -> Optional[str]:
    text = line.strip()
    if not text.endswith(";") or "(" in text or ")" in text or text.startswith("static "):
        return None

    declaration = text[:-1].split("=", 1)[0].strip()
    match = re.match(
        r"(?P<type>(?:std::)?[A-Za-z_][A-Za-z0-9_:<>]*"
        r"(?:\s+[A-Za-z_][A-Za-z0-9_:<>]*)*)\s+"
        r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?:\s*\[[^\]]+\])?$",
        declaration,
    )
    return match.group("name") if match else None


def parse_documented_c_struct_fields(path: Path, struct_name: str) -> List[DocField]:
    lines = path.read_text(encoding="utf-8").splitlines()
    fields: List[DocField] = []
    pending_doc: List[str] = []
    in_struct = False
    saw_open = False
    i = 0

    while i < len(lines):
        text = lines[i].strip()
        if not in_struct:
            if struct_name == "LibrpaOptions":
                if re.search(r"\btypedef\s+struct\b", text):
                    in_struct = True
                    saw_open = "{" in text
            elif re.search(rf"\bstruct\s+{re.escape(struct_name)}\b", text):
                in_struct = True
                saw_open = "{" in text
            i += 1
            continue

        if not saw_open:
            saw_open = "{" in text
            i += 1
            continue

        if struct_name == "LibrpaOptions" and re.match(r"}\s*LibrpaOptions\s*;", text):
            break
        if struct_name != "LibrpaOptions" and re.match(r"}\s*;", text):
            break

        if text.startswith(("//!", "///", "/*!", "/**")):
            doc, i = iter_c_doc_lines(lines, i)
            pending_doc.extend(doc)
            continue

        code = strip_c_comments([lines[i]])[0].strip()
        name = parse_c_doc_field_name(code)
        if name:
            fields.append(DocField(name=name, path=path, line=i + 1, doc="\n".join(pending_doc).strip()))
            pending_doc = []
        elif text and not text.startswith(("/*", "*", "//")):
            pending_doc = []
        i += 1

    return fields


def clean_fortran_doc_line(line: str) -> str:
    text = line.strip()
    for prefix in ("!>", "!!"):
        if text.startswith(prefix):
            return text[len(prefix):].strip()
    return text


def parse_fortran_type_field_docs(path: Path, type_name: str) -> List[DocField]:
    fields: List[DocField] = []
    pending_doc: List[str] = []

    for line_no, raw in extract_fortran_type(path, type_name):
        text = raw.strip()
        if text.startswith(("!>", "!!")):
            pending_doc.append(clean_fortran_doc_line(raw))
            continue

        code = strip_fortran_comment(raw).strip()
        if not code:
            continue
        if re.match(r"^\s*contains\b", code, flags=re.IGNORECASE):
            break
        if "::" not in code:
            if not text.startswith("!"):
                pending_doc = []
            continue

        left, rhs = code.split("::", 1)
        if re.match(r"^\s*(procedure|generic|private|public)\b", left, flags=re.IGNORECASE):
            pending_doc = []
            continue

        for var in split_top_level_commas(rhs):
            name, kind, _ = fortran_decl_kind(left, var)
            if name and kind != "derived":
                fields.append(DocField(name=name, path=path, line=line_no, doc="\n".join(pending_doc).strip()))
        pending_doc = []

    return fields


def parse_runtime_parameter_refs(path: Path) -> Dict[str, set[str]]:
    refs: Dict[str, set[str]] = {}
    ref_re = re.compile(r"^\s*-\s+([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\s*(?:#.*)?$")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = ref_re.match(line)
        if match:
            refs.setdefault(match.group(1), set()).add(match.group(2))
    return refs


def has_doc_text(text: str) -> bool:
    return any(line.strip() for line in text.splitlines())


def split_top_level_commas(text: str) -> List[str]:
    pieces: List[str] = []
    start = 0
    depth = 0

    for i, ch in enumerate(text):
        if ch == "(":
            depth += 1
        elif ch == ")" and depth:
            depth -= 1
        elif ch == "," and depth == 0:
            pieces.append(text[start:i].strip())
            start = i + 1

    pieces.append(text[start:].strip())
    return [p for p in pieces if p]


def fortran_statements(numbered_lines: Iterable[Tuple[int, str]]) -> Iterable[Tuple[int, str]]:
    start_line: Optional[int] = None
    statement = ""

    for line_no, raw in numbered_lines:
        line = strip_fortran_comment(raw).rstrip()
        if not line.strip():
            continue

        if start_line is None:
            start_line = line_no
            statement = line.strip()
        else:
            statement += " " + line.strip().lstrip("&").strip()

        if statement.rstrip().endswith("&"):
            statement = statement.rstrip()[:-1].strip()
            continue

        yield start_line, re.sub(r"\s+", " ", statement).strip()
        start_line = None
        statement = ""

    if start_line is not None and statement:
        yield start_line, re.sub(r"\s+", " ", statement).strip()


def parse_c_field_kind(raw_type: str, array: str) -> str:
    c_type = re.sub(r"\s+", " ", raw_type.strip())
    if c_type == "char" and array:
        return "string"
    if c_type == "double":
        return "double"
    if c_type == "long long":
        return "long_long"
    if c_type == "LibrpaSwitch":
        return "switch"
    if c_type in {
        "int",
        "LibrpaParallelRouting",
        "LibrpaTimeFreqGrid",
        "LibrpaMethod",
        "LibrpaAnalyContMethod",
        "LibrpaVerbose",
        "LibrpaKind",
    }:
        return "int"
    return f"unknown:{c_type}"


def parse_c_options(path: Path) -> List[Field]:
    lines = path.read_text(encoding="utf-8").splitlines()
    clean = strip_c_comments(lines)
    fields: List[Field] = []
    in_struct = False
    saw_open = False

    for i, line in enumerate(clean, start=1):
        text = line.strip()
        if not in_struct:
            if re.search(r"\btypedef\s+struct\b", text):
                in_struct = True
                saw_open = "{" in text
            continue

        if not saw_open:
            saw_open = "{" in text
            continue

        if re.match(r"}\s*LibrpaOptions\s*;", text):
            break

        if not text or not text.endswith(";"):
            continue

        declaration = text[:-1].strip()
        match = re.match(
            r"(?P<type>[A-Za-z_][A-Za-z0-9_\s\*]*?)\s+"
            r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
            r"(?P<array>(?:\s*\[[^\]]+\])*)$",
            declaration,
        )
        if match is None:
            continue

        raw_type = match.group("type").strip()
        name = match.group("name")
        array_text = match.group("array").strip()
        array = ""
        if array_text:
            array_match = re.match(r"\[\s*([^\]]+?)\s*\]$", array_text)
            if array_match:
                array = array_match.group(1).strip()

        fields.append(
            Field(
                name=name,
                kind=parse_c_field_kind(raw_type, array),
                raw_type=raw_type,
                path=path,
                line=i,
                array=array,
            )
        )

    return fields


def extract_fortran_type(path: Path, type_name: str) -> List[Tuple[int, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    start_re = re.compile(
        rf"^\s*type\s*(?:,\s*[^:]+)?::\s*{re.escape(type_name)}\b",
        re.IGNORECASE,
    )
    end_re = re.compile(
        rf"^\s*end\s+type(?:\s+{re.escape(type_name)})?\b",
        re.IGNORECASE,
    )

    in_type = False
    block: List[Tuple[int, str]] = []
    for i, line in enumerate(lines, start=1):
        code = strip_fortran_comment(line)
        if not in_type:
            if start_re.match(code):
                in_type = True
            continue

        if end_re.match(code):
            return block

        block.append((i, line))

    raise RuntimeError(f"could not find Fortran type {type_name} in {path}")


def fortran_decl_kind(left: str, rhs_var: str) -> Tuple[str, str, str]:
    raw_type = left.strip()
    compact = re.sub(r"\s+", "", raw_type.lower())

    name_match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)", rhs_var.strip())
    name = name_match.group(1) if name_match else ""
    array = ""
    array_match = re.match(r"[A-Za-z_][A-Za-z0-9_]*\s*\(\s*([^)]+?)\s*\)", rhs_var.strip())
    if array_match:
        array = array_match.group(1).strip()

    if compact.startswith("character"):
        len_match = re.search(r"\blen\s*=\s*([^,\)]+)", raw_type, flags=re.IGNORECASE)
        char_len = len_match.group(1).strip() if len_match else ""
        if "kind=c_char" in compact and norm_expr(char_len) == "1" and array:
            return name, "c_string", raw_type
        return name, "f_string", raw_type

    if compact.startswith("integer"):
        if "c_long_long" in compact:
            return name, "c_long_long", raw_type
        if "c_int" in compact:
            return name, "c_int", raw_type
        return name, "f_int", raw_type

    if compact.startswith("real"):
        if "c_double" in compact:
            return name, "c_double", raw_type
        if "dp" in compact:
            return name, "f_dp", raw_type
        return name, "real", raw_type

    if compact.startswith("logical"):
        return name, "logical", raw_type

    if compact.startswith("type("):
        return name, "derived", raw_type

    return name, f"unknown:{raw_type}", raw_type


def parse_fortran_type_fields(path: Path, type_name: str) -> List[Field]:
    block = extract_fortran_type(path, type_name)
    fields: List[Field] = []

    for line_no, stmt in fortran_statements(block):
        if re.match(r"^\s*contains\b", stmt, flags=re.IGNORECASE):
            break
        if "::" not in stmt:
            continue

        left, rhs = stmt.split("::", 1)
        if re.match(r"^\s*(procedure|generic|private|public)\b", left, flags=re.IGNORECASE):
            continue

        for var in split_top_level_commas(rhs):
            name, kind, raw_type = fortran_decl_kind(left, var)
            if not name or kind == "derived":
                continue

            array = ""
            array_match = re.match(r"[A-Za-z_][A-Za-z0-9_]*\s*\(\s*([^)]+?)\s*\)", var.strip())
            if array_match:
                array = array_match.group(1).strip()
            len_match = re.search(r"\blen\s*=\s*([^,\)]+)", raw_type, flags=re.IGNORECASE)
            char_len = len_match.group(1).strip() if len_match else ""

            fields.append(
                Field(
                    name=name,
                    kind=kind,
                    raw_type=raw_type,
                    path=path,
                    line=line_no,
                    array=array,
                    char_len=char_len,
                )
            )

    return fields


def extract_c_function(path: Path, function_name: str) -> List[Tuple[int, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    clean = strip_c_comments(lines)
    signature_re = re.compile(rf"\b{re.escape(function_name)}\s*\(")

    for start, line in enumerate(clean):
        if signature_re.search(line) is None:
            continue

        body: List[Tuple[int, str]] = []
        depth = 0
        started = False
        for j in range(start, len(clean)):
            text = clean[j]
            if "{" in text:
                started = True
            if started:
                body.append((j + 1, text))
                depth += text.count("{") - text.count("}")
                if depth == 0:
                    return body

    raise RuntimeError(f"could not find C/C++ function {function_name} in {path}")


def parse_cpp_initialized_fields(path: Path) -> Dict[str, List[int]]:
    body = extract_c_function(path, "librpa_init_options")
    initialized: Dict[str, List[int]] = {}
    assign_re = re.compile(r"\bopts\s*->\s*([A-Za-z_][A-Za-z0-9_]*)\s*=")
    setter_re = re.compile(r"\blibrpa_set_([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*opts\s*,")
    string_copy_re = re.compile(
        r"\b(?:std::)?strncpy\s*\(\s*opts\s*->\s*([A-Za-z_][A-Za-z0-9_]*)\s*,"
    )

    for line_no, line in body:
        for match in assign_re.finditer(line):
            initialized.setdefault(match.group(1), []).append(line_no)
        setter_match = setter_re.search(line)
        if setter_match:
            initialized.setdefault(setter_match.group(1), []).append(line_no)
        for match in string_copy_re.finditer(line):
            initialized.setdefault(match.group(1), []).append(line_no)

    return initialized


def extract_fortran_subroutine(path: Path, subroutine_name: str) -> List[Tuple[int, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    start_re = re.compile(
        rf"^\s*subroutine\s+{re.escape(subroutine_name)}\s*\(",
        re.IGNORECASE,
    )
    end_re = re.compile(
        rf"^\s*end\s+subroutine(?:\s+{re.escape(subroutine_name)})?\b",
        re.IGNORECASE,
    )

    in_subroutine = False
    body: List[Tuple[int, str]] = []
    for i, line in enumerate(lines, start=1):
        code = strip_fortran_comment(line)
        if not in_subroutine:
            if start_re.match(code):
                in_subroutine = True
            continue

        if end_re.match(code):
            return body

        body.append((i, line))

    raise RuntimeError(f"could not find Fortran subroutine {subroutine_name} in {path}")


def parse_sync_pairs(path: Path) -> List[SyncPair]:
    body = extract_fortran_subroutine(path, "sync_opts")
    pairs: List[SyncPair] = []
    sync_re = re.compile(
        r"\bcall\s+sync_opt\s*\(\s*opts%([A-Za-z_][A-Za-z0-9_]*)\s*,"
        r"\s*opts%opts_c%([A-Za-z_][A-Za-z0-9_]*)\s*,",
        re.IGNORECASE,
    )

    for line_no, stmt in fortran_statements(body):
        match = sync_re.search(stmt)
        if match:
            pairs.append(SyncPair(match.group(1), match.group(2), path, line_no))

    return pairs


def wrapper_init_calls(path: Path) -> Tuple[bool, bool, List[str]]:
    body = extract_fortran_subroutine(path, "librpa_init_options")
    statements = list(fortran_statements(body))
    calls_c_init = any(
        re.search(r"\bcall\s+librpa_init_options_c\s*\(\s*opts%opts_c\s*\)", stmt, re.IGNORECASE)
        for _, stmt in statements
    )
    calls_sync = any(
        re.search(r"\bcall\s+sync_opts\s*\(\s*opts\s*,\s*SYNC_OPTS_C2F\s*\)", stmt, re.IGNORECASE)
        for _, stmt in statements
    )
    call_text = [f"{line_no}: {stmt}" for line_no, stmt in statements if "call" in stmt.lower()]
    return calls_c_init, calls_sync, call_text


def expected_bindc_kind(c_field: Field) -> str:
    if c_field.kind == "string":
        return "c_string"
    if c_field.kind == "double":
        return "c_double"
    if c_field.kind == "long_long":
        return "c_long_long"
    if c_field.kind in {"int", "switch"}:
        return "c_int"
    return "unknown"


def expected_wrapper_kind(c_field: Field) -> str:
    if c_field.kind == "string":
        return "f_string"
    if c_field.kind == "double":
        return "f_dp"
    if c_field.kind == "long_long":
        return "c_long_long"
    if c_field.kind == "switch":
        return "logical"
    if c_field.kind == "int":
        return "f_int"
    return "unknown"


KIND_LABELS = {
    "string": "C char array",
    "double": "C double",
    "long_long": "C long long",
    "switch": "LibrpaSwitch",
    "int": "C integer/enum",
    "c_string": "Fortran C char array",
    "c_double": "real(c_double)",
    "c_long_long": "integer(c_long_long)",
    "c_int": "integer(c_int)",
    "f_string": "Fortran character",
    "f_dp": "real(dp)",
    "f_int": "Fortran integer",
    "logical": "Fortran logical",
}


def kind_label(kind: str) -> str:
    return KIND_LABELS.get(kind, kind)


def find_first_order_difference(left: Sequence[Field], right: Sequence[Field]) -> Optional[Tuple[int, Optional[Field], Optional[Field]]]:
    max_len = max(len(left), len(right))
    for i in range(max_len):
        lf = left[i] if i < len(left) else None
        rf = right[i] if i < len(right) else None
        if lf is None or rf is None or lf.name != rf.name:
            return i, lf, rf
    return None


def field_map(fields: Sequence[Field]) -> Dict[str, Field]:
    return {field.name: field for field in fields}


def add_missing_extra_issues(
    issues: List[str],
    expected: Sequence[Field],
    actual: Sequence[Field],
    label: str,
    root: Path,
) -> None:
    expected_by_name = field_map(expected)
    actual_by_name = field_map(actual)

    missing = [field for field in expected if field.name not in actual_by_name]
    extra = [field for field in actual if field.name not in expected_by_name]

    if missing:
        names = ", ".join(f"{field.name} ({loc(field.path, field.line, root)})" for field in missing)
        issues.append(f"[{label}] missing fields: {names}")
    if extra:
        names = ", ".join(f"{field.name} ({loc(field.path, field.line, root)})" for field in extra)
        issues.append(f"[{label}] extra fields: {names}")


def check_field_types(
    issues: List[str],
    c_fields: Sequence[Field],
    other_fields: Sequence[Field],
    label: str,
    expected_kind_fn,
    root: Path,
) -> None:
    other_by_name = field_map(other_fields)

    for c_field in c_fields:
        other = other_by_name.get(c_field.name)
        if other is None:
            continue

        expected = expected_kind_fn(c_field)
        if other.kind != expected:
            issues.append(
                f"[{label}] {c_field.name} has type {kind_label(other.kind)} "
                f"at {loc(other.path, other.line, root)}, expected {kind_label(expected)} "
                f"for {c_field.raw_type} at {loc(c_field.path, c_field.line, root)}"
            )
            continue

        if c_field.kind == "string":
            c_len = norm_expr(c_field.array)
            other_len = norm_expr(other.array if other.kind == "c_string" else other.char_len)
            if c_len and other_len and c_len != other_len:
                issues.append(
                    f"[{label}] {c_field.name} string length differs: "
                    f"{c_field.array} at {loc(c_field.path, c_field.line, root)} vs "
                    f"{other.array or other.char_len} at {loc(other.path, other.line, root)}"
                )


def check_bindc_layout(
    issues: List[str],
    c_fields: Sequence[Field],
    bindc_fields: Sequence[Field],
    root: Path,
) -> None:
    label = "Fortran bind(c) layout"
    add_missing_extra_issues(issues, c_fields, bindc_fields, label, root)
    diff = find_first_order_difference(c_fields, bindc_fields)
    if diff is not None:
        idx, c_field, f_field = diff
        c_desc = (
            f"{c_field.name} ({loc(c_field.path, c_field.line, root)})"
            if c_field is not None
            else "<none>"
        )
        f_desc = (
            f"{f_field.name} ({loc(f_field.path, f_field.line, root)})"
            if f_field is not None
            else "<none>"
        )
        issues.append(
            f"[{label}] field order mismatch at position {idx + 1}: "
            f"C has {c_desc}; Fortran has {f_desc}"
        )
    check_field_types(issues, c_fields, bindc_fields, label, expected_bindc_kind, root)


def check_wrapper_fields(
    issues: List[str],
    c_fields: Sequence[Field],
    wrapper_fields: Sequence[Field],
    root: Path,
    strict_order: bool,
) -> None:
    label = "Fortran high-level wrapper"
    add_missing_extra_issues(issues, c_fields, wrapper_fields, label, root)
    if strict_order:
        diff = find_first_order_difference(c_fields, wrapper_fields)
        if diff is not None:
            idx, c_field, f_field = diff
            c_desc = (
                f"{c_field.name} ({loc(c_field.path, c_field.line, root)})"
                if c_field is not None
                else "<none>"
            )
            f_desc = (
                f"{f_field.name} ({loc(f_field.path, f_field.line, root)})"
                if f_field is not None
                else "<none>"
            )
            issues.append(
                f"[{label}] field order mismatch at position {idx + 1}: "
                f"C has {c_desc}; Fortran has {f_desc}"
            )
    check_field_types(issues, c_fields, wrapper_fields, label, expected_wrapper_kind, root)


def check_cpp_initialization(
    issues: List[str],
    c_fields: Sequence[Field],
    initialized: Dict[str, List[int]],
    cpp_path: Path,
    root: Path,
) -> None:
    label = "C++ librpa_init_options"
    c_names = {field.name for field in c_fields}
    missing = [field for field in c_fields if field.name not in initialized]
    extra = sorted(name for name in initialized if name not in c_names)

    if missing:
        names = ", ".join(f"{field.name} ({loc(field.path, field.line, root)})" for field in missing)
        issues.append(f"[{label}] fields are not initialized: {names}")
    if extra:
        names = ", ".join(
            f"{name} ({', '.join(loc(cpp_path, line, root) for line in initialized[name])})"
            for name in extra
        )
        issues.append(f"[{label}] initializes unknown fields: {names}")


def check_sync_opts(
    issues: List[str],
    c_fields: Sequence[Field],
    bindc_fields: Sequence[Field],
    wrapper_fields: Sequence[Field],
    pairs: Sequence[SyncPair],
    root: Path,
) -> None:
    label = "Fortran sync_opts"
    c_names = {field.name for field in c_fields}
    bindc_names = {field.name for field in bindc_fields}
    wrapper_names = {field.name for field in wrapper_fields}

    pair_f_names = [pair.f_name for pair in pairs]
    pair_c_names = [pair.c_name for pair in pairs]
    pair_f_set = set(pair_f_names)
    pair_c_set = set(pair_c_names)

    missing_f = [field for field in c_fields if field.name not in pair_f_set]
    missing_c = [field for field in c_fields if field.name not in pair_c_set]
    if missing_f or missing_c:
        names = sorted({field.name for field in missing_f + missing_c})
        issues.append(f"[{label}] fields are not synchronized: {', '.join(names)}")

    for pair in pairs:
        if pair.f_name != pair.c_name:
            issues.append(
                f"[{label}] sync pair uses different names at {loc(pair.path, pair.line, root)}: "
                f"opts%{pair.f_name} <-> opts%opts_c%{pair.c_name}"
            )
        if pair.f_name not in wrapper_names:
            issues.append(
                f"[{label}] wrapper field opts%{pair.f_name} at {loc(pair.path, pair.line, root)} "
                "is not declared in LibrpaOptions"
            )
        if pair.c_name not in bindc_names:
            issues.append(
                f"[{label}] C field opts%opts_c%{pair.c_name} at {loc(pair.path, pair.line, root)} "
                "is not declared in LibrpaOptions_c"
            )
        if pair.f_name not in c_names or pair.c_name not in c_names:
            issues.append(
                f"[{label}] sync pair at {loc(pair.path, pair.line, root)} references a field "
                "not present in include/librpa_options.h"
            )

    duplicate_f = sorted(name for name in pair_f_set if pair_f_names.count(name) > 1)
    duplicate_c = sorted(name for name in pair_c_set if pair_c_names.count(name) > 1)
    if duplicate_f:
        issues.append(f"[{label}] duplicate wrapper sync fields: {', '.join(duplicate_f)}")
    if duplicate_c:
        issues.append(f"[{label}] duplicate C sync fields: {', '.join(duplicate_c)}")


def check_wrapper_init(issues: List[str], fortran_path: Path, root: Path) -> None:
    calls_c_init, calls_sync, call_text = wrapper_init_calls(fortran_path)
    label = "Fortran librpa_init_options"
    if not calls_c_init:
        issues.append(
            f"[{label}] does not call librpa_init_options_c(opts%opts_c). "
            f"Calls seen: {', '.join(call_text) if call_text else '<none>'}"
        )
    if not calls_sync:
        issues.append(
            f"[{label}] does not call sync_opts(opts, SYNC_OPTS_C2F). "
            f"Calls seen: {', '.join(call_text) if call_text else '<none>'}"
        )


def check_source_docs(
    warnings: List[str],
    fields: Sequence[Field],
    documented_fields: Sequence[DocField],
    label: str,
    root: Path,
) -> None:
    docs = {field.name: field for field in documented_fields}
    missing = [field for field in fields if field.name not in docs or not has_doc_text(docs[field.name].doc)]
    if missing:
        names = ", ".join(f"{field.name} ({loc(field.path, field.line, root)})" for field in missing)
        warnings.append(f"[{label}] fields lack source documentation: {names}")


def check_docfield_docs(
    warnings: List[str],
    fields: Sequence[DocField],
    label: str,
    root: Path,
) -> None:
    missing = [field for field in fields if not has_doc_text(field.doc)]
    if missing:
        names = ", ".join(f"{field.name} ({loc(field.path, field.line, root)})" for field in missing)
        warnings.append(f"[{label}] fields lack source documentation: {names}")


def check_runtime_parameter_coverage(
    warnings: List[str],
    c_fields: Sequence[Field],
    driver_fields: Sequence[DocField],
    refs: Dict[str, set[str]],
    runtime_parameters: Path,
    root: Path,
) -> None:
    label = "Runtime parameter guide"
    api_names = {field.name for field in c_fields}
    driver_names = {field.name for field in driver_fields}
    ref_api = refs.get("api", set())
    ref_driver = refs.get("driver", set())

    missing_api = [field for field in c_fields if field.name not in ref_api]
    missing_driver = [field for field in driver_fields if field.name not in ref_driver]
    extra_api = sorted(ref_api - api_names)
    extra_driver = sorted(ref_driver - driver_names)

    if missing_api:
        names = ", ".join(f"{field.name} ({loc(field.path, field.line, root)})" for field in missing_api)
        warnings.append(f"[{label}] API fields are not listed in {rel(runtime_parameters, root)}: {names}")
    if missing_driver:
        names = ", ".join(f"{field.name} ({loc(field.path, field.line, root)})" for field in missing_driver)
        warnings.append(f"[{label}] driver fields are not listed in {rel(runtime_parameters, root)}: {names}")
    if extra_api:
        warnings.append(f"[{label}] unknown API fields listed in {rel(runtime_parameters, root)}: {', '.join(extra_api)}")
    if extra_driver:
        warnings.append(f"[{label}] unknown driver fields listed in {rel(runtime_parameters, root)}: {', '.join(extra_driver)}")


def print_summary(
    c_fields: Sequence[Field],
    bindc_fields: Sequence[Field],
    wrapper_fields: Sequence[Field],
    driver_fields: Sequence[DocField],
    initialized: Dict[str, List[int]],
    pairs: Sequence[SyncPair],
    runtime_refs: Dict[str, set[str]],
) -> None:
    print("Parsed option sources:")
    print(f"  C struct fields:             {len(c_fields)}")
    print(f"  Fortran bind(c) fields:      {len(bindc_fields)}")
    print(f"  Fortran wrapper fields:      {len(wrapper_fields)}")
    print(f"  Driver fields:               {len(driver_fields)}")
    print(f"  C++ initialized fields:      {len(initialized)}")
    print(f"  Fortran sync_opts pairs:     {len(pairs)}")
    print(f"  Runtime guide field refs:    {sum(len(names) for names in runtime_refs.values())}")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    script_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=script_root,
        help="LibRPA source root (default: parent of this script)",
    )
    parser.add_argument(
        "--c-header",
        type=Path,
        default=Path("include/librpa_options.h"),
        help="Path to librpa_options.h, relative to --root unless absolute",
    )
    parser.add_argument(
        "--fortran-binding",
        type=Path,
        default=Path("binding/fortran/librpa_f03.f90"),
        help="Path to librpa_f03.f90, relative to --root unless absolute",
    )
    parser.add_argument(
        "--cpp-options",
        type=Path,
        default=Path("src/api/options.cpp"),
        help="Path to options.cpp, relative to --root unless absolute",
    )
    parser.add_argument(
        "--driver-header",
        type=Path,
        default=Path("driver/driver.h"),
        help="Path to driver.h, relative to --root unless absolute",
    )
    parser.add_argument(
        "--runtime-parameters",
        type=Path,
        default=Path("docs/user_guide/runtime_parameters.yml"),
        help="Path to runtime_parameters.yml, relative to --root unless absolute",
    )
    parser.add_argument(
        "--strict-wrapper-order",
        action="store_true",
        help="Also require LibrpaOptions wrapper field order to match the C struct",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the parsed-source summary",
    )
    return parser.parse_args(argv)


def resolve_under_root(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    c_header = resolve_under_root(root, args.c_header).resolve()
    fortran_binding = resolve_under_root(root, args.fortran_binding).resolve()
    cpp_options = resolve_under_root(root, args.cpp_options).resolve()
    driver_header = resolve_under_root(root, args.driver_header).resolve()
    runtime_parameters = resolve_under_root(root, args.runtime_parameters).resolve()

    c_fields = parse_c_options(c_header)
    bindc_fields = parse_fortran_type_fields(fortran_binding, "LibrpaOptions_c")
    wrapper_fields = parse_fortran_type_fields(fortran_binding, "LibrpaOptions")
    c_doc_fields = parse_documented_c_struct_fields(c_header, "LibrpaOptions")
    fortran_doc_fields = parse_fortran_type_field_docs(fortran_binding, "LibrpaOptions")
    driver_fields = parse_documented_c_struct_fields(driver_header, "DriverParams")
    runtime_refs = parse_runtime_parameter_refs(runtime_parameters)
    initialized = parse_cpp_initialized_fields(cpp_options)
    pairs = parse_sync_pairs(fortran_binding)

    issues: List[str] = []
    warnings: List[str] = []
    if not c_fields:
        issues.append(f"[C struct] no fields found in {rel(c_header, root)}")
    if not bindc_fields:
        issues.append(f"[Fortran bind(c) layout] no fields found in {rel(fortran_binding, root)}")
    if not wrapper_fields:
        issues.append(f"[Fortran high-level wrapper] no fields found in {rel(fortran_binding, root)}")
    if not driver_fields:
        warnings.append(f"[Driver documentation] no fields found in {rel(driver_header, root)}")

    check_bindc_layout(issues, c_fields, bindc_fields, root)
    check_wrapper_fields(issues, c_fields, wrapper_fields, root, args.strict_wrapper_order)
    check_cpp_initialization(issues, c_fields, initialized, cpp_options, root)
    check_sync_opts(issues, c_fields, bindc_fields, wrapper_fields, pairs, root)
    check_wrapper_init(issues, fortran_binding, root)
    check_source_docs(warnings, c_fields, c_doc_fields, "C API documentation", root)
    check_source_docs(warnings, wrapper_fields, fortran_doc_fields, "Fortran wrapper documentation", root)
    check_docfield_docs(warnings, driver_fields, "Driver documentation", root)
    check_runtime_parameter_coverage(warnings, c_fields, driver_fields, runtime_refs, runtime_parameters, root)

    if not args.quiet:
        print_summary(c_fields, bindc_fields, wrapper_fields, driver_fields, initialized, pairs, runtime_refs)

    if warnings:
        print(f"\nFound {len(warnings)} runtime option documentation warning(s):")
        for warning in warnings:
            print(f"  - {warning}")

    if issues:
        print(f"\nFound {len(issues)} LibrpaOptions consistency issue(s):")
        for issue in issues:
            print(f"  - {issue}")
        return 1

    suffix = f", {len(warnings)} warning(s)" if warnings else ""
    print(f"OK: LibrpaOptions definitions and initialization are consistent ({len(c_fields)} fields{suffix}).")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
