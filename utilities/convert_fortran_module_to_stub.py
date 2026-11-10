#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os
import re
from typing import List, Tuple, Set, Optional

__doc__ = """
Convert a Fortran module source file into a stub module. Rules:

1) Remove all iso_c_binding-related usage (use/import/bind(c)/c_* etc.).
2) Remove all non-public variables/procedures/functions (but keep private parameters
   that are referenced by public declarations so the stub can still compile).
3) Replace implementations of public procedures/functions with error_on_call(...)
   (for functions, also assign a reasonable default return value).
4) Preserve comments/doc blocks for public procedures (contiguous comment block
   immediately above the procedure definition).
5) Preserve the original order (relative order among preserved items).

Usage:
  python3 {} input.f90 output_stubs.f90
""".format(os.path.basename(__file__))

# You can change the dp replacement if needed (kept consistent with the sample stubs).
DP_REPLACEMENT = "integer, parameter, public :: dp = 8"

# Coarse detection of iso_c_binding-related tokens (prefer false-positives over false-negatives).
_C_BINDING_PAT = re.compile(
    r"\biso_c_binding\b|\bbind\s*\(\s*c\s*\)|\bc_[A-Za-z0-9_]+\b|\bc_ptr\b|\bc_null_ptr\b",
    re.IGNORECASE,
)

_C_COMMENT_PAT = re.compile(
    r"""
    \biso_c_binding\b|
    \bbind\s*\(\s*c\s*\)\b|
    \bc[_-]?(api|interface|binding)\b|
    \bfortran[- ]c\b|
    \bc\s*interoperab(le|ility)\b|
    \bc[_ ]ptr\b|
    \bc_null_ptr\b|
    \bc_int\b|\bc_double\b|\bc_char\b|
    \bffi\b|LIBRPA_FORTRAN_DP|control it through
    """,
    re.IGNORECASE | re.VERBOSE,
)

def strip_inline_comment(line: str) -> str:
    """Remove trailing inline Fortran comments starting with '!' (but keep full-line comments)."""
    if "!" not in line:
        return line
    if line.lstrip().startswith("!"):
        return line
    return line.split("!", 1)[0]


def has_c_binding_tokens(line: str) -> bool:
    return _C_BINDING_PAT.search(line) is not None


def comment_is_c_related(line: str) -> bool:
    """Return True if a comment line looks like a C-binding / interop explanatory comment."""
    if not line.lstrip().startswith("!"):
        return False
    return _C_COMMENT_PAT.search(line) is not None


def find_module_contains_index(lines: List[str]) -> Optional[int]:
    """Find the module-level CONTAINS line (ignore CONTAINS inside derived type blocks)."""
    in_type_depth = 0
    for i, ln in enumerate(lines):
        st = strip_inline_comment(ln).strip()
        if st == "":
            continue

        if in_type_depth == 0:
            # Derived type definition typically looks like: type [,...] :: TypeName
            # Variable declaration looks like: type(TypeName) :: var
            if re.match(r"^\s*type\b", st, flags=re.IGNORECASE) and not re.match(r"^\s*type\s*\(", st, flags=re.IGNORECASE):
                if not re.match(r"^\s*end\s*type\b", st, flags=re.IGNORECASE):
                    in_type_depth = 1
                    continue
        else:
            if re.match(r"^\s*end\s*type\b", st, flags=re.IGNORECASE):
                in_type_depth = 0
            continue

        if st.lower() == "contains":
            return i
    return None


def _gather_continued_statement(lines: List[str], start: int) -> Tuple[str, int]:
    """Gather a Fortran statement that may use ampersand continuation; return (merged_stmt, next_index)."""
    parts: List[str] = []
    i = start
    while i < len(lines):
        raw = lines[i]
        txt = strip_inline_comment(raw).rstrip("\n")
        parts.append(txt)
        cur = txt.rstrip()
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        continued = cur.endswith("&") or nxt.lstrip().startswith("&")
        i += 1
        if not continued:
            break
    stmt = " ".join(p.replace("&", " ") for p in parts)
    stmt = re.sub(r"\s+", " ", stmt).strip()
    return stmt, i


def parse_public_names(pre_lines: List[str]) -> Set[str]:
    """Collect explicitly-public names from the module header (public :: ... and , public :: ...)."""
    public: Set[str] = set()

    i = 0
    while i < len(pre_lines):
        ln = pre_lines[i]
        stripped = strip_inline_comment(ln).strip()

        if re.match(r"^public\s*::", stripped, flags=re.IGNORECASE):
            stmt, i2 = _gather_continued_statement(pre_lines, i)
            rhs = re.split(r"public\s*::", stmt, flags=re.IGNORECASE, maxsplit=1)[1]
            for tok in rhs.split(","):
                name = tok.strip()
                if not name:
                    continue
                name = re.split(r"\s+", name, maxsplit=1)[0]
                name = re.sub(r"[^A-Za-z0-9_]", "", name)
                if name:
                    public.add(name)
            i = i2
            continue

        m = re.search(r",\s*public\s*::\s*([^!]+)$", stripped, flags=re.IGNORECASE)
        if m:
            tail = m.group(1)
            for tok in tail.split(","):
                t = tok.strip()
                if not t:
                    continue
                name = t.split("=", 1)[0].strip()
                name = name.split("(", 1)[0].strip()
                name = re.sub(r"[^A-Za-z0-9_]", "", name)
                if name:
                    public.add(name)

        i += 1

    return public


def parse_type_bound_targets(pre_lines: List[str]) -> Set[str]:
    """Collect type-bound binding targets: procedure :: a => target"""
    targets: Set[str] = set()
    for ln in pre_lines:
        s = strip_inline_comment(ln)
        m = re.search(r"=>\s*([A-Za-z_][A-Za-z0-9_]*)", s)
        if m:
            targets.add(m.group(1))
    return targets


def _is_type_start(line: str) -> Optional[str]:
    """If this line starts a derived type definition, return the type name; otherwise None."""
    s = strip_inline_comment(line).strip()
    if not re.match(r"^type\b", s, flags=re.IGNORECASE):
        return None
    if re.match(r"^type\s*\(", s, flags=re.IGNORECASE):
        return None  # type(...) :: var is a variable declaration

    m = re.search(r"::\s*([A-Za-z_][A-Za-z0-9_]*)\b", s)
    if m:
        return m.group(1)

    m2 = re.match(r"^type\s+([A-Za-z_][A-Za-z0-9_]*)\b", s, flags=re.IGNORECASE)
    if m2:
        return m2.group(1)
    return None


def build_header(pre_lines: List[str], public_names: Set[str], public_types: Set[str]) -> List[str]:
    """Build the stub module header: keep public declarations + public types + required private parameters.
    Additional rules:
      -   - Drop all comment blocks that belong to discarded (non-public) items (only flush comments before a kept code line).
      -   - Remove interop/C-binding explanatory comments (comment_is_c_related).
    """
    out: List[str] = []

    # --- Dependency handling: keep private parameters referenced by public declarations (so the stub still compiles). ---
    def _declared_names_from_param_line(line: str) -> List[str]:
        s = strip_inline_comment(line)
        if "::" not in s:
            return []
        rhs = s.split("::", 1)[1]
        names: List[str] = []
        for part in rhs.split(","):
            p = part.strip()
            if not p:
                continue
            p = p.split("=", 1)[0].strip()
            p = p.split("(", 1)[0].strip()
            p = re.sub(r"[^A-Za-z0-9_]", "", p)
            if p:
                names.append(p)
        return names

    cand_param: List[Tuple[int, str, List[str]]] = []
    for idx, ln in enumerate(pre_lines):
        s = strip_inline_comment(ln)
        st = s.strip()
        if not st or ln.lstrip().startswith("!"):
            continue
        if has_c_binding_tokens(s):
            continue
        if re.match(r"^type\b", st, flags=re.IGNORECASE):
            continue
        if re.search(r"\bparameter\b", st, flags=re.IGNORECASE) and "::" in st and not re.search(r",\s*public\s*::", st, flags=re.IGNORECASE):
            names = _declared_names_from_param_line(ln)
            if names:
                cand_param.append((idx, ln, names))

    _ident_re = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")

    def _idents_in_line(line: str) -> Set[str]:
        s = strip_inline_comment(line)
        if line.lstrip().startswith("!"):
            return set()
        return {m.group(0) for m in _ident_re.finditer(s)}

    used_idents: Set[str] = set()
    in_public_type = False
    for ln in pre_lines:
        s = strip_inline_comment(ln)
        st = s.strip()
        low = st.lower()

        if not in_public_type:
            tname = _is_type_start(ln)
            if tname and tname in public_types:
                in_public_type = True
                used_idents |= _idents_in_line(ln)
                continue
        else:
            used_idents |= _idents_in_line(ln)
            if re.match(r"^\s*end\s*type\b", st, flags=re.IGNORECASE):
                in_public_type = False
            continue

        if (
            (re.match(r"^module\s+[A-Za-z_]", low) and not re.match(r"^module\s+procedure\b", low))
            or low in {"implicit none", "private"}
        ):
            used_idents |= _idents_in_line(ln)
            continue

        if re.match(r"^public\s*::", st, flags=re.IGNORECASE):
            used_idents |= _idents_in_line(ln)
            continue

        if re.search(r",\s*public\s*::", st, flags=re.IGNORECASE):
            used_idents |= _idents_in_line(ln)
            continue

    needed_private_param_names: Set[str] = set()
    for _, _, names in cand_param:
        if any(n in used_idents for n in names):
            needed_private_param_names.update(names)

    # --- Pending comment buffer: only flush right before a kept code line. ---
    pending: List[str] = []

    def flush_pending_if_any() -> None:
        nonlocal pending
        if not pending:
            return
        # Filter out C-interop explanatory comments
        kept = [ln for ln in pending if not comment_is_c_related(ln)]
        # If the block becomes empty after filtering, emit nothing
        out.extend(kept)
        pending = []

    def drop_pending() -> None:
        nonlocal pending
        pending = []

    # --- Emit the header, preserving the original order. ---
    i = 0
    while i < len(pre_lines):
        ln = pre_lines[i]
        s_nocom = strip_inline_comment(ln)
        s = s_nocom.strip()
        low = s.lower()

        # First collect comments/blank lines into pending (do not emit immediately).
        if ln.lstrip().startswith("!"):
            pending.append(ln)
            i += 1
            continue
        if s == "":
            pending.append(ln)
            i += 1
            continue

        # Drop use iso_c_binding
        if re.match(r"^use\b", s, flags=re.IGNORECASE) and "iso_c_binding" in s.lower():
            # This line is discarded, so its preceding pending comments should be dropped as well.
            drop_pending()
            i += 1
            continue

        # Keep: module header line (exclude module procedure), implicit none, private
        if (
            (re.match(r"^module\s+[A-Za-z_]", low) and not re.match(r"^module\s+procedure\b", low))
            or low in {"implicit none", "private"}
        ):
            flush_pending_if_any()
            out.append(ln)
            i += 1
            continue

        # Keep: public :: ...
        if re.match(r"^public\s*::", s, flags=re.IGNORECASE):
            flush_pending_if_any()
            _, i2 = _gather_continued_statement(pre_lines, i)
            out.extend(pre_lines[i:i2])
            i = i2
            continue

        # Rewrite dp (keep the line but replace its content)
        if (
            re.search(r"\bdp\b", s, flags=re.IGNORECASE)
            and re.search(r"\bparameter\b", s, flags=re.IGNORECASE)
            and re.search(r",\s*public\s*::", s, flags=re.IGNORECASE)
        ):
            flush_pending_if_any()
            indent = re.match(r"^(\s*)", ln).group(1)
            out.append(indent + DP_REPLACEMENT + ("\n" if not DP_REPLACEMENT.endswith("\n") else ""))
            i += 1
            continue

        # Keep: inline public declarations
        if re.search(r",\s*public\s*::", s, flags=re.IGNORECASE):
            if not has_c_binding_tokens(s):
                flush_pending_if_any()
                out.append(ln)
            else:
                # Discard C-related public declarations, and also drop their pending comments.
                drop_pending()
            i += 1
            continue

        # Keep: required private parameters
        if (
            re.search(r"\bparameter\b", s, flags=re.IGNORECASE)
            and "::" in s
            and not re.search(r",\s*public\s*::", s, flags=re.IGNORECASE)
            and not has_c_binding_tokens(s)
        ):
            decl_names = _declared_names_from_param_line(ln)
            if any(n in needed_private_param_names for n in decl_names):
                flush_pending_if_any()
                out.append(ln)
            else:
                drop_pending()
            i += 1
            continue

        # Keep: public derived type blocks
        tname = _is_type_start(ln)
        if tname and tname in public_types:
            flush_pending_if_any()
            block: List[str] = [ln]
            i += 1
            while i < len(pre_lines):
                block.append(pre_lines[i])
                if re.match(r"^\s*end\s*type\b", strip_inline_comment(pre_lines[i]).strip(), flags=re.IGNORECASE):
                    i += 1
                    break
                i += 1

            # Inside the type block:
            # - filter C-related lines
            # - filter C-interop explanatory comments
            # - drop leading blank line(s) right after the 'type :: Name' header (cosmetic cleanup)
            saw_type_header = False
            saw_any_emitted_after_header = False
            for b in block:
                if not saw_type_header:
                    out.append(b)
                    saw_type_header = True
                    continue

                # Skip blank lines immediately after the type header.
                if not saw_any_emitted_after_header and b.strip() == "":
                    continue

                if b.lstrip().startswith("!"):
                    # Keep comments only if they are not filtered.
                    if not comment_is_c_related(b):
                        out.append(b)
                        saw_any_emitted_after_header = True
                    continue

                bnc = strip_inline_comment(b)

                # Drop C-binding related declarations inside the type.
                if has_c_binding_tokens(bnc):
                    bs = bnc.strip()
                    # Keep only the type header (non 'type(...)' variable declarations) and 'end type'.
                    if (
                        (re.match(r"^type\b", bs, flags=re.IGNORECASE) and not re.match(r"^type\s*\(", bs, flags=re.IGNORECASE))
                        or re.match(r"^end\s*type\b", bs, flags=re.IGNORECASE)
                    ):
                        out.append(b)
                        saw_any_emitted_after_header = True
                    continue

                # Drop private components inside a public type.
                if re.search(r"\bprivate\b", bnc, flags=re.IGNORECASE) and "::" in bnc:
                    continue

                out.append(b)
                saw_any_emitted_after_header = True

            continue


        # Everything else (interfaces, private helpers, C-binding layers, etc.) is dropped; drop its pending comments too.
        drop_pending()
        i += 1

    # End of header: if only pending remains (meaning it belongs to no kept item), drop it.
    drop_pending()

    if out and out[-1].strip() != "":
        out.append("\n")
    return out


_PROC_START_RE = re.compile(r"\b(subroutine|function)\b", re.IGNORECASE)


def is_proc_start(line: str) -> bool:
    s = strip_inline_comment(line)
    st = s.strip().lower()
    if st.startswith("end "):
        return False
    if line.lstrip().startswith("!"):
        return False
    return (
        _PROC_START_RE.search(s) is not None
        and re.match(r"^\s*[A-Za-z0-9_\(\)\*\s,=:\+\-\.]*\b(subroutine|function)\b", s, re.IGNORECASE) is not None
    )


def proc_kind_and_name(line: str) -> Optional[Tuple[str, str]]:
    s = strip_inline_comment(line)
    m = re.search(r"\bsubroutine\s+([A-Za-z_][A-Za-z0-9_]*)", s, flags=re.IGNORECASE)
    if m:
        return "subroutine", m.group(1)
    m = re.search(r"\bfunction\s+([A-Za-z_][A-Za-z0-9_]*)", s, flags=re.IGNORECASE)
    if m:
        return "function", m.group(1)
    return None


def gather_proc_block(lines: List[str], start: int) -> Tuple[List[str], int, str, str]:
    kind, name = proc_kind_and_name(lines[start])  # type: ignore[misc]
    block: List[str] = []
    i = start
    while i < len(lines):
        block.append(lines[i])
        end = strip_inline_comment(lines[i]).strip().lower()
        if end.startswith("end subroutine") or end.startswith("end function"):
            i += 1
            break
        i += 1
    return block, i, kind, name



def gather_leading_doc_comments(all_lines: List[str], proc_start_idx: int) -> List[str]:
    """Collect the contiguous comment block immediately above a procedure definition (stop at blank/non-comment).
    Also filter out comments that look like C-binding/iso_c_binding/bind(c) explanations.
    """
    docs: List[str] = []
    j = proc_start_idx - 1
    while j >= 0:
        ln = all_lines[j]
        if ln.strip() == "":
            break
        if ln.lstrip().startswith("!"):
            docs.append(ln)
            j -= 1
            continue
        break
    docs.reverse()

    # Filter out C-related comment lines
    docs = [ln for ln in docs if not comment_is_c_related(ln)]
    return docs


def extract_header_lines(block: List[str]) -> Tuple[List[str], int]:
    """Extract the procedure header (handling ampersand continuation)."""
    header: List[str] = []
    i = 0
    while i < len(block):
        header.append(block[i])
        cur = strip_inline_comment(block[i]).rstrip()
        nxt = strip_inline_comment(block[i + 1]).lstrip() if i + 1 < len(block) else ""
        continued = cur.endswith("&") or nxt.startswith("&")
        i += 1
        if not continued:
            break
    return header, i


def parse_dummy_args(header_lines: List[str], name: str) -> List[str]:
    header_join = " ".join(strip_inline_comment(h).replace("&", " ") for h in header_lines)
    header_join = re.sub(r"\s+", " ", header_join)

    m = re.search(rf"\b{name}\s*\(", header_join, flags=re.IGNORECASE)
    if not m:
        return []
    start = m.end()
    depth = 1
    i = start
    while i < len(header_join) and depth > 0:
        if header_join[i] == "(":
            depth += 1
        elif header_join[i] == ")":
            depth -= 1
        i += 1
    arg_text = header_join[start : i - 1].strip() if depth == 0 else ""
    if not arg_text:
        return []
    args = [a.strip() for a in arg_text.split(",")]

    cleaned: List[str] = []
    for a in args:
        if not a:
            continue
        a = a.split("=", 1)[0].strip()
        a = re.sub(r"[^A-Za-z0-9_]", "", a)
        if a:
            cleaned.append(a)
    return cleaned


def function_result_var(header_lines: List[str], func_name: str) -> str:
    header_join = " ".join(strip_inline_comment(h) for h in header_lines)
    m = re.search(r"\bresult\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)", header_join, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    return func_name


def function_return_category(header_lines: List[str]) -> str:
    """Heuristically determine function return category: integer/real/complex/logical/character/unknown."""
    first = strip_inline_comment(header_lines[0]).strip()
    m = re.search(r"\bfunction\b", first, flags=re.IGNORECASE)
    prefix = first[: m.start()].strip().lower() if m else ""

    prefix = re.sub(r"\b(module|recursive|pure|elemental)\b", " ", prefix)
    prefix = re.sub(r"\s+", " ", prefix).strip()

    if prefix.startswith("integer"):
        return "integer"
    if prefix.startswith("real"):
        return "real"
    if prefix.startswith("complex"):
        return "complex"
    if prefix.startswith("logical"):
        return "logical"
    if prefix.startswith("character"):
        return "character"
    return "unknown"


def keep_decl_line_for_args(line: str, dummy_args: List[str]) -> bool:
    s = strip_inline_comment(line)
    if has_c_binding_tokens(s):
        return False
    st = s.strip()
    if st.lower() == "implicit none":
        return True
    if "::" not in st:
        return False
    return any(re.search(rf"\b{re.escape(a)}\b", st) for a in dummy_args)


def make_stub_for_proc(block: List[str], kind: str, name: str) -> List[str]:
    """Replace a public procedure with a stub: keep signature + dummy-arg decls + error_on_call."""
    out: List[str] = []
    header_lines, after_header = extract_header_lines(block)
    dummy_args = parse_dummy_args(header_lines, name)

    out.extend(header_lines)

    kept_decl: List[str] = []
    i = after_header
    while i < len(block):
        statement, next_i = _gather_continued_statement(block, i)
        end = statement.lower()
        if end.startswith("end subroutine") or end.startswith("end function"):
            break
        # Classify the complete declaration, but preserve its source formatting.
        if keep_decl_line_for_args(statement, dummy_args):
            kept_decl.extend(block[i:next_i])
        i = next_i

    if not any(strip_inline_comment(ln).strip().lower() == "implicit none" for ln in kept_decl):
        indent = re.match(r"^(\s*)", header_lines[0]).group(1)
        kept_decl.insert(0, indent + "   implicit none\n")

    out.extend(kept_decl)

    header_indent = re.match(r"^(\s*)", header_lines[0]).group(1)
    body_indent = header_indent + "   "

    if kind.lower() == "function":
        res = function_result_var(header_lines, name)
        cat = function_return_category(header_lines)
        if cat == "integer":
            out.append(body_indent + f"{res} = -1\n")
        elif cat == "logical":
            out.append(body_indent + f"{res} = .false.\n")
        elif cat == "complex":
            out.append(body_indent + f"{res} = (0.0d0, 0.0d0)\n")
        elif cat == "character":
            out.append(body_indent + f'{res} = ""\n')
        else:
            out.append(body_indent + f"{res} = 0.0d0\n")

    out.append(body_indent + f'call error_on_call("{name}")\n')

    end_line = None
    for ln in reversed(block):
        if strip_inline_comment(ln).strip() == "":
            continue
        if re.match(r"^\s*end\s+(subroutine|function)\b", strip_inline_comment(ln).strip(), flags=re.IGNORECASE):
            end_line = ln
            break
    if end_line is None:
        end_line = header_indent + f"end {kind} {name}\n"
    out.append(end_line if end_line.endswith("\n") else end_line + "\n")
    out.append("\n")
    return out


def make_error_on_call_block(indent: str = "   ") -> List[str]:
    b: List[str] = []
    b.append(indent + "! Can be customized by actual hosting code\n")
    b.append(indent + "subroutine error_on_call(func)\n")
    b.append(indent + "   implicit none\n")
    b.append(indent + "   character(len=*), intent(in) :: func\n")
    b.append(indent + "   character(len=200) :: info_str\n")
    b.append("\n")
    b.append(indent + "   write(info_str,'(1X,A,A,A)') '* You have called a librpa_f03_stub routine: ', &\n")
    b.append(indent + "      trim(func), '. Make sure you have linked LibRPA library.'\n")
    b.append(indent + "   write(*,'(A)') info_str\n")
    b.append(indent + "   stop\n")
    b.append(indent + "end subroutine error_on_call\n")
    b.append("\n")
    return b

def make_no_need_change_comment_block(indent: str = "   ") -> List[str]:
    b: List[str] = []
    b.append(indent + "!=======================================================================\n")
    b.append(indent + "! Usually no need change things below\n")
    b.append(indent + "!=======================================================================\n")
    b.append("\n")
    return b


def stubify_fortran_module(src: str, src_fname: str) -> str:
    lines = src.splitlines(keepends=True)
    contains_idx = find_module_contains_index(lines)
    if contains_idx is None:
        raise SystemExit("Could not find a module-level CONTAINS")

    pre_lines = lines[:contains_idx]
    post_lines = lines[contains_idx + 1 :]

    public_names = parse_public_names(pre_lines)
    type_bound_targets = parse_type_bound_targets(pre_lines)

    public_types: Set[str] = set()
    for ln in pre_lines:
        tname = _is_type_start(ln)
        if tname and tname in public_names:
            public_types.add(tname)

    public_procs: Set[str] = set(public_names) | type_bound_targets

    header = build_header(pre_lines, public_names, public_types)

    out: List[str] = []
    if header and re.match(r"^\s*module\b", strip_inline_comment(header[0]).strip(), flags=re.IGNORECASE):
        out.append(header[0])
        out.append("   !=======================================================================\n")
        out.append("   ! Stub Fortran module for using LibRPA\n")
        out.append("   !\n")
        out.append("   ! Auto-generated by {} from {}.\n".format(os.path.basename(__file__), src_fname))
        out.append("   ! Manual edit to this file will be lost.\n")
        out.append("   !=======================================================================\n")
        out.extend(header[1:])
    else:
        out.extend(header)

    out.append("contains\n\n")
    out.extend(make_error_on_call_block(indent="   "))
    out.extend(make_no_need_change_comment_block(indent="   "))

    # Scan module-level procedures in original order, and only stub the public ones.
    start_abs = contains_idx + 1
    i = 0
    while i < len(post_lines):
        abs_i = start_abs + i
        ln = post_lines[i]

        if not is_proc_start(ln):
            i += 1
            continue

        kn = proc_kind_and_name(ln)
        if kn is None:
            i += 1
            continue
        kind, name = kn

        block, next_i, _, _ = gather_proc_block(post_lines, i)

        if name in public_procs:
            docs = gather_leading_doc_comments(lines, abs_i)
            out.extend(docs)
            out.extend(make_stub_for_proc(block, kind, name))

        i = next_i

    # end module
    end_module = None
    for ln in reversed(lines):
        if re.match(r"^\s*end\s*module\b", strip_inline_comment(ln).strip(), flags=re.IGNORECASE):
            end_module = ln
            break
    if end_module is None:
        end_module = "end module\n"
    out.append(end_module if end_module.endswith("\n") else end_module + "\n")

    return "".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="Input Fortran source (.f90)")
    ap.add_argument("output", help="Output stub Fortran source (.f90)")
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        src = f.read()

    out = stubify_fortran_module(src, os.path.basename(args.input))

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(out)


if __name__ == "__main__":
    main()
