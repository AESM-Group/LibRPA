import pathlib
import re
import importlib
import struct
from typing import Tuple

from .binary_presets import extractors


def _extract_files(directory: str, file: str,
                   regex: re.Pattern, headers: int, rows: int, occurences: Tuple[int, int, int],
                   binary_extract=None, binary_extract_preset=None, collective=False):
    """Extract text selections or decode binary values."""
    raw = {}
    d = pathlib.Path(directory)
    matches = list(d.glob(file))
    canonical_prefix = None
    if not matches and file.startswith("librpa/"):
        fallback = pathlib.PurePosixPath(file).name
        matches = list(d.glob(fallback))
        canonical_prefix = pathlib.Path("librpa")

    if binary_extract_preset is not None and not matches:
        raise ValueError("no files found for {} in {}".format(file, directory))

    for f in sorted(matches):
        rela = f.relative_to(d)
        if canonical_prefix is not None:
            rela = canonical_prefix / f.name
        if binary_extract_preset is not None:
            try:
                blocks = extractors[binary_extract_preset](
                    f.read_bytes(), raw if collective else None)
            except (struct.error, ValueError) as exc:
                raise ValueError("invalid binary data in {}: {}".format(f, exc)) from exc
            if not collective:
                raw[rela] = blocks
            continue
        if binary_extract is not None:
            try:
                raw[rela] = binary_extract(f.read_bytes())
            except struct.error as exc:
                raise ValueError("invalid binary data in {}: {}".format(f, exc)) from exc
            continue
        with open(f, 'r') as h:
            lines = h.readlines()
        # When regex is None, treat the whole file as necessary data
        if regex is None:
            if rows is None:
                raw[rela] = "".join(lines[headers:])
            else:
                raw[rela] = "".join(lines[headers:rows])
            continue

        st, ed, intv = occurences
        matched = []
        for i, l in enumerate(lines):
            m = regex.search(l)
            if m is not None:
                # the matched line have the data
                if rows is None:
                    if len(m.groups()) == 1:
                        matched.append(m.group(1))
                    else:
                        matched.append(" ".join(m.groups()))
                else:
                    matched.append("".join(lines[i + headers: i + headers + rows]))
        if ed is None:
            ed = len(matched) - 1
        raw[rela] = [matched[i] for i in range(st, ed + 1, intv)]
    return raw


def _align_extracted_files(test, refr, align):
    if not align or not isinstance(test, dict) or not isinstance(refr, dict):
        return test, refr
    if set(test) == set(refr) or len(test) != len(refr):
        return test, refr

    test_items = sorted(test.items(), key=lambda item: str(item[0]))
    refr_items = sorted(refr.items(), key=lambda item: str(item[0]))
    keys = [key for key, _ in test_items]
    return (
        dict(zip(keys, [value for _, value in test_items])),
        dict(zip(keys, [value for _, value in refr_items])),
    )


def _process_regex(regex: str):
    if regex is not None:
        regex = re.compile(regex)
    return regex


def _import_comparison(comparison: str):
    """
    import a function from a given module and construct the closure from
    the additional arguments, if any
    """
    REGEX_IMPORTCHECK = re.compile(r"(?P<mod>[A-z0-9_]+).(?P<func>[A-z0-9_]+)"
                                   r"[ ]*(?:\((?P<args>.*)\)|)$")
    try:
        match = REGEX_IMPORTCHECK.match(comparison)
        modname = "backend.comparisons.%s" % (match.group("mod"))
        module = importlib.import_module(modname)
        if match.group("args") is not None:
            paramlist = match.group("args").split(",")
            args = tuple(x for x in paramlist if "=" not in x)
            keywords = tuple(x.split("=") for x in paramlist if "=" in x)
            keywords = {x[0].strip(): x[1] for x in keywords}
            return getattr(module, match.group("func"))(*args, **keywords)
        else:
            return getattr(module, match.group("func"))
    except ImportError:
        raise ImportError("Failed to import comparision method: " + comparison)


def _import_binary_extract(binary_extract: str):
    """Compile a binary layout into a numeric data extractor.

    Numeric struct codes bBhHiIlLqQefd? and padding x accept repeat counts.
    Without parentheses, all values are returned: 2i128d yields two integers
    and 128 doubles. With parentheses, only grouped values are returned:
    2i(128d) yields 128 doubles; 2i(128d)2i(128d) yields 256 doubles in file order.
    Groups must be nonempty and cannot be nested. Unselected fields still
    contribute to the layout; padding never produces values.

    The default prefix is = (native byte order, standard sizes, no alignment
    padding). Explicit @, =, <, >, and ! prefixes retain their struct meanings.
    Whitespace is ignored. Decoding requires an exact file-size match.

    Args:
        binary_extract: Numeric struct layout with optional parenthesized
            selections, or None to use plain-text extraction.

    Returns:
        A callable accepting bytes and returning selected numeric values as a
        list, or None. The callable raises struct.error for a size mismatch.

    Raises:
        ValueError: The layout contains unsupported codes or invalid groups.
        struct.error: The resulting struct format cannot be compiled.
    """
    if binary_extract is None:
        return None
    layout = re.sub(r"\s+", "", binary_extract)
    byte_order = "="
    if layout and layout[0] in "@=<>!":
        byte_order, layout = layout[0], layout[1:]
    field = r"\d*[bBhHiIlLqQefd?x]"
    if not re.fullmatch(r"(?:" + field + r"|\((?:" + field + r")+\))+", layout):
        raise ValueError("invalid binary_extract layout: {}".format(binary_extract))
    selected, index = [], 0
    include = "(" not in layout
    for token in re.findall(field + r"|[()]", layout):
        if token in ("(", ")"):
            include = token == "("
        elif token[-1] != "x":
            count = int(token[:-1]) if token[:-1] else 1
            if include:
                selected.extend(range(index, index + count))
            index += count
    unpacker = struct.Struct(byte_order + layout.replace("(", "").replace(")", ""))

    def extract(data):
        values = unpacker.unpack(data)
        return [values[index] for index in selected]

    return extract


def _process_rows(rows: str):
    if rows is not None:
        rows = int(rows)
    return rows


def _process_headers(headers: str):
    if headers is not None:
        return int(headers)
    return 0


def _process_occurences(occurences: str):
    st, ed, intv = 0, None, 1
    if occurences is not None:
        words = occurences.split(":")
        if len(words) == 3:
            if words[0]:
                st = int(words[0])
            if words[1]:
                ed = int(words[1])
            if words[2]:
                intv = int(words[2])
        elif len(words) == 2:
            if words[0] == "":
                st = 0
                ed = int(words[1])
            elif words[1] == "":
                st = words[0]
                ed = None
            else:
                st, ed = map(int, words)
        elif len(words) == 1:
            # single occurence
            st = int(words[0])
            ed = int(words[0])
        else:
            raise ValueError("invalid occurences")
    return st, ed, intv


class Validate():

    def __init__(self, name: str, file: str, comparison: str, headers: str, rows: str,
                 regex: str, occurences: str, binary_extract: str,
                 file_test: str = None, file_refr: str = None,
                 collective: bool = False, binary_extract_preset: str = None):
        """Configure file matching, data extraction, and comparison.

        Args:
            name: Display name of this validation entry.
            file: Common glob pattern relative to the test and reference
                directories.
            comparison: Comparison function specification, including optional
                arguments, such as "cmp_float.abs_diff(1e-8)".
            headers: Number of text lines to skip from a regex match, or from
                the start of a file when regex is None. Defaults to zero.
            rows: Number of lines to extract from a regex match. Without regex,
                this is the exclusive ending line index. None selects matched
                capture groups or the remaining file, respectively.
            regex: Pattern locating text data, or None to select the whole file.
            occurences: Zero-based match selector, such as "0" or "0:3";
                range endpoints are inclusive. None selects all matches.
            binary_extract: Binary layout, or None for text extraction. See
                _import_binary_extract for syntax and selection rules. When
                provided, regex, headers, rows, and occurences are not used
                to extract data.
            file_test: Optional test-file pattern overriding file.
            file_refr: Optional reference-file pattern overriding file.
            collective: Merge all matched preset files into one block dictionary.
                False preserves the outer filename dictionary. Requires a preset.
            binary_extract_preset: ExxR or SigcRF indexed binary reader, mutually
                exclusive with binary_extract. Text selectors are ignored.

        Raises:
            ImportError: The comparison module cannot be imported.
            ValueError: A numeric option, occurrence selector, or binary
                layout is invalid.
            re.error: The regular expression is invalid.
            struct.error: The binary struct format cannot be compiled.
        """
        if not isinstance(collective, bool):
            raise ValueError("collective must be a boolean")
        if binary_extract_preset is not None:
            if binary_extract is not None:
                raise ValueError("binary_extract and binary_extract_preset are mutually exclusive")
            if binary_extract_preset not in extractors:
                raise ValueError("unknown binary_extract_preset: " + binary_extract_preset)
        elif collective:
            raise ValueError("collective requires binary_extract_preset")
        self._collective = collective
        self._binary_extract_preset = binary_extract_preset
        self._name = name
        self._file_test = file_test if file_test is not None else file
        self._file_refr = file_refr if file_refr is not None else file
        self._align_files = self._file_test != self._file_refr
        self._comparison = _import_comparison(comparison)
        self._headers = _process_headers(headers)
        self._rows = _process_rows(rows)
        self._regex = _process_regex(regex)
        self._occurences = _process_occurences(occurences)
        self._binary_extract = _import_binary_extract(binary_extract)

    def evaluate(self, dir_test, dir_refr):
        try:
            test = _extract_files(dir_test, self._file_test, self._regex, self._headers,
                                  self._rows, self._occurences, self._binary_extract,
                                  self._binary_extract_preset, self._collective)
            refr = _extract_files(dir_refr, self._file_refr, self._regex, self._headers,
                                  self._rows, self._occurences, self._binary_extract,
                                  self._binary_extract_preset, self._collective)
        except (OSError, ValueError) as exc:
            return False, str(exc)

        if not self._collective:
            test, refr = _align_extracted_files(test, refr, self._align_files)

        # print(dir_test, dir_refr)
        return self._comparison(test, refr)
