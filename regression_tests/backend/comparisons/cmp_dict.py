"""Compare numeric data in dictionaries, independently of their source format."""

import math


__all__ = ["abs_diff"]


def abs_diff(tolerance, precision=3):
    """Compare matching keys, sequence shapes, and real/complex numeric values.

    Dictionaries may be nested, as in filename -> block key -> matrix rows.
    Empty subdictionaries are allowed; inputs with no numeric values fail.
    Report the largest absolute difference and its key/index path. Sequence
    indices in diagnostics are one-based. Nonfinite values always fail.
    """
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    precision = int(precision)

    def compare(test, reference):
        largest = 0.0
        location = ()
        count = 0

        def visit(a, b, path):
            nonlocal largest, location, count
            if isinstance(a, dict) or isinstance(b, dict):
                if not isinstance(a, dict) or not isinstance(b, dict):
                    raise ValueError("dictionary mismatch at {}".format(path))
                if a.keys() != b.keys():
                    raise ValueError("key mismatch at {}: test-only {}, reference-only {}".format(
                        path, sorted(a.keys() - b.keys(), key=str),
                        sorted(b.keys() - a.keys(), key=str)))
                for key in sorted(a, key=str):
                    visit(a[key], b[key], path + (key,))
            elif isinstance(a, (tuple, list)) or isinstance(b, (tuple, list)):
                if (not isinstance(a, (tuple, list)) or not isinstance(b, (tuple, list))
                        or len(a) != len(b)):
                    raise ValueError("shape mismatch at {}".format(path))
                for index, (x, y) in enumerate(zip(a, b), 1):
                    visit(x, y, path + (index,))
            else:
                x, y = complex(a), complex(b)
                if not all(math.isfinite(v) for v in (x.real, x.imag, y.real, y.imag)):
                    raise ValueError("nonfinite value at {}".format(path))
                difference = abs(x - y)
                if difference > largest:
                    largest, location = difference, path
                count += 1

        if not isinstance(test, dict) or not isinstance(reference, dict):
            return False, "expected dictionaries"
        try:
            visit(test, reference, ())
        except (TypeError, ValueError) as exc:
            return False, str(exc)
        if count == 0:
            return False, "no numeric values found"
        message = "max abs diff = {0:.{p}E} (tol = {1:.{p}E}) over {2} values".format(
            largest, tolerance, count, p=precision)
        if location:
            message += ", max at {}".format(location)
        return largest <= tolerance, message

    return compare
