import struct
from functools import partial


SIZE_T = struct.Struct("@N")
INT = struct.Struct("=i")
# Follow driver/reader/reader_coulomb.cpp; extend to single precision
DTYPES = {0: ("d", 1), 1: ("d", 2), 2: ("f", 1), 3: ("f", 2)}


def _extract_r_space_blocks(data, blocks=None, *, prefix_bytes=0, **kwargs):
    """Read one file into a block dictionary, rejecting duplicate IJR keys.

    Pass the same dictionary for all files in a collective extraction. Keys
    are (R1, R2, R3, I, J); values are tuples of matrix rows. Spin, spinor and
    frequency selection belongs to the filename glob in the validation entry.
    """
    BLOCK = struct.Struct("=7i")
    if blocks is None:
        blocks = {}
    pos = prefix_bytes
    n_blocks, = SIZE_T.unpack_from(data, pos)
    pos += SIZE_T.size
    dtype, = INT.unpack_from(data, pos)
    pos += INT.size
    if dtype not in DTYPES:
        raise ValueError("unknown matrix dtype {}".format(dtype))
    code, components = DTYPES[dtype]
    table_end = pos + n_blocks * (BLOCK.size + SIZE_T.size)
    if table_end > len(data):
        raise ValueError("truncated block index table")

    intervals = []
    for _ in range(n_blocks):
        *key, rows, cols = BLOCK.unpack_from(data, pos)
        key = tuple(key)
        pos += BLOCK.size
        offset, = SIZE_T.unpack_from(data, pos)
        pos += SIZE_T.size
        if key[3] < 0 or key[4] < 0 or rows <= 0 or cols <= 0:
            raise ValueError("invalid atom index or matrix dimensions for {}".format(key))
        if key in blocks:
            raise ValueError("duplicate matrix block {}".format(key))
        count = rows * cols * components
        end = offset + count * struct.calcsize("=" + code)
        if offset < table_end or end > len(data):
            raise ValueError("payload outside data region for {}".format(key))
        intervals.append((offset, end))
        values = struct.unpack_from("={}{}".format(count, code), data, offset)
        if components == 2:
            values = tuple(complex(re, im) for re, im in zip(values[::2], values[1::2]))
        blocks[key] = tuple(values[i * cols:(i + 1) * cols] for i in range(rows))

    intervals.sort()
    if any(left[1] > right[0] for left, right in zip(intervals, intervals[1:])):
        raise ValueError("overlapping matrix payloads")
    return blocks


# Add new preset to extractors
extractors = {
    "ExxR": _extract_r_space_blocks,
    "SigcRF": partial(_extract_r_space_blocks, prefix_bytes=struct.calcsize("=d")),
}
