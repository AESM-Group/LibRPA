"""Readers of the indexed real-space block files written by LibRPA (ExxR, SigcRF).

File layout (native byte order, see RspaceFileHeader in src/io/output_matrix.h):

    char[8] magic "LIBRPAIJ", uint32 endian marker 0x01020304, int32 version,
    int32 kind (0 ExxR, 1 SigcRF), int32 nprocs, int32 myid, int32 dtype,
    uint64 run_id, uint64 n_blocks, [double omega, SigcRF only]   (48 or 56 bytes)

followed by n_blocks records of int32[7] (R1, R2, R3, I, J, rows, cols) and a uint64
absolute payload offset, and then the row-major payloads. One MPI rank writes one file;
the header names the rank count and a run identifier shared by the files of one output,
so files left over from another run can be told apart.
"""

import struct
import warnings


MAGIC = b"LIBRPAIJ"
ENDIAN_MARKER = 0x01020304
VERSION = 1
KINDS = {"ExxR": 0, "SigcRF": 1}
HEADER = struct.Struct("=8sIiiiiiQQ")
OMEGA = struct.Struct("=d")
BLOCK = struct.Struct("=7i")
OFFSET = struct.Struct("=Q")
# Follow driver/reader/reader_coulomb.cpp; extend to single precision
DTYPES = {0: ("d", 1), 1: ("d", 2), 2: ("f", 1), 3: ("f", 2)}


def parse_header(data, preset):
    """Return the header of one file as a dictionary; raise ValueError if it is invalid.

    The keys are kind, nprocs, myid, dtype, run_id, n_blocks, omega (None for ExxR) and
    size (the number of header bytes). The kind must be the one of the preset.
    """
    if preset not in KINDS:
        raise ValueError("unknown binary_extract_preset: " + preset)
    if len(data) < HEADER.size:
        raise ValueError("truncated file header")
    magic, endian, version, kind, nprocs, myid, dtype, run_id, n_blocks = \
        HEADER.unpack_from(data, 0)
    if magic != MAGIC:
        raise ValueError("not a real-space matrix file (bad magic)")
    if endian != ENDIAN_MARKER:
        raise ValueError("file has a different byte order")
    if version != VERSION:
        raise ValueError("unsupported file version {}".format(version))
    if kind != KINDS[preset]:
        raise ValueError("file kind {} does not match preset {}".format(kind, preset))
    if nprocs < 1 or not 0 <= myid < nprocs:
        raise ValueError("invalid rank information myid={} nprocs={}".format(myid, nprocs))
    if dtype not in DTYPES:
        raise ValueError("unknown matrix dtype {}".format(dtype))
    if run_id == 0:
        raise ValueError("invalid run identifier")
    size = HEADER.size
    omega = None
    if preset == "SigcRF":
        if len(data) < size + OMEGA.size:
            raise ValueError("truncated file header")
        omega, = OMEGA.unpack_from(data, size)
        size += OMEGA.size
    return {"kind": kind, "nprocs": nprocs, "myid": myid, "dtype": dtype, "run_id": run_id,
            "n_blocks": n_blocks, "omega": omega, "size": size}


def read_blocks(data, header, blocks=None):
    """Read the blocks of one file into a dictionary, rejecting duplicate IJR keys.

    Pass the same dictionary for several files to merge them. Keys are
    (R1, R2, R3, I, J); values are tuples of matrix rows. Spin, spinor and
    frequency selection belongs to the filename glob in the validation entry.
    """
    if blocks is None:
        blocks = {}
    n_blocks = header["n_blocks"]
    code, components = DTYPES[header["dtype"]]
    pos = header["size"]
    if n_blocks > (len(data) - pos) // (BLOCK.size + OFFSET.size):
        raise ValueError("truncated block index table")
    table_end = pos + n_blocks * (BLOCK.size + OFFSET.size)

    intervals = []
    for _ in range(n_blocks):
        *key, rows, cols = BLOCK.unpack_from(data, pos)
        key = tuple(key)
        pos += BLOCK.size
        offset, = OFFSET.unpack_from(data, pos)
        pos += OFFSET.size
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


def extract_file(data, preset):
    """Return the block dictionary of a single file."""
    return read_blocks(data, parse_header(data, preset))


def extract_collective(files, preset):
    """Merge the files of one output, one per MPI rank, into one block dictionary.

    files maps a name (used in messages) to the bytes of the file. The file of rank 0 defines
    the set: its header gives the rank count, the run identifier and, for SigcRF, the
    frequency. Files of ranks at or above that rank count are left over from another
    run in the same directory; they are ignored with a warning. The remaining files must be
    exactly the ranks 0..nprocs-1 of this run, without duplicates or files of another run.
    The header values are checked here and are not part of the result.
    """
    if not files:
        raise ValueError("no files to merge")
    headers = {name: parse_header(data, preset) for name, data in files.items()}
    first = [name for name, h in headers.items() if h["myid"] == 0]
    if not first:
        raise ValueError("file of rank 0 is missing")
    reference = headers[sorted(first, key=str)[0]]
    nprocs = reference["nprocs"]
    kept, leftover = {}, []
    for name, header in sorted(headers.items(), key=lambda item: str(item[0])):
        if header["myid"] >= nprocs:
            leftover.append(name)
            continue
        if header["nprocs"] != nprocs:
            raise ValueError("{}: written by {} ranks, but the run has {}".format(
                name, header["nprocs"], nprocs))
        if header["run_id"] != reference["run_id"]:
            raise ValueError("{}: belongs to another run".format(name))
        if header["omega"] != reference["omega"]:
            raise ValueError("{}: different frequency".format(name))
        if header["myid"] in kept:
            raise ValueError("{} and {}: both claim rank {}".format(
                kept[header["myid"]], name, header["myid"]))
        kept[header["myid"]] = name
    missing = sorted(set(range(nprocs)) - set(kept))
    if missing:
        raise ValueError("files of ranks {} are missing (run of {} ranks)".format(
            missing, nprocs))
    if leftover:
        warnings.warn("ignoring files of ranks >= {} from another run: {}".format(
            nprocs, ", ".join(str(n) for n in leftover)))
    blocks = {}
    for myid in range(nprocs):
        name = kept[myid]
        try:
            read_blocks(files[name], headers[name], blocks)
        except ValueError as exc:
            raise ValueError("{}: {}".format(name, exc)) from exc
    return blocks
