import pathlib
import struct
import tempfile
import unittest
import warnings

from regression_tests.backend import binary_presets as bp
from regression_tests.backend import validate as validate_module


def make_file(kind="ExxR", nprocs=1, myid=0, run_id=7, dtype=0, omega=None, blocks=(),
              magic=bp.MAGIC, endian=bp.ENDIAN_MARKER, version=bp.VERSION):
    """Build a file. blocks is a list of ((R1, R2, R3, I, J), rows) with rows of numbers."""
    header = bp.HEADER.pack(magic, endian, version, bp.KINDS[kind], nprocs, myid, dtype,
                            run_id, len(blocks))
    if kind == "SigcRF":
        header += bp.OMEGA.pack(1.5 if omega is None else omega)
    table_end = len(header) + len(blocks) * (bp.BLOCK.size + bp.OFFSET.size)
    code, components = bp.DTYPES[dtype]
    index, payload = b"", b""
    for key, rows in blocks:
        values = [x for row in rows for x in row]
        if components == 2:
            values = [part for v in values for part in (complex(v).real, complex(v).imag)]
        index += bp.BLOCK.pack(*key, len(rows), len(rows[0]))
        index += bp.OFFSET.pack(table_end + len(payload))
        payload += struct.pack("={}{}".format(len(values), code), *values)
    return header + index + payload


def block(i, value, key=(0, 0, 0, 0, 0)):
    return ((key[0], key[1], key[2], i, key[4]), [[value, value + 1.0]])


class TestHeader(unittest.TestCase):

    def test_header_size_follows_the_layout(self):
        self.assertEqual(bp.HEADER.size, 48)
        self.assertEqual(len(make_file("SigcRF")), 56)

    def test_header_fields_are_read(self):
        data = make_file("SigcRF", nprocs=4, myid=2, run_id=99, dtype=1, omega=2.5)
        header = bp.parse_header(data, "SigcRF")
        self.assertEqual((header["nprocs"], header["myid"], header["run_id"]), (4, 2, 99))
        self.assertEqual((header["dtype"], header["omega"], header["size"]), (1, 2.5, 56))

    def test_invalid_headers_are_rejected(self):
        bad = {
            "magic": make_file(magic=b"LIBRPAXX"),
            "endian": make_file(endian=0x04030201),
            "version": make_file(version=2),
            "nprocs": make_file(nprocs=0, myid=0),
            "myid": make_file(nprocs=2, myid=2),
            "dtype": make_file(dtype=0)[:28] + struct.pack("=i", 9) + make_file()[32:],
            "run_id": make_file(run_id=0),
            "truncated": make_file()[:20],
            "truncated omega": make_file("SigcRF")[:50],
        }
        for name, data in bad.items():
            with self.subTest(name=name):
                preset = "SigcRF" if name == "truncated omega" else "ExxR"
                with self.assertRaises(ValueError):
                    bp.parse_header(data, preset)

    def test_kind_must_match_the_preset(self):
        with self.assertRaises(ValueError):
            bp.parse_header(make_file("SigcRF"), "ExxR")
        with self.assertRaises(ValueError):
            bp.parse_header(make_file("ExxR"), "SigcRF")

    def test_unknown_preset_is_rejected(self):
        with self.assertRaises(ValueError):
            bp.parse_header(make_file(), "Other")


class TestBlocks(unittest.TestCase):

    def test_blocks_are_decoded(self):
        data = make_file(blocks=[block(0, 1.0), block(1, 3.0)])
        blocks = bp.extract_file(data, "ExxR")
        self.assertEqual(blocks[(0, 0, 0, 1, 0)], ((3.0, 4.0),))
        self.assertEqual(len(blocks), 2)

    def test_complex_and_single_precision_blocks(self):
        data = make_file(dtype=1, blocks=[block(0, 1.0)])
        self.assertEqual(bp.extract_file(data, "ExxR")[(0, 0, 0, 0, 0)], ((1 + 0j, 2 + 0j),))
        data = make_file(dtype=2, blocks=[block(0, 0.5)])
        self.assertEqual(bp.extract_file(data, "ExxR")[(0, 0, 0, 0, 0)], ((0.5, 1.5),))

    def test_truncated_index_table(self):
        data = make_file(blocks=[block(0, 1.0), block(1, 2.0)])
        with self.assertRaises(ValueError):
            bp.extract_file(data[:60], "ExxR")

    def test_huge_block_count_is_rejected_without_allocation(self):
        data = bytearray(make_file(blocks=[block(0, 1.0)]))
        struct.pack_into("=Q", data, 40, 2 ** 62)
        with self.assertRaises(ValueError):
            bp.extract_file(bytes(data), "ExxR")

    def test_payload_outside_the_file(self):
        data = bytearray(make_file(blocks=[block(0, 1.0)]))
        struct.pack_into("=Q", data, 48 + bp.BLOCK.size, 1 << 40)
        with self.assertRaises(ValueError):
            bp.extract_file(bytes(data), "ExxR")

    def test_payload_inside_the_index_table(self):
        data = bytearray(make_file(blocks=[block(0, 1.0)]))
        struct.pack_into("=Q", data, 48 + bp.BLOCK.size, 8)
        with self.assertRaises(ValueError):
            bp.extract_file(bytes(data), "ExxR")

    def test_duplicate_block_in_a_file(self):
        with self.assertRaises(ValueError):
            bp.extract_file(make_file(blocks=[block(0, 1.0), block(0, 2.0)]), "ExxR")

    def test_overlapping_payloads(self):
        data = bytearray(make_file(blocks=[block(0, 1.0), block(1, 2.0)]))
        first = struct.unpack_from("=Q", data, 48 + bp.BLOCK.size)[0]
        struct.pack_into("=Q", data, 48 + 2 * bp.BLOCK.size + bp.OFFSET.size, first + 8)
        with self.assertRaises(ValueError):
            bp.extract_file(bytes(data), "ExxR")

    def test_invalid_dimensions(self):
        data = bytearray(make_file(blocks=[block(0, 1.0)]))
        struct.pack_into("=i", data, 48 + 5 * 4, 0)
        with self.assertRaises(ValueError):
            bp.extract_file(bytes(data), "ExxR")


class TestCollective(unittest.TestCase):

    def _run(self, nprocs=2, **overrides):
        files = {}
        for myid in range(nprocs):
            files["f{}".format(myid)] = make_file(
                nprocs=nprocs, myid=myid, blocks=[block(myid, float(myid))])
        files.update(overrides)
        return bp.extract_collective(files, "ExxR")

    def test_blocks_of_all_ranks_are_merged(self):
        # The result holds exactly the blocks: no header value (rank count, rank, run
        # identifier) is part of it.
        expected = {(0, 0, 0, i, 0): ((float(i), float(i) + 1.0),) for i in range(3)}
        self.assertEqual(self._run(3), expected)

    def test_merge_does_not_depend_on_the_split(self):
        both = bp.extract_collective(
            {"a": make_file(nprocs=1, blocks=[block(0, 0.0), block(1, 1.0)])}, "ExxR")
        self.assertEqual(self._run(2), both)

    def test_missing_shard(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            bp.extract_collective({"f1": make_file(nprocs=2, myid=1)}, "ExxR")
        files = {"f0": make_file(nprocs=3, myid=0), "f2": make_file(nprocs=3, myid=2)}
        with self.assertRaisesRegex(ValueError, "missing"):
            bp.extract_collective(files, "ExxR")

    def test_mixed_run_id(self):
        files = {"f0": make_file(nprocs=2, myid=0, run_id=1, blocks=[block(0, 0.0)]),
                 "f1": make_file(nprocs=2, myid=1, run_id=2, blocks=[block(1, 0.0)])}
        with self.assertRaisesRegex(ValueError, "another run"):
            bp.extract_collective(files, "ExxR")

    def test_wrong_rank_count_below_the_run_size(self):
        files = {"f0": make_file(nprocs=2, myid=0), "f1": make_file(nprocs=3, myid=1)}
        with self.assertRaisesRegex(ValueError, "ranks"):
            bp.extract_collective(files, "ExxR")

    def test_duplicate_rank(self):
        files = {"a": make_file(nprocs=2, myid=0), "b": make_file(nprocs=2, myid=0),
                 "c": make_file(nprocs=2, myid=1)}
        with self.assertRaisesRegex(ValueError, "both claim rank"):
            bp.extract_collective(files, "ExxR")

    def test_leftover_files_of_higher_ranks_are_ignored_with_a_warning(self):
        files = {"f0": make_file(nprocs=2, myid=0, blocks=[block(0, 0.0)]),
                 "f1": make_file(nprocs=2, myid=1, blocks=[block(1, 1.0)]),
                 # from an earlier run with four ranks; it duplicates a block of this run
                 "f2": make_file(nprocs=4, myid=2, run_id=5, blocks=[block(0, 9.0)]),
                 "f3": make_file(nprocs=4, myid=3, run_id=5)}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            blocks = bp.extract_collective(files, "ExxR")
        self.assertEqual(len(caught), 1)
        self.assertEqual(blocks[(0, 0, 0, 0, 0)], ((0.0, 1.0),))

    def test_leftover_below_the_run_size_is_an_error(self):
        # An old file with a rank that this run should have overwritten.
        files = {"f0": make_file(nprocs=2, myid=0), "f1": make_file(nprocs=4, myid=1, run_id=5)}
        with self.assertRaises(ValueError):
            bp.extract_collective(files, "ExxR")

    def test_duplicate_block_across_ranks(self):
        files = {"f0": make_file(nprocs=2, myid=0, blocks=[block(0, 0.0)]),
                 "f1": make_file(nprocs=2, myid=1, blocks=[block(0, 1.0)])}
        with self.assertRaisesRegex(ValueError, "duplicate"):
            bp.extract_collective(files, "ExxR")

    def test_bad_file_among_good_ones(self):
        files = {"f0": make_file(nprocs=2, myid=0),
                 "f1": make_file(nprocs=2, myid=1, magic=b"LIBRPAXX")}
        with self.assertRaises(ValueError):
            bp.extract_collective(files, "ExxR")

    def test_sigc_frequency_must_agree(self):
        files = {"f0": make_file("SigcRF", nprocs=2, myid=0, omega=1.0),
                 "f1": make_file("SigcRF", nprocs=2, myid=1, omega=2.0)}
        with self.assertRaisesRegex(ValueError, "frequency"):
            bp.extract_collective(files, "SigcRF")

    def test_no_files(self):
        with self.assertRaises(ValueError):
            bp.extract_collective({}, "ExxR")


class TestExtractFiles(unittest.TestCase):

    def _extract(self, directory, collective):
        return validate_module._extract_files(
            directory, "librpa/out/ExxR_*.bin", None, 0, None, (0, None, 1),
            binary_extract_preset="ExxR", collective=collective)

    def test_collective_extraction_from_a_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp) / "librpa" / "out"
            out.mkdir(parents=True)
            for myid in range(2):
                (out / "ExxR_ispin_00_s_00_myid_{:05d}.bin".format(myid)).write_bytes(
                    make_file(nprocs=2, myid=myid, blocks=[block(myid, 1.0)]))
            (out / "ExxR_ispin_00_s_00_myid_00003.bin").write_bytes(
                make_file(nprocs=4, myid=3, run_id=5, blocks=[block(0, 9.0)]))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                blocks = self._extract(tmp, True)
            self.assertEqual(sorted(k[3] for k in blocks), [0, 1])
            per_file = self._extract_single(tmp)
            self.assertEqual(len(per_file), 3)

    def _extract_single(self, tmp):
        return self._extract(tmp, False)

    def test_pattern_without_files_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                self._extract(tmp, True)


if __name__ == "__main__":
    unittest.main()
