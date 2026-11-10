#!/usr/bin/env python3
"""Tests for the MKL threading layer that cmake/FindIntelMKL.cmake selects."""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MKL_LIBRARIES = (
    "mkl_intel_lp64",
    "mkl_intel_thread",
    "mkl_gnu_thread",
    "mkl_core",
    "mkl_scalapack_lp64",
    "mkl_blacs_intelmpi_lp64",
    "mkl_blacs_openmpi_lp64",
)

PROJECT = """\
cmake_minimum_required(VERSION 3.18)
project(find_intel_mkl_fixture CXX)
set(CMAKE_CXX_COMPILER_ID "${{TEST_COMPILER_ID}}")
add_library(MPI::MPI_CXX INTERFACE IMPORTED)
set_property(TARGET MPI::MPI_CXX PROPERTY INTERFACE_INCLUDE_DIRECTORIES "{mpi_include}")
list(APPEND CMAKE_MODULE_PATH "{module_dir}")
find_package(IntelMKL REQUIRED)
file(WRITE "${{CMAKE_BINARY_DIR}}/mkl_libraries.txt" "${{MKL_LIBRARIES}}")
"""


@unittest.skipUnless(shutil.which("cmake"), "cmake is not available")
class ThreadingLayerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.mklroot = self.tmp / "mkl"
        (self.mklroot / "include").mkdir(parents=True)
        (self.mklroot / "include" / "mkl_dfti.h").touch()
        (self.mklroot / "lib").mkdir()
        for name in MKL_LIBRARIES:
            (self.mklroot / "lib" / f"lib{name}.a").touch()
        self.mpi_include = self.tmp / "mpi"
        self.mpi_include.mkdir()
        self.source = self.tmp / "src"
        self.source.mkdir()
        (self.source / "CMakeLists.txt").write_text(
            PROJECT.format(module_dir=(REPO / "cmake").as_posix(),
                           mpi_include=self.mpi_include.as_posix())
        )

    def selected_libraries(self, compiler_id, mpi_macro="MPICH_VERSION"):
        (self.mpi_include / "mpi.h").write_text(f"#define {mpi_macro} 1\n")
        build = self.tmp / f"build-{compiler_id}"
        subprocess.run(
            [
                "cmake",
                "-S", str(self.source),
                "-B", str(build),
                f"-DMKLROOT={self.mklroot}",
                f"-DTEST_COMPILER_ID={compiler_id}",
            ],
            env={key: value for key, value in os.environ.items()
                 if key not in ("CPATH", "CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH")},
            check=True,
            capture_output=True,
            text=True,
        )
        text = (build / "mkl_libraries.txt").read_text()
        return [Path(path).name for path in text.split(";")]

    def test_gnu_uses_gnu_thread(self):
        names = self.selected_libraries("GNU")
        self.assertIn("libmkl_gnu_thread.a", names)
        self.assertNotIn("libmkl_intel_thread.a", names)

    def test_intel_and_llvm_use_intel_thread(self):
        for compiler_id in ("IntelLLVM", "Intel", "Clang"):
            with self.subTest(compiler_id=compiler_id):
                names = self.selected_libraries(compiler_id)
                self.assertIn("libmkl_intel_thread.a", names)
                self.assertNotIn("libmkl_gnu_thread.a", names)

    def test_blacs_matches_mpi_and_refreshes_cached_path(self):
        for macro, library in (
            ("MPICH_VERSION", "intelmpi"),
            ("OMPI_MAJOR_VERSION", "openmpi"),
            ("I_MPI_VERSION", "intelmpi"),
        ):
            with self.subTest(macro=macro):
                names = self.selected_libraries("GNU", macro)
                blacs = [name for name in names if "blacs" in name]
                self.assertEqual(blacs, [f"libmkl_blacs_{library}_lp64.a"])

    def test_unknown_mpi_is_rejected(self):
        with self.assertRaises(subprocess.CalledProcessError) as error:
            self.selected_libraries("GNU", "UNKNOWN_MPI")
        self.assertIn("cannot identify the selected MPI", error.exception.stderr)


if __name__ == "__main__":
    unittest.main()
