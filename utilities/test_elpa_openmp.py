#!/usr/bin/env python3
"""Configure fixtures for ELPA's OpenMP dependency checks."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("cmake"), "cmake is not available")
class ElpaOpenMPTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)

    def configure(self, body, success=True):
        (self.tmp / "CMakeLists.txt").write_text(
            'cmake_minimum_required(VERSION 3.18)\nproject(elpa_fixture NONE)\n'
            f'list(APPEND CMAKE_MODULE_PATH "{REPO / "cmake"}")\n'
            'include(CheckELPAOpenMP)\n'
            'set(OpenMP_CXX_LIBRARIES /gnu/libgomp.so)\n' + body
        )
        result = subprocess.run(
            ["cmake", "-S", str(self.tmp), "-B", str(self.tmp / "build")],
            capture_output=True, text=True,
            env=dict(os.environ, PATH=f"{self.tmp}:{os.environ['PATH']}"),
        )
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode == 0, success, output)
        return output

    def test_parent_target_transitive_runtime_conflict(self):
        output = self.configure('''
add_library(math INTERFACE IMPORTED)
set_property(TARGET math PROPERTY INTERFACE_LINK_LIBRARIES -lmkl_intel_thread)
add_library(ELPA::ELPA INTERFACE IMPORTED)
set_property(TARGET ELPA::ELPA PROPERTY INTERFACE_LINK_LIBRARIES math)
librpa_check_elpa_openmp(ELPA::ELPA)
''', success=False)
        self.assertIn("Prebuilt ELPA: incompatible OpenMP runtimes", output)

    def test_matching_parent_target_with_cycle(self):
        self.configure('''
add_library(math INTERFACE IMPORTED)
set_property(TARGET math PROPERTY INTERFACE_LINK_LIBRARIES "-lmkl_gnu_thread;ELPA::ELPA")
add_library(ELPA::ELPA INTERFACE IMPORTED)
set_property(TARGET ELPA::ELPA PROPERTY INTERFACE_LINK_LIBRARIES math)
librpa_check_elpa_openmp(ELPA::ELPA)
''')

    def fake_ldd(self, *dependencies):
        # ldd starts each dependency line with a tab.
        lines = "".join(f"\\t{dependency}\\n" for dependency in dependencies)
        ldd = self.tmp / "ldd"
        ldd.write_text(f"#!/bin/sh\nprintf '{lines}'\n")
        ldd.chmod(0o755)

    def configure_shared_elpa(self, openmp_library="/gnu/libgomp.so", success=True):
        library = self.tmp / "libelpa.so"
        library.touch()
        return self.configure(f'''
set(CMAKE_SYSTEM_NAME Linux)
set(OpenMP_CXX_LIBRARIES "{openmp_library}")
add_library(ELPA::ELPA SHARED IMPORTED)
set_property(TARGET ELPA::ELPA PROPERTY IMPORTED_LOCATION "{library}")
librpa_check_elpa_openmp(ELPA::ELPA)
''', success=success)

    def test_unresolved_runtime_conflict(self):
        self.fake_ldd("libiomp5.so => not found", "libc.so.6 => /lib/libc.so.6 (0x0)")
        output = self.configure_shared_elpa(success=False)
        self.assertIn("Prebuilt ELPA: incompatible OpenMP runtimes", output)

    def test_intel_and_llvm_runtimes_conflict(self):
        self.fake_ldd("libiomp5.so => /intel/libiomp5.so (0x0)")
        output = self.configure_shared_elpa("/llvm/libomp.so", success=False)
        self.assertIn("Prebuilt ELPA: incompatible OpenMP runtimes", output)

    def test_intel_thread_layer_accepts_one_intel_or_llvm_runtime(self):
        for runtime in ("iomp5", "omp"):
            with self.subTest(runtime=runtime):
                self.fake_ldd(
                    "libmkl_intel_thread.so.2 => /mkl/libmkl_intel_thread.so.2 (0x0)",
                    f"lib{runtime}.so => /rt/lib{runtime}.so (0x0)")
                output = self.configure_shared_elpa(f"/rt/lib{runtime}.so")
                self.assertNotIn("check is incomplete", output)

    def test_mkl_rt_reports_incomplete_check(self):
        # mkl_rt selects its threading layer when the program runs.
        self.fake_ldd("libmkl_rt.so.2 => /mkl/libmkl_rt.so.2 (0x0)")
        output = self.configure_shared_elpa()
        self.assertIn("ELPA OpenMP dependency check is incomplete", output)

    def test_installed_shared_library_dependency_conflict(self):
        # Exercise FindELPA and the Linux inspection path with deterministic
        # ldd output, including a runtime from a transitive BLAS dependency.
        (self.tmp / "include/elpa").mkdir(parents=True)
        (self.tmp / "include/elpa/elpa.h").touch()
        (self.tmp / "lib").mkdir()
        library = self.tmp / "lib/libelpa.so"
        library.touch()
        self.fake_ldd("libiomp5.so => /intel/libiomp5.so (0x0)")
        output = self.configure(f'''
set(CMAKE_SYSTEM_NAME Linux)
set(ELPA_PATH "{self.tmp}")
set(ELPA_LIBRARY "{library}")
set(CMAKE_FIND_LIBRARY_SUFFIXES .so)
find_package(ELPA REQUIRED)
librpa_check_elpa_openmp(ELPA::ELPA)
''', success=False)
        self.assertIn("Prebuilt ELPA: incompatible OpenMP runtimes", output)

    def test_static_library_reports_incomplete_check(self):
        archive = self.tmp / "libelpa.a"
        archive.touch()
        output = self.configure(f'''
add_library(ELPA::ELPA UNKNOWN IMPORTED)
set_property(TARGET ELPA::ELPA PROPERTY IMPORTED_LOCATION "{archive}")
librpa_check_elpa_openmp(ELPA::ELPA)
''')
        self.assertIn("ELPA OpenMP dependency check is incomplete", output)

    @unittest.skipUnless(sys.platform.startswith("linux") and shutil.which("cc"),
                         "native Linux shared-library inspection")
    def test_real_transitive_shared_library_dependency(self):
        # The symbol names only keep the dependency alive under --as-needed.
        # These tiny libraries exercise ldd, not the OpenMP implementation.
        for name, source, dependencies in (
            ("iomp5", "int runtime_fixture(void) { return 0; }", []),
            ("blas", "int runtime_fixture(void); int blas_fixture(void) { return runtime_fixture(); }", ["-liomp5"]),
            ("elpa", "int blas_fixture(void); int elpa_fixture(void) { return blas_fixture(); }", ["-lblas"]),
        ):
            src = self.tmp / f"{name}.c"
            src.write_text(source)
            subprocess.run(
                ["cc", "-shared", "-fPIC", str(src), "-o", str(self.tmp / f"lib{name}.so"),
                 f"-L{self.tmp}", f"-Wl,-rpath,{self.tmp}", *dependencies],
                check=True, capture_output=True, text=True,
            )
        output = self.configure(f'''
set(CMAKE_SYSTEM_NAME Linux)
add_library(ELPA::ELPA SHARED IMPORTED)
set_property(TARGET ELPA::ELPA PROPERTY IMPORTED_LOCATION "{self.tmp / 'libelpa.so'}")
librpa_check_elpa_openmp(ELPA::ELPA)
''', success=False)
        self.assertIn("Prebuilt ELPA: incompatible OpenMP runtimes", output)

    def test_source_fortran_runtime_conflict(self):
        # Supply FindOpenMP results without requiring Intel compilers in CI.
        (self.tmp / "FindOpenMP.cmake").write_text(
            'set(OpenMP_C_LIBRARIES /gnu/libgomp.so)\n'
            'set(OpenMP_Fortran_LIBRARIES /intel/libiomp5.so)\n'
        )
        for settings in ("set(LIBRPA_ELPA_OPENMP ON)",
                         'set(LIBRPA_ELPA_CONFIGURE_ARGS "--enable-openmp")'):
            with self.subTest(settings=settings):
                output = self.configure(f'''
list(PREPEND CMAKE_MODULE_PATH "{self.tmp}")
{settings}
librpa_check_elpa_source_openmp()
''', success=False)
                self.assertIn("ELPA source build: incompatible OpenMP runtimes", output)

    def test_source_without_openmp_does_not_add_fortran_runtime(self):
        self.configure('''
set(LIBRPA_ELPA_OPENMP OFF)
set(OpenMP_Fortran_LIBRARIES /intel/libiomp5.so)
set(MKL_LIBRARIES /mkl/libmkl_gnu_thread.so)
librpa_check_elpa_source_openmp()
''')

    def test_custom_source_libs_conflict(self):
        output = self.configure('''
set(LIBRPA_ELPA_LIBS "-L/intel -lmkl_intel_thread")
librpa_check_elpa_source_openmp()
''', success=False)
        self.assertIn("ELPA source build: incompatible OpenMP runtimes", output)


if __name__ == "__main__":
    unittest.main()
