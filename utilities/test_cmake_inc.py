"""Check the removed cmake.inc entry point before compiler discovery."""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("cmake"), "CMake is required")
class CMakeIncTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.source = self.root / "source"
        (self.source / "src").mkdir(parents=True)
        (self.source / "thirdparty").mkdir()
        shutil.copy2(ROOT / "CMakeLists.txt", self.source)
        shutil.copy2(ROOT / "src/version.h", self.source / "src")
        # Files that CMakeLists.txt includes before compiler discovery.
        shutil.copytree(ROOT / "cmake", self.source / "cmake")
        shutil.copy2(ROOT / "thirdparty/manifest.cmake", self.source / "thirdparty")

    def configure(self, name, *args):
        result = subprocess.run(
            ["cmake", "-S", str(self.source), "-B", str(self.root / name),
             f"-DCMAKE_CXX_COMPILER={self.root / 'missing-cxx'}",
             f"-DCMAKE_Fortran_COMPILER={self.root / 'missing-fortran'}", *args],
            capture_output=True, text=True,
        )
        output = result.stdout + result.stderr
        self.assertNotEqual(result.returncode, 0, output)
        self.assertNotIn("LEGACY_FILE_WAS_LOADED", output)
        return output

    def write_legacy_file(self):
        (self.source / "cmake.inc").write_text(
            'message(FATAL_ERROR "LEGACY_FILE_WAS_LOADED")\n'
        )

    def test_enabled_legacy_option_reports_migration(self):
        for has_cmake_inc in (False, True):
            with self.subTest(has_cmake_inc=has_cmake_inc):
                if has_cmake_inc:
                    self.write_legacy_file()
                output = self.configure(str(has_cmake_inc), "-DLIBRPA_USE_CMAKE_INC=ON")
                # CMake wraps the message text, so compare it without line breaks.
                text = " ".join(output.split())
                self.assertIn("LIBRPA_USE_CMAKE_INC is no longer supported", text)
                self.assertIn(
                    "Use -C <initial-cache.cmake> or -D<variable>=<value> instead", text
                )
                self.assertIn("-U LIBRPA_USE_CMAKE_INC", text)
                self.assertNotIn("CXX compiler identification", output)

    def test_default_and_old_off_cache_reach_compiler_discovery(self):
        self.write_legacy_file()
        cache = self.root / "initial-cache.cmake"
        cache.write_text('set(LIBRPA_USE_CMAKE_INC OFF CACHE BOOL "Old default")\n')
        for name, args in (("default", ()), ("old-off", ("-C", str(cache)))):
            with self.subTest(name=name):
                output = self.configure(name, *args)
                self.assertIn("CXX compiler identification", output)
                self.assertNotIn("LIBRPA_USE_CMAKE_INC is no longer supported", output)


if __name__ == "__main__":
    unittest.main()
