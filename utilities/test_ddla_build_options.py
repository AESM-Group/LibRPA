import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("cmake"), "CMake is required")
class DDLABuildOptionsTest(unittest.TestCase):
    def test_managed_backend_arguments(self):
        for backend in ("CPU", "CUDA", "HIP"):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source = root / "ddla"
                (source / "include/ddla").mkdir(parents=True)
                (source / "include/ddla/ddla.h").write_text("// fixture\n")
                (root / "CMakeLists.txt").write_text(f'''
cmake_minimum_required(VERSION 3.18)
project(DDLAFixture CXX)
include(GNUInstallDirs)
set(LIBRPA_USE_DDLA ON)
set(LIBRPA_USE_CUDA {"ON" if backend == "CUDA" else "OFF"})
set(LIBRPA_USE_HIP {"ON" if backend == "HIP" else "OFF"})
set(_libddla_source_dir "{source}")
include("{ROOT / 'cmake/BuildLibDDLA.cmake'}")
get_property(args TARGET librpa_bundled_libddla PROPERTY _EP_CMAKE_ARGS)
file(WRITE "${{CMAKE_BINARY_DIR}}/child-args.txt" "${{args}}")
''')
                build = root / "build"
                result = subprocess.run(["cmake", "-S", str(root), "-B", str(build)],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                args = (build / "child-args.txt").read_text().split(";")
                expected = {
                    "DDLA_USE_CUDA": backend == "CUDA",
                    "DDLA_USE_HIP": backend == "HIP",
                    "DDLA_USE_CCL": backend != "CPU",
                }
                for name, enabled in expected.items():
                    self.assertIn(f"-D{name}:BOOL={'ON' if enabled else 'OFF'}", args)
                self.assertFalse(any("DDLA_USE_GPU_CPU_TUNNEL" in arg for arg in args))
                self.assertIn(f"Managed LibDDLA backend         : {backend}", result.stdout)


if __name__ == "__main__":
    unittest.main()
