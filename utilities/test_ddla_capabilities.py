"""Portable compile/link checks for external LibDDLA capability selection."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "cmake/CheckLibDDLACapabilities.cmake"


def fixture_header(current=True):
    prefix = """#pragma once
#include <complex>
#include <cstddef>
#include <type_traits>
namespace ddla {
struct Handle;
using DdlaHandle_t = Handle*;
"""
    if not current:
        return prefix + "}\n"
    declarations = ["inline constexpr bool requires_cxx17 = std::is_same_v<int, int>;"]
    declarations += ["""
template<class T> inline void diag_logdet(const DdlaHandle_t&, int, const T*, int,
                                         std::complex<double>&) {}
template<class T> inline void pdiag_logdet(const DdlaHandle_t&, int, const T*, const int*,
                                          std::complex<double>&) {}
template<class R> std::size_t gemm_mixed_workspace_bytes(const DdlaHandle_t&, int, int, int, bool);
"""]
    z = "std::complex<R>"
    for a, b in (("R", z), (z, "R")):
        declarations.append(f"""
template<class R> inline void gemm(const DdlaHandle_t&, char, char, int, int, int, const {z}&,
    const {a}*, int, const {b}*, int, const {z}&, {z}*, int, void*, std::size_t) {{}}
template<class R> inline void gemmVbatched(const DdlaHandle_t&, char, char, int*, int*, int*, {z},
    const {a}* const*, int*, const {b}* const*, int*, {z}, {z}**, int*, int) {{}}
""")
        for external in ("R", z):
            declarations.append(f"""
template<class R> inline void gemmVbatched2s(const DdlaHandle_t&,
    char, char, int*, int*, int*, {z}, const {a}* const*, int*, const {b}* const*, int*,
    {z}, {z}**, int*, char, char, int*, int*, int*, {z}, const {external}* const*, int*, int*,
    {z}, {z}**, int*, bool, int, const int*, int) {{}}
""")
    return prefix + "\n".join(declarations) + "}\n"


@unittest.skipUnless(shutil.which("cmake"), "CMake is required")
class DDLACapabilitiesTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="librpa ddla capability ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.include = self.root / "include"
        (self.include / "ddla").mkdir(parents=True)
        self.header = self.include / "ddla/ddla.h"
        self.header.write_text(fixture_header())
        self.provider = self.root / "provider"
        self.provider.mkdir()
        (self.provider / "CMakeLists.txt").write_text(f"""
cmake_minimum_required(VERSION 3.18)
project(DDLAProvider CXX)
add_library(fixture STATIC fixture.cpp)
target_include_directories(fixture PUBLIC "{self.include}")
target_compile_features(fixture PUBLIC cxx_std_17)
set_target_properties(fixture PROPERTIES POSITION_INDEPENDENT_CODE ON)
file(GENERATE OUTPUT "${{CMAKE_BINARY_DIR}}/library-$<CONFIG>.txt" CONTENT "$<TARGET_FILE:fixture>")
""")
        self.consumer = self.root / "consumer"
        self.consumer.mkdir()

    def run_command(self, arguments, success=True):
        result = subprocess.run(arguments, capture_output=True, text=True)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def build_provider(self, symbol=True):
        source = '#include <ddla/ddla.h>\n'
        if symbol:
            source += """
namespace ddla {
template<class R> std::size_t gemm_mixed_workspace_bytes(const DdlaHandle_t&,int,int,int,bool)
{ return 0; }
template std::size_t gemm_mixed_workspace_bytes<double>(const DdlaHandle_t&,int,int,int,bool);
}
"""
        else:
            source += "void fixture_without_new_symbol() {}\n"
        (self.provider / "fixture.cpp").write_text(source)
        build = self.root / "provider-build"
        self.run_command(["cmake", "-S", str(self.provider), "-B", str(build)])
        self.run_command(["cmake", "--build", str(build), "--config", "Debug"])
        for path in build.glob("library-*.txt"):
            library = Path(path.read_text())
            if library.is_file():
                return library
        self.fail("Fixture archive was not built")

    def configure_consumer(self, library=None, source_target=False, managed=False):
        if source_target:
            target = "add_library(fixture INTERFACE)"
        else:
            target = f"""add_library(fixture STATIC IMPORTED)
set_target_properties(fixture PROPERTIES IMPORTED_LOCATION "{library}"
  INTERFACE_INCLUDE_DIRECTORIES "{self.include}")"""
        managed_setup = """add_custom_target(managed_ddla)
set(LIBRPA_BUNDLED_DDLA_PROJECT_TARGET managed_ddla)""" if managed else ""
        (self.consumer / "CMakeLists.txt").write_text(f"""
cmake_minimum_required(VERSION 3.18)
project(DDLAConsumer CXX)
{target}
add_library(ddla::ddla ALIAS fixture)
{managed_setup}
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
set(CMAKE_POSITION_INDEPENDENT_CODE OFF)
set(CMAKE_REQUIRED_DEFINITIONS -DPARENT_CHECK_STATE=1)
include("{MODULE}")
librpa_check_ddla_capabilities(ddla::ddla)
if(NOT CMAKE_CXX_STANDARD EQUAL 11 OR
   NOT CMAKE_TRY_COMPILE_TARGET_TYPE STREQUAL "STATIC_LIBRARY" OR
   CMAKE_POSITION_INDEPENDENT_CODE OR
   NOT CMAKE_REQUIRED_DEFINITIONS STREQUAL "-DPARENT_CHECK_STATE=1")
  message(FATAL_ERROR "Capability check changed caller state")
endif()
""")
        return self.run_command(
            ["cmake", "-S", str(self.consumer), "-B", str(self.root / "consumer-build")],
            success=False)

    def assert_actionable_failure(self, result):
        output = " ".join((result.stdout + result.stderr).split())
        self.assertNotEqual(result.returncode, 0, output)
        self.assertIn("lacks the required declarations or linkable APIs", output)
        self.assertIn("LIBDDLA_PATH", output)
        self.assertIn("managed dependency", output)

    def test_current_imported_alias_compiles_and_links_without_changing_parent(self):
        result = self.configure_consumer(self.build_provider())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_old_header_is_rejected(self):
        library = self.build_provider()
        self.header.write_text(fixture_header(current=False))
        self.assert_actionable_failure(self.configure_consumer(library))

    def test_current_header_with_old_library_is_rejected_at_link(self):
        self.assert_actionable_failure(self.configure_consumer(self.build_provider(symbol=False)))

    def test_reconfigure_does_not_reuse_a_stale_success(self):
        library = self.build_provider()
        result = self.configure_consumer(library)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.build_provider(symbol=False)
        self.assert_actionable_failure(self.configure_consumer(library))

    def test_caller_source_target_is_not_probed_before_build(self):
        self.header.write_text(fixture_header(current=False))
        result = self.configure_consumer(source_target=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("source target; checked by the consumer build", result.stdout)

    def test_managed_external_project_is_not_probed_before_build(self):
        result = self.configure_consumer(self.root / "not-built.a", managed=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("Performing Test _librpa_ddla_capable", result.stdout)


if __name__ == "__main__":
    unittest.main()
