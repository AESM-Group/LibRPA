#!/usr/bin/env python3
"""Offline integration checks for managed dependency preparation (no compilers needed)."""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import unittest


MODULE = Path(__file__).resolve().parents[1] / "cmake/LibRPAThirdparty.cmake"


def fixture_environment():
    environment = os.environ.copy()
    for variable in ("LIBRPA_THIRDPARTY_ARCHIVE_DIR", "GREENX_SOURCE_DIR", "EXTERNAL_GREENX_DIR",
                     "ELPA_SOURCE_DIR", "EXTERNAL_ELPA_DIR"):
        environment.pop(variable, None)
    return environment


def tree_hash(root):
    lines = "".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root).as_posix()}\n"
        for path in sorted(root.rglob("*")) if path.is_file()
    )
    return hashlib.sha256(lines.encode()).hexdigest()


class ThirdpartyTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="librpa thirdparty ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.upstream = self.root / "upstream"
        self.header = self.upstream / "include/vendor/header.hpp"
        self.header.parent.mkdir(parents=True)
        self.header.write_text("// upstream\n")
        self.pristine = tree_hash(self.upstream / "include")
        self.header.write_text("// patched\n")
        self.patched = tree_hash(self.upstream / "include")
        self.header.write_text("// upstream\n")
        self.mirror = self.root / "mirror"
        self.mirror.mkdir()
        with tarfile.open(self.mirror / "fixture.tar.gz", "w:gz") as archive:
            archive.add(self.upstream, arcname="fixture")
        self.patch = self.root / "thirdparty/patches/fix.patch"
        self.patch.parent.mkdir(parents=True)
        self.patch.write_text(
            "--- a/include/vendor/header.hpp\n+++ b/include/vendor/header.hpp\n"
            "@@ -1 +1 @@\n-// upstream\n+// patched\n"
        )
        self.storage = self.root / "managed"
        self.final = self.storage / f"src/Fixture-{self.patched[:12]}"
        self.preamble = f'''
cmake_minimum_required(VERSION 3.13)
include("{MODULE}")
set(LIBRPA_SOURCE_DIR "{self.root}")
set(LIBRPA_THIRDPARTY_DIR "{self.storage}")
set(LIBRPA_THIRDPARTY_MIRRORS "{self.mirror.as_uri()}")
librpa_thirdparty_declare(Fixture VERSION 1 URL "file:///nonexistent/fixture.tar.gz"
  ARCHIVE fixture.tar.gz HASH_SUBDIR include PATCHES fix.patch
  PRISTINE_SHA256 {self.pristine} TREE_SHA256 {self.patched} LICENSE test)
'''

    def script(self, body):
        path = self.root / "run.cmake"
        path.write_text(self.preamble + body)
        return ["cmake", "-P", str(path)]

    def run_cmake(self, body, success=True, definitions=(), env=None):
        command = self.script(body)
        command[1:1] = list(definitions)
        result = subprocess.run(command, capture_output=True, text=True, env=fixture_environment() if env is None else env)
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode == 0, success, output)
        return output

    def test_elpa_mode_and_path_conflicts(self):
        for external, bundled in ((False, False), (True, False), (False, True), (True, True)):
            for source, prefix in ((False, False), (True, False), (False, True), (True, True)):
                for origin in ("cmake", "environment", "mixed"):
                    with self.subTest(external=external, bundled=bundled,
                                      source=source, prefix=prefix, origin=origin):
                        body = (f"set(LIBRPA_USE_EXTERNAL_ELPA {int(external)})\n"
                                f"set(LIBRPA_USE_BUNDLED_ELPA {int(bundled)})\n")
                        environment = fixture_environment()
                        for variable, enabled in (("ELPA_SOURCE_DIR", source), ("EXTERNAL_ELPA_DIR", prefix)):
                            if enabled:
                                if origin == "environment" or (origin == "mixed" and variable == "ELPA_SOURCE_DIR"):
                                    environment[variable] = str(self.root)
                                else:
                                    body += f'set({variable} "{self.root}")\n'
                        body += f'include("{MODULE.parent}/ValidateELPAOptions.cmake")\n'
                        conflict = ((external and bundled) or (source and prefix)
                                    or (bundled and prefix) or (external and source))
                        output = self.run_cmake(body, success=not conflict, env=environment)
                        if not external and not bundled and source != prefix:
                            option = "LIBRPA_USE_BUNDLED_ELPA" if source else "LIBRPA_USE_EXTERNAL_ELPA"
                            self.assertIn(f"is ignored unless {option}=ON", " ".join(output.split()))
                        self.assertNotIn("Downloading", output)

    def test_directory_symlink_hash(self):
        (self.upstream / "include/alias").symlink_to("vendor", target_is_directory=True)
        self.run_cmake(f'''librpa_thirdparty_tree_sha256(hash "{self.upstream}/include")
if(NOT hash STREQUAL "{self.pristine}")
  message(FATAL_ERROR "Directory symlink changed the file tree hash: ${{hash}}")
endif()
''')

    def test_prepare_repeat_and_tamper(self):
        body = "librpa_thirdparty_prepare(Fixture source)\n"
        self.run_cmake(body)
        self.assertEqual(tree_hash(self.final / "include"), self.patched)
        output = self.run_cmake("set(LIBRPA_THIRDPARTY_DOWNLOAD OFF)\n" + body)
        self.assertNotIn("Downloading", output)
        header = self.final / "include/vendor/header.hpp"
        header.write_text("// local change\n")
        output = self.run_cmake(body, success=False)
        self.assertIn("managed directory was modified", output)
        self.assertIn(self.patched, output)
        self.assertEqual(header.read_text(), "// local change\n")

    def test_overlay_preserves_user_tree(self):
        body = f'''librpa_thirdparty_use_path(Fixture "{self.upstream}/include" result status)
if(NOT status STREQUAL "overlay")
  message(FATAL_ERROR "Expected overlay, got ${{status}}")
endif()
librpa_thirdparty_use_path(Fixture "${{result}}" pinned status)
if(NOT status STREQUAL "pinned")
  message(FATAL_ERROR "Expected pinned, got ${{status}}")
endif()
'''
        self.run_cmake(body)
        self.assertEqual(tree_hash(self.upstream / "include"), self.pristine)
        self.assertFalse((self.storage / "downloads").exists())
        self.header.write_text("// developer version\n")
        output = self.run_cmake(f'''
librpa_thirdparty_use_path(Fixture "{self.upstream}/include" result status)
if(NOT status STREQUAL "unpinned")
  message(FATAL_ERROR "Expected unpinned, got ${{status}}")
endif()
''')
        self.assertIn("differs from the pinned version", " ".join(output.split()))

    def test_read_only_pristine_overlays(self):
        script = self.upstream / "configure"
        script.write_text("#!/bin/sh\nexit 0\n")
        script.chmod(0o755)
        for whole_tree in (False, True):
            with self.subTest(whole_tree=whole_tree):
                source = self.upstream if whole_tree else self.upstream / "include"
                pristine = tree_hash(source)
                self.header.write_text("// patched\n")
                patched = tree_hash(source)
                self.header.write_text("// upstream\n")
                subprocess.run(["chmod", "-R", "a-w", str(self.upstream)], check=True)
                modes = {p: p.stat().st_mode for p in self.upstream.rglob("*")}
                try:
                    output = self.run_cmake(f'''set_property(GLOBAL PROPERTY LIBRPA_THIRDPARTY_Fixture_HASH_SUBDIR "{'' if whole_tree else 'include'}")
set_property(GLOBAL PROPERTY LIBRPA_THIRDPARTY_Fixture_PRISTINE_SHA256 "{pristine}")
set_property(GLOBAL PROPERTY LIBRPA_THIRDPARTY_Fixture_TREE_SHA256 "{patched}")
librpa_thirdparty_use_path(Fixture "{source}" result status)
if(NOT status STREQUAL "overlay")
  message(FATAL_ERROR "Expected overlay")
endif()
''')
                    self.assertEqual(tree_hash(source), pristine)
                    self.assertEqual({p: p.stat().st_mode for p in modes}, modes)
                    result = self.storage / f"src/Fixture-{patched[:12]}"
                    if whole_tree:
                        self.assertTrue(os.access(result / "configure", os.X_OK))
                    else:
                        result /= "include"
                    self.assertEqual(tree_hash(result), patched)
                finally:
                    subprocess.run(["chmod", "-R", "u+w", str(self.upstream)], check=True)

    @unittest.skipIf(hasattr(os, "geteuid") and os.geteuid() == 0, "root can write read-only directories")
    def test_unwritable_overlay_directory(self):
        self.storage.mkdir()
        for blocked in (self.storage, self.storage / "src"):
            blocked.mkdir(exist_ok=True)
            blocked.chmod(0o555)
            try:
                output = self.run_cmake(
                    f'librpa_thirdparty_use_path(Fixture "{self.upstream}/include" result status)',
                    success=False)
                self.assertIn(str(self.storage / "src"), " ".join(output.split()))
                self.assertIn("An overlay needs a writable LIBRPA_THIRDPARTY_DIR", " ".join(output.split()))
            finally:
                blocked.chmod(0o755)

    def test_whole_tree_overlay_preserves_symlink(self):
        (self.upstream / "alias").symlink_to("include", target_is_directory=True)
        pristine = tree_hash(self.upstream)
        self.header.write_text("// patched\n")
        patched = tree_hash(self.upstream)
        self.header.write_text("// upstream\n")
        body = f'''set_property(GLOBAL PROPERTY LIBRPA_THIRDPARTY_Fixture_HASH_SUBDIR "")
set_property(GLOBAL PROPERTY LIBRPA_THIRDPARTY_Fixture_PRISTINE_SHA256 "{pristine}")
set_property(GLOBAL PROPERTY LIBRPA_THIRDPARTY_Fixture_TREE_SHA256 "{patched}")
librpa_thirdparty_use_path(Fixture "{self.upstream}" result status)
if(NOT status STREQUAL "overlay" OR NOT IS_SYMLINK "${{result}}/alias")
  message(FATAL_ERROR "Whole-tree overlay did not preserve the symlink")
endif()
librpa_thirdparty_use_path(Fixture "${{result}}" pinned status)
if(NOT status STREQUAL "pinned" OR NOT pinned STREQUAL result)
  message(FATAL_ERROR "Expected pinned whole-tree source")
endif()
'''
        self.run_cmake(body)
        self.assertEqual(tree_hash(self.upstream), pristine)
        result = self.storage / f"src/Fixture-{patched[:12]}"
        self.assertTrue((result / "alias").is_symlink())
        self.assertEqual(os.readlink(result / "alias"), "include")
        self.assertEqual(tree_hash(result), patched)

    def test_offline_archive_and_failed_patch(self):
        body = "set(LIBRPA_THIRDPARTY_DOWNLOAD OFF)\nlibrpa_thirdparty_prepare(Fixture source)\n"
        output = self.run_cmake(body, success=False)
        self.assertIn("Missing third-party component Fixture", output)
        self.assertIn("LibRPAPrepareThirdparty.cmake", output)
        self.assertIn("Alternatively", output)
        (self.storage / "downloads").mkdir()
        shutil.copy2(self.mirror / "fixture.tar.gz", self.storage / "downloads")
        self.patch.write_text(self.patch.read_text().replace("-// upstream", "-// bad context"))
        output = self.run_cmake(body, success=False)
        self.assertIn("patch failed", output)
        self.assertFalse(list((self.storage / "src").glob(".tmp-*")))
        self.assertFalse(self.final.exists())

    def test_separate_archive_directory_and_precedence(self):
        # Only the selected directory has the archive, and downloads are disabled.
        body = "set(LIBRPA_THIRDPARTY_DOWNLOAD OFF)\nlibrpa_thirdparty_prepare(Fixture source)\n"
        environment = dict(fixture_environment(), LIBRPA_THIRDPARTY_ARCHIVE_DIR=str(self.root / "wrong"))
        output = self.run_cmake(body, definitions=[f"-DLIBRPA_THIRDPARTY_ARCHIVE_DIR={self.mirror}"],
                                env=environment)
        self.assertIn(str(self.mirror), output)
        self.assertTrue(self.final.is_dir())
        self.assertFalse((self.storage / "downloads").exists())
        shutil.rmtree(self.storage)
        environment["LIBRPA_THIRDPARTY_ARCHIVE_DIR"] = str(self.mirror)
        self.run_cmake(body, env=environment)
        self.assertTrue(self.final.is_dir())

    def test_nonempty_relative_archive_path(self):
        environment = dict(fixture_environment(), LIBRPA_THIRDPARTY_ARCHIVE_DIR=str(self.mirror))
        self.run_cmake('''librpa_thirdparty_archive_dir(resolved)
get_filename_component(expected "OFF" ABSOLUTE)
if(NOT resolved STREQUAL expected)
  message(FATAL_ERROR "A nonempty path was treated as a boolean: ${resolved}")
endif()
''', definitions=["-DLIBRPA_THIRDPARTY_ARCHIVE_DIR:PATH=OFF"], env=environment)

    def test_shared_archive_reuse(self):
        archives = self.root / "shared archives"
        body = f'set(LIBRPA_THIRDPARTY_ARCHIVE_DIR "{archives}")\nlibrpa_thirdparty_prepare(Fixture source)\n'
        self.run_cmake(body)
        archive = archives / "fixture.tar.gz"
        timestamp = archive.stat().st_mtime_ns
        self.storage = self.root / "second managed"
        output = self.run_cmake(f'set(LIBRPA_THIRDPARTY_DIR "{self.storage}")\n' + body)
        self.assertNotIn("Downloading", output)
        self.assertEqual(archive.stat().st_mtime_ns, timestamp)
        self.assertTrue((self.storage / self.final.relative_to(self.final.parents[1])).is_dir())

    def test_archive_default_tracks_managed_directory(self):
        self.run_cmake(f'''librpa_thirdparty_archive_dir(first)
set(LIBRPA_THIRDPARTY_DIR "{self.root}/changed")
librpa_thirdparty_archive_dir(second)
if(NOT second STREQUAL "{self.root}/changed/downloads" OR first STREQUAL second)
  message(FATAL_ERROR "Archive default did not follow the managed directory")
endif()
''')

    @unittest.skipIf(hasattr(os, "geteuid") and os.geteuid() == 0, "root can write read-only directories")
    def test_unwritable_archive_directory(self):
        archives = self.root / "read only"
        archives.mkdir()
        archives.chmod(0o555)
        self.addCleanup(archives.chmod, 0o755)
        output = self.run_cmake(f'set(LIBRPA_THIRDPARTY_ARCHIVE_DIR "{archives}")\nlibrpa_thirdparty_prepare(Fixture source)\n',
                                success=False)
        self.assertIn(str(archives), " ".join(output.split()))
        self.assertIn("Use a writable", output)
        self.assertIn("prefetch", output)

    def test_corrupt_shared_archive_preserved(self):
        archive = self.mirror / "fixture.tar.gz"
        archive.write_text("not a tarball")
        body = f'set(LIBRPA_THIRDPARTY_ARCHIVE_DIR "{self.mirror}")\nlibrpa_thirdparty_prepare(Fixture source)\n'
        output = self.run_cmake(body, success=False)
        self.assertIn(str(archive), " ".join(output.split()))
        self.assertIn("delete it and configure again", " ".join(output.split()))
        self.assertEqual(archive.read_text(), "not a tarball")
        self.assertFalse(list((self.storage / "src").glob(".tmp-*")))
        # An extractable archive with incorrect content must also name the archive.
        self.header.write_text("// upstream\n// unexpected content\n")
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(self.upstream, arcname="fixture")
        output = self.run_cmake(body.replace("librpa_thirdparty_prepare", 'set_property(GLOBAL PROPERTY LIBRPA_THIRDPARTY_Fixture_PATCHES "")\nlibrpa_thirdparty_prepare'), success=False)
        self.assertIn("prepared tree hash mismatch", output)
        self.assertIn(str(archive), " ".join(output.split()))
        self.assertTrue(archive.is_file())

    def test_concurrent_preparation(self):
        command = self.script("librpa_thirdparty_prepare(Fixture source)\n")
        children = [subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     text=True, env=fixture_environment()) for _ in range(2)]
        for child in children:
            output, error = child.communicate(timeout=30)
            self.assertEqual(child.returncode, 0, output + error)
        self.assertEqual(tree_hash(self.final / "include"), self.patched)
        self.assertFalse(list((self.storage / "src").glob(".tmp-*")))


class ELPAPathTest(unittest.TestCase):
    def test_source_and_build_whitespace_rejected_at_configure(self):
        with tempfile.TemporaryDirectory(prefix="librpa-elpa-") as temporary:
            root = Path(temporary)
            for kind in ("source", "build"):
                for whitespace in (" ", "\t"):
                    with self.subTest(kind=kind, whitespace=repr(whitespace)):
                        source = root / ("source" + whitespace + "tree" if kind == "source" else "source")
                        build = root / ("build" + whitespace + "tree" if kind == "build" else "build")
                        (root / "CMakeLists.txt").write_text(f'''cmake_minimum_required(VERSION 3.18)
project(ELPAPath NONE)
set(_elpa_source_dir "{source}")
set(LIBRPA_BUNDLED_ELPA_VERSION 2026.02.001)
include("{MODULE.parent.parent}/thirdparty/ELPA/CMakeLists.txt")
''')
                        result = subprocess.run(["cmake", "-S", str(root), "-B", str(build)],
                                                capture_output=True, text=True, env=fixture_environment())
                        self.assertNotEqual(result.returncode, 0)
                        output = " ".join((result.stdout + result.stderr).split())
                        self.assertIn("ELPA source builds do not support whitespace", output)
                        self.assertIn(" ".join(str(source if kind == "source" else build).split()), output)
                        self.assertIn("LIBRPA_THIRDPARTY_DIR", output)


class GreenXResolutionTest(unittest.TestCase):
    """Exercise target and package resolution without a compiler or network."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="librpa greenx ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prefix = self.root / "installation"
        self.modules = self.prefix / "include/greenX/GNU-13.4.0"
        for component, module in (("TimeFrequency", "gx_minimax"), ("common", "kinds")):
            directory = self.modules / component / "modules"
            directory.mkdir(parents=True)
            (directory / (module + ".mod")).touch()
        config = self.prefix / "lib/cmake/greenX/greenXConfig.cmake"
        config.parent.mkdir(parents=True)
        config.write_text(f'''add_library(greenX::GXCommon INTERFACE IMPORTED)
set_target_properties(greenX::GXCommon PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "{self.modules}/common/modules")
add_library(greenX::LibGXMiniMax INTERFACE IMPORTED)
set_target_properties(greenX::LibGXMiniMax PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "{self.modules}/TimeFrequency/modules"
  INTERFACE_LINK_LIBRARIES greenX::GXCommon)
''')

    def configure(self, body, success=True):
        (self.root / "CMakeLists.txt").write_text(f'''cmake_minimum_required(VERSION 3.18)
project(Resolution NONE)
set(LIBRPA_THIRDPARTY_DIR "{self.root}/managed" CACHE PATH "")
include("{MODULE}")
set(CMAKE_Fortran_COMPILER_ID GNU)
set(CMAKE_Fortran_COMPILER_VERSION 13.4.0)
{body}
include("{MODULE.parent}/ResolveGreenX.cmake")
message(STATUS "Summary: ${{LIBRPA_THIRDPARTY_SUMMARY}}")
''')
        result = subprocess.run(["cmake", "-S", str(self.root), "-B", str(self.root / "build")],
                                capture_output=True, text=True, env=fixture_environment())
        output = " ".join((result.stdout + result.stderr).split())
        self.assertEqual(result.returncode == 0, success, output)
        return output

    def test_managed_source_keeps_parent_install_prefix(self):
        source = self.root / "greenx"
        source.mkdir()
        (source / "CMakeLists.txt").write_text('''project(greenX NONE)
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${PROJECT_SOURCE_DIR}/install" CACHE PATH "" FORCE)
endif()
add_library(LibGXMiniMax INTERFACE)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt" DESTINATION share/greenx)
''')
        digest = tree_hash(source)
        managed = self.root / f"managed/src/greenX-{digest[:12]}"
        shutil.copytree(source, managed)
        body = f'''librpa_thirdparty_declare(greenX VERSION 1234567 URL unused ARCHIVE unused
  TREE_SHA256 {digest} LICENSE test)
'''
        self.configure(body)
        cache = (self.root / "build/CMakeCache.txt").read_text()
        prefix = next(line.split("=", 1)[1] for line in cache.splitlines()
                      if line.startswith("CMAKE_INSTALL_PREFIX:"))
        self.assertFalse(Path(prefix).is_relative_to(self.root / "managed"))
        subprocess.run(["cmake", "--install", str(self.root / "build"),
                        "--prefix", str(self.root / "install")], check=True, capture_output=True)
        self.assertTrue((self.root / "install/share/greenx/CMakeLists.txt").is_file())
        self.configure(body)
        self.assertEqual(tree_hash(managed), digest)

    def test_parent_target_with_external_option_off(self):
        for target in ("LibGXMiniMax", "greenX::LibGXMiniMax"):
            with self.subTest(target=target):
                output = self.configure(f'''set(LIBRPA_USE_EXTERNAL_GREENX OFF)
add_library({target} INTERFACE IMPORTED)
''')
                self.assertIn("greenX=target", output)
                self.assertNotIn("Downloading", output)
                self.assertNotIn("prebuilt installation", output)

    def test_installed_transitive_modules_and_compiler_warning(self):
        body = f'set(EXTERNAL_GREENX_DIR "{self.prefix}")'
        output = self.configure(body)
        self.assertIn(f"greenX=installed({self.prefix})", output)
        self.assertIn("using a prebuilt installation", output)
        self.assertNotIn("installed Fortran modules were built", output)
        output = self.configure(body + '\nset(CMAKE_Fortran_COMPILER_VERSION 14.1.0)')
        self.assertIn("installed Fortran modules were built with GNU-13.4.0", output)
        self.assertIn("LibRPA uses GNU-14.1.0", output)
        (self.modules / "common/modules/kinds.mod").unlink()
        output = self.configure(body, success=False)
        self.assertIn("missing kinds.mod", output)

    def test_ambiguous_or_missing_external_source(self):
        output = self.configure('set(GREENX_SOURCE_DIR /source)\nset(EXTERNAL_GREENX_DIR /prefix)',
                                success=False)
        self.assertIn("mutually exclusive", output)
        output = self.configure('set(LIBRPA_USE_EXTERNAL_GREENX ON)', success=False)
        self.assertIn("LIBRPA_USE_EXTERNAL_GREENX requires", output)


if __name__ == "__main__":
    unittest.main()
