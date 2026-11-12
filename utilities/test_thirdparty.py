#!/usr/bin/env python3
"""Offline integration checks for managed dependency preparation and include selection.

The include-selection check compiles a small probe project. LibRPA itself is not built.
"""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import unittest


PATH_VARIABLES = ("LIBRI_INCLUDE_DIR", "LIBCOMM_INCLUDE_DIR", "CEREAL_INCLUDE_DIR",
                  "GREENX_SOURCE_DIR", "GREENX_PATH", "EXTERNAL_GREENX_DIR", "ELPA_SOURCE_DIR",
                  "ELPA_PATH", "EXTERNAL_ELPA_DIR", "ELPA_ROOT", "LIBDDLA_SOURCE_DIR", "LIBDDLA_PATH",
                  "LIBRPA_THIRDPARTY_ARCHIVE_DIR")

MODULE = Path(__file__).resolve().parents[1] / "cmake/LibRPAThirdparty.cmake"


def copy_helper(checkout):
    module = checkout / "cmake" / MODULE.name
    module.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(MODULE, module)
    return module


def fixture_environment():
    environment = os.environ.copy()
    for variable in PATH_VARIABLES:
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
        self.archives = self.root / "archives"
        self.archives.mkdir()
        with tarfile.open(self.archives / "fixture.tar.gz", "w:gz") as archive:
            archive.add(self.upstream, arcname="fixture")
        self.checkout = self.root / "LibRPA"
        self.module = copy_helper(self.checkout)
        self.patch = self.checkout / "thirdparty/patches/fix.patch"
        self.patch.parent.mkdir(parents=True)
        self.patch.write_text(
            "--- a/include/vendor/header.hpp\n+++ b/include/vendor/header.hpp\n"
            "@@ -1 +1 @@\n-// upstream\n+// patched\n"
        )
        self.storage = self.checkout / "thirdparty"
        self.final = self.storage / f"src/Fixture-{self.patched[:12]}"
        self.preamble = f'''
cmake_minimum_required(VERSION 3.13)
include("{self.module}")
librpa_thirdparty_declare(Fixture VERSION 1 URL "{(self.archives / "fixture.tar.gz").as_uri()}"
  ARCHIVE fixture.tar.gz ARCHIVE_SHA256 {hashlib.sha256((self.archives / "fixture.tar.gz").read_bytes()).hexdigest()} HASH_SUBDIR include PATCHES fix.patch
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

    def test_elpa_selection(self):
        for enabled in (False, True):
            for source in (False, True):
                for prefix in (None, "ELPA_PATH", "EXTERNAL_ELPA_DIR", "ELPA_ROOT"):
                    for origin in ("cmake", "environment", "mixed"):
                        with self.subTest(enabled=enabled, source=source,
                                          prefix=prefix, origin=origin):
                            body = f"set(LIBRPA_USE_ELPA {int(enabled)})\n"
                            environment = fixture_environment()
                            for variable in (["ELPA_SOURCE_DIR"] if source else []) + ([prefix] if prefix else []):
                                if origin == "environment" or (origin == "mixed" and variable == "ELPA_SOURCE_DIR"):
                                    environment[variable] = str(self.root)
                                else:
                                    body += f'set({variable} "{self.root}")\n'
                            body += f'include("{MODULE.parent}/ValidateELPAOptions.cmake")\n'
                            body += 'message(STATUS "ELPA=${LIBRPA_USE_ELPA}, mode=${_elpa_mode}")\n'
                            output = self.run_cmake(body, success=not (source and prefix), env=environment)
                            if not (source and prefix):
                                mode = "installed" if prefix else "source" if source or enabled else ""
                                state = "ON" if mode else "0"
                                self.assertIn(f"ELPA={state}, mode={mode}", output)
                                if prefix == "EXTERNAL_ELPA_DIR":
                                    self.assertIn("EXTERNAL_ELPA_DIR is deprecated", output)
                                else:
                                    self.assertNotRegex(output, r"CMake (Deprecation Warning|Warning \(deprecated\))")
                            self.assertNotIn("Downloading", output)

    def test_elpa_deprecated_options(self):
        for option in ("LIBRPA_USE_EXTERNAL_ELPA", "LIBRPA_USE_BUNDLED_ELPA"):
            for enabled in (False, True):
                with self.subTest(option=option, enabled=enabled):
                    output = self.run_cmake(f'''set(LIBRPA_USE_ELPA OFF)
include("{MODULE.parent}/ValidateELPAOptions.cmake")
message(STATUS "ELPA=${{LIBRPA_USE_ELPA}}, mode=${{_elpa_mode}}")
''', definitions=(f"-D{option}={'ON' if enabled else 'OFF'}",))
                    self.assertRegex(output, r"CMake (Deprecation Warning|Warning \(deprecated\))")
                    self.assertIn(f"{option} is deprecated", output)
                    mode = ("installed" if option == "LIBRPA_USE_EXTERNAL_ELPA" else "source") if enabled else ""
                    self.assertIn(f"ELPA={'ON' if enabled else 'OFF'}, mode={mode}", output)

    def test_elpa_parent_target_and_reconfigure(self):
        project = self.root / "parent"
        project.mkdir()
        (project / "CMakeLists.txt").write_text(f'''
cmake_minimum_required(VERSION 3.18)
project(Parent NONE)
include("{self.module}")
option(LIBRPA_USE_ELPA "Use ELPA" OFF)
add_library(ELPA::ELPA INTERFACE IMPORTED)
include("{MODULE.parent}/ValidateELPAOptions.cmake")
if(NOT LIBRPA_USE_ELPA OR NOT _elpa_mode STREQUAL "target")
  message(FATAL_ERROR "Parent target must enable ELPA and take precedence over paths")
endif()
''')
        build = self.root / "parent-build"
        for path_option in (None, "ELPA_SOURCE_DIR", "ELPA_PATH"):
            with self.subTest(path=path_option):
                command = ["cmake", "-S", str(project), "-B", str(build), "-DLIBRPA_USE_ELPA=OFF",
                           "-DELPA_SOURCE_DIR=", "-DELPA_PATH="]
                if path_option:
                    command.append(f"-D{path_option}={self.root}")
                result = subprocess.run(command, capture_output=True, text=True, env=fixture_environment())
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertNotRegex(result.stderr, r"CMake (Deprecation Warning|Warning \(deprecated\))")
                self.assertIn("LIBRPA_USE_ELPA:BOOL=ON", (build / "CMakeCache.txt").read_text())
                result = subprocess.run(["cmake", "-S", str(project), "-B", str(build)],
                                        capture_output=True, text=True, env=fixture_environment())
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertNotRegex(result.stderr, r"CMake (Deprecation Warning|Warning \(deprecated\))")

    def test_directory_symlink_hash(self):
        (self.upstream / "include/alias").symlink_to("vendor", target_is_directory=True)
        self.run_cmake(f'''librpa_thirdparty_tree_sha256(hash "{self.upstream}/include")
if(NOT hash STREQUAL "{self.pristine}")
  message(FATAL_ERROR "Directory symlink changed the file tree hash: ${{hash}}")
endif()
''')

    def test_root_git_metadata_ignored_in_hash(self):
        # A submodule has a .git file and a clone has a .git directory.
        # Neither at the root of the hashed tree may change the hash.
        for kind in ("file", "dir"):
            with self.subTest(kind=kind):
                git = self.upstream / ".git"
                if kind == "file":
                    git.write_text("gitdir: ../.git/modules/upstream\n")
                else:
                    (git / "objects").mkdir(parents=True)
                    (git / "HEAD").write_text("ref: refs/heads/main\n")
                self.run_cmake(f'''librpa_thirdparty_tree_sha256(hash "{self.upstream}")
librpa_thirdparty_tree_sha256(reference "{self.upstream}/include")
file(WRITE "{self.root}/hash.txt" "${{hash}}\n${{reference}}")
''')
                shutil.rmtree(git) if kind == "dir" else git.unlink()
                lines = (self.root / "hash.txt").read_text().splitlines()
                self.assertEqual(lines[0], tree_hash(self.upstream))
                self.assertEqual(lines[1], self.pristine)

    def test_nested_git_metadata_changes_hash(self):
        nested = self.upstream / "include/vendor/.git"
        nested.write_text("gitdir: elsewhere\n")
        self.run_cmake(f'''librpa_thirdparty_tree_sha256(hash "{self.upstream}/include")
if(hash STREQUAL "{self.pristine}")
  message(FATAL_ERROR "A nested .git file must change the hash")
endif()
''')

    def test_prepare_repeat_and_local_edits(self):
        body = "librpa_thirdparty_prepare(Fixture source)\n"
        self.run_cmake(body)
        self.assertEqual(tree_hash(self.final / "include"), self.patched)
        output = self.run_cmake("set(LIBRPA_THIRDPARTY_DOWNLOAD OFF)\n" + body)
        self.assertNotIn("Downloading", output)
        header = self.final / "include/vendor/header.hpp"
        header.write_text("// local change\n")
        output = self.run_cmake("set(LIBRPA_THIRDPARTY_DOWNLOAD OFF)\n" + body + '''
librpa_thirdparty_prepare(Fixture source status)
if(NOT status STREQUAL "unpinned")
  message(FATAL_ERROR "Modified managed source must be unpinned")
endif()
''')
        self.assertIn("managed source differs from the pinned tree", " ".join(output.split()))
        self.assertNotIn("Downloading", output)
        self.assertIn(self.patched, output)
        self.assertEqual(header.read_text(), "// local change\n")

    def test_added_and_removed_managed_files(self):
        body = 'librpa_thirdparty_prepare(Fixture source status)\n'
        self.run_cmake(body)
        added = self.final / "include/local.hpp"
        added.write_text("// local addition\n")
        output = self.run_cmake(body)
        self.assertIn("unpinned", output)
        self.assertTrue(added.exists())
        added.unlink()
        output = self.run_cmake(body)
        self.assertNotIn("unpinned", output)
        header = self.final / "include/vendor/header.hpp"
        header.unlink()
        output = self.run_cmake(body)
        self.assertIn("unpinned", output)
        self.assertFalse(header.exists())

    def test_modified_overlay_reused_by_path_and_header_resolution(self):
        body = f'librpa_thirdparty_use_path(Fixture "{self.upstream}/include" result status)\n'
        self.run_cmake(body)
        header = self.final / "include/vendor/header.hpp"
        header.write_text("// edited overlay\n")
        # Header targets need configure mode rather than cmake -P.
        project = self.root / "project"
        project.mkdir()
        (project / "CMakeLists.txt").write_text(self.preamble + 'project(Headers NONE)\n' + body + '''
if(NOT status STREQUAL "unpinned")
  message(FATAL_ERROR "Modified overlay must be unpinned")
endif()
librpa_thirdparty_resolve_headers(Fixture Fixture::Fixture UNUSED_PATH
  vendor/header.hpp resolved resolved_status)
if(NOT resolved_status STREQUAL "unpinned" OR NOT resolved STREQUAL result)
  message(FATAL_ERROR "Header resolution lost modified source status or path")
endif()
''')
        result = subprocess.run(["cmake", "-S", str(project), "-B", str(self.root / "build")],
                                capture_output=True, text=True, env=fixture_environment())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(header.read_text(), "// edited overlay\n")
        self.assertEqual(tree_hash(self.upstream / "include"), self.pristine)

    def test_prepared_tree_difference_warns(self):
        self.patch.write_text(self.patch.read_text().replace("+// patched", "+// revised patch"))
        output = self.run_cmake("librpa_thirdparty_prepare(Fixture source)\n")
        self.assertIn("unpinned", output)
        self.assertEqual((self.final / "include/vendor/header.hpp").read_text(), "// revised patch\n")

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
        self.storage.mkdir(exist_ok=True)
        for blocked in (self.storage, self.storage / "src"):
            blocked.mkdir(exist_ok=True)
            blocked.chmod(0o555)
            try:
                output = self.run_cmake(
                    f'librpa_thirdparty_use_path(Fixture "{self.upstream}/include" result status)',
                    success=False)
                self.assertIn(str(self.storage / "src"), " ".join(output.split()))
                self.assertIn("An overlay needs a writable thirdparty/src directory", " ".join(output.split()))
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

    def test_failed_download_cleanup_and_retry(self):
        missing = (self.root / "missing.tar.gz").as_uri()
        output = self.run_cmake(f'''set_property(GLOBAL PROPERTY LIBRPA_THIRDPARTY_Fixture_URL "{missing}")
librpa_thirdparty_prepare(Fixture source)
''', success=False)
        self.assertIn("Missing third-party component Fixture", output)
        self.assertIn(missing, output)
        self.assertFalse(list((self.storage / "downloads").iterdir()))
        self.assertFalse(self.final.exists())
        self.run_cmake("librpa_thirdparty_prepare(Fixture source)\n")
        self.assertEqual(tree_hash(self.final / "include"), self.patched)
        self.assertFalse(list((self.storage / "downloads").glob("*.tmp-*")))

    def test_offline_archive_and_failed_patch(self):
        body = "set(LIBRPA_THIRDPARTY_DOWNLOAD OFF)\nlibrpa_thirdparty_prepare(Fixture source)\n"
        output = self.run_cmake(body, success=False)
        self.assertIn("Missing third-party component Fixture", output)
        self.assertIn("LibRPADownloadThirdparty.cmake", output)
        self.assertIn("Alternatively", output)
        (self.storage / "downloads").mkdir()
        shutil.copy2(self.archives / "fixture.tar.gz", self.storage / "downloads")
        self.patch.write_text(self.patch.read_text().replace("-// upstream", "-// bad context"))
        output = self.run_cmake(body, success=False)
        self.assertIn("patch failed", output)
        self.assertFalse(list((self.storage / "src").glob(".tmp-*")))
        self.assertFalse(self.final.exists())

    def test_download_hash_mismatch_cleanup(self):
        bad_archive = self.root / "bad.tar.gz"
        # Same extracted files, different archive bytes: the archive hash controls acceptance.
        with tarfile.open(bad_archive, "w") as archive:
            archive.add(self.upstream, arcname="fixture")
        output = self.run_cmake(f'''set_property(GLOBAL PROPERTY LIBRPA_THIRDPARTY_Fixture_URL "{bad_archive.as_uri()}")
librpa_thirdparty_prepare(Fixture source)
''', success=False)
        self.assertIn("archive SHA256 mismatch", output)
        self.assertFalse(list((self.storage / "downloads").iterdir()))
        self.assertFalse(self.final.exists())
        self.run_cmake("librpa_thirdparty_prepare(Fixture source)\n")
        self.assertEqual(tree_hash(self.final / "include"), self.patched)
        self.assertEqual((self.storage / "downloads/fixture.tar.gz").read_bytes(),
                         (self.archives / "fixture.tar.gz").read_bytes())
        self.assertFalse(list((self.storage / "downloads").glob("*.tmp-*")))

    def test_separate_archive_directory_and_precedence(self):
        # Only the selected directory has the archive, and downloads are disabled.
        body = "set(LIBRPA_THIRDPARTY_DOWNLOAD OFF)\nlibrpa_thirdparty_prepare(Fixture source)\n"
        environment = dict(fixture_environment(), LIBRPA_THIRDPARTY_ARCHIVE_DIR=str(self.root / "wrong"))
        output = self.run_cmake(body, definitions=[f"-DLIBRPA_THIRDPARTY_ARCHIVE_DIR={self.archives}"],
                                env=environment)
        self.assertIn(str(self.archives), output)
        self.assertTrue(self.final.is_dir())
        self.assertFalse((self.storage / "downloads").exists())
        shutil.rmtree(self.storage / "src")
        environment["LIBRPA_THIRDPARTY_ARCHIVE_DIR"] = str(self.archives)
        self.run_cmake(body, env=environment)
        self.assertTrue(self.final.is_dir())

    def test_nonempty_relative_archive_path(self):
        environment = dict(fixture_environment(), LIBRPA_THIRDPARTY_ARCHIVE_DIR=str(self.archives))
        output = self.run_cmake("librpa_thirdparty_archive_dir(resolved)", success=False,
                                definitions=["-DLIBRPA_THIRDPARTY_ARCHIVE_DIR:PATH=OFF"], env=environment)
        self.assertIn("Supply an absolute path", " ".join(output.split()))

    def test_shared_archive_reuse(self):
        archives = self.root / "shared archives"
        body = f'set(LIBRPA_THIRDPARTY_ARCHIVE_DIR "{archives}")\nlibrpa_thirdparty_prepare(Fixture source)\n'
        self.run_cmake(body)
        archive = archives / "fixture.tar.gz"
        timestamp = archive.stat().st_mtime_ns
        second = self.root / "second checkout"
        module = copy_helper(second)
        shutil.copytree(self.checkout / "thirdparty/patches", second / "thirdparty/patches")
        self.preamble = self.preamble.replace(str(self.module), str(module))
        output = self.run_cmake("set(LIBRPA_THIRDPARTY_DOWNLOAD OFF)\n" + body)
        self.assertNotIn("Downloading", output)
        self.assertEqual(archive.stat().st_mtime_ns, timestamp)
        self.assertTrue((second / "thirdparty/src" / self.final.name).is_dir())
        self.assertTrue(self.final.is_dir())

    def test_archive_default_is_checkout_downloads(self):
        self.run_cmake(f'''librpa_thirdparty_archive_dir(archives)
if(NOT archives STREQUAL "{self.checkout}/thirdparty/downloads")
  message(FATAL_ERROR "Unexpected archive default: ${{archives}}")
endif()
''')

    def test_parent_project_uses_checkout_sources(self):
        parent = self.root / "parent"
        parent.mkdir()
        (parent / "CMakeLists.txt").write_text(f'''cmake_minimum_required(VERSION 3.18)
project(Parent NONE)
add_subdirectory("{self.checkout}" dependencies)
''')
        (self.checkout / "CMakeLists.txt").write_text(self.preamble + f'''
librpa_thirdparty_prepare(Fixture source)
if(NOT source STREQUAL "{self.final}")
  message(FATAL_ERROR "Source directory is outside the LibRPA checkout: ${{source}}")
endif()
''')
        result = subprocess.run(["cmake", "-S", str(parent), "-B", str(self.root / "build"),
                                 f"-DLIBRPA_THIRDPARTY_ARCHIVE_DIR={self.archives}",
                                 "-DLIBRPA_THIRDPARTY_DOWNLOAD=OFF"], cwd=self.root,
                                capture_output=True, text=True, env=fixture_environment())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(tree_hash(self.final / "include"), self.patched)
        self.assertFalse((parent / "thirdparty").exists())
        self.assertFalse((self.root / "build/thirdparty").exists())

    def test_offline_archive_transfer(self):
        # The connected and offline machines have independent archive locations.
        downloads = self.root / "connected Downloads"
        script = self.standalone_script("LibRPADownloadThirdparty.cmake")
        result = subprocess.run(["cmake", f"-DLIBRPA_THIRDPARTY_ARCHIVE_DIR={downloads}", "-P", str(script)],
                                capture_output=True, text=True, env=fixture_environment())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse((self.storage / "src").exists())
        original_preamble = self.preamble
        for custom in (False, True):
            with self.subTest(custom=custom):
                checkout = self.root / f"offline-{custom}" / "LibRPA"
                module = copy_helper(checkout)
                shutil.copytree(self.checkout / "thirdparty/patches", checkout / "thirdparty/patches")
                archives = checkout.parent / "Downloads" if custom else checkout / "thirdparty/downloads"
                archives.mkdir(parents=True, exist_ok=True)
                shutil.copy2(downloads / "fixture.tar.gz", archives)
                self.preamble = original_preamble.replace(str(self.module), str(module))
                body = 'set(LIBRPA_THIRDPARTY_DOWNLOAD OFF)\n'
                if custom:
                    body += f'set(LIBRPA_THIRDPARTY_ARCHIVE_DIR "{archives}")\n'
                output = self.run_cmake(body + 'librpa_thirdparty_prepare(Fixture source)\n')
                self.assertNotIn("Downloading", output)
                self.assertEqual(tree_hash(checkout / "thirdparty/src" / self.final.name / "include"), self.patched)
                self.assertFalse((archives / "src").exists())

    def standalone_script(self, name):
        script = self.checkout / "cmake" / name
        shutil.copy2(MODULE.parent / name, script)
        declaration = self.preamble[self.preamble.index("librpa_thirdparty_declare"):]
        (self.storage / "manifest.cmake").write_text(declaration)
        return script

    def test_download_script_with_existing_sources(self):
        self.run_cmake("librpa_thirdparty_prepare(Fixture source)\n")
        header = self.final / "include/vendor/header.hpp"
        header.write_text("// local changes\n")
        script = self.standalone_script("LibRPADownloadThirdparty.cmake")
        downloads = self.root / "Downloads"
        command = ["cmake", f"-DLIBRPA_THIRDPARTY_ARCHIVE_DIR={downloads}", "-P", str(script)]
        result = subprocess.run(command, capture_output=True, text=True, env=fixture_environment())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual((downloads / "fixture.tar.gz").read_bytes(),
                         (self.archives / "fixture.tar.gz").read_bytes())
        self.assertEqual(header.read_text(), "// local changes\n")
        self.assertNotIn("unpinned", result.stdout + result.stderr)
        # Cached archives are still verified even when prepared sources exist.
        (downloads / "fixture.tar.gz").write_bytes(b"corrupt")
        result = subprocess.run(command, capture_output=True, text=True, env=fixture_environment())
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("archive SHA256 mismatch", result.stdout + result.stderr)

    def test_download_script_component_selection(self):
        script = self.standalone_script("LibRPADownloadThirdparty.cmake")
        manifest = self.storage / "manifest.cmake"
        declaration = manifest.read_text().replace("declare(Fixture ", "declare(Second ").replace(
            "ARCHIVE fixture.tar.gz", "ARCHIVE second.tar.gz")
        manifest.write_text(manifest.read_text() + declaration)
        # Downloads do not require patch files or create source directories.
        self.patch.unlink()
        downloads = self.root / "Downloads"
        command = ["cmake", f"-DLIBRPA_THIRDPARTY_ARCHIVE_DIR={downloads}"]
        for components in (["-DCOMPONENTS=Fixture"], []):
            result = subprocess.run(command + components + ["-P", str(script)],
                                    capture_output=True, text=True, env=fixture_environment())
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue((downloads / "fixture.tar.gz").exists())
            self.assertEqual((downloads / "second.tar.gz").exists(), not components)
            self.assertFalse((self.storage / "src").exists())

    def test_preparation_script_reuses_sources_without_archives(self):
        script = self.standalone_script("LibRPAPrepareThirdparty.cmake")
        command = ["cmake", "-P", str(script)]
        result = subprocess.run(command, capture_output=True, text=True, env=fixture_environment())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(tree_hash(self.final / "include"), self.patched)
        header = self.final / "include/vendor/header.hpp"
        header.write_text("// local changes\n")
        downloads = self.root / "empty Downloads"
        result = subprocess.run(["cmake", f"-DLIBRPA_THIRDPARTY_ARCHIVE_DIR={downloads}",
                                 "-DLIBRPA_THIRDPARTY_DOWNLOAD=OFF", "-P", str(script)],
                                capture_output=True, text=True, env=fixture_environment())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(downloads.exists())
        self.assertEqual(header.read_text(), "// local changes\n")
        self.assertIn("unpinned", result.stdout + result.stderr)

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
        archive = self.archives / "fixture.tar.gz"
        archive.write_text("not a tarball")
        body = f'set(LIBRPA_THIRDPARTY_ARCHIVE_DIR "{self.archives}")\nlibrpa_thirdparty_prepare(Fixture source)\n'
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
        self.assertIn("archive SHA256 mismatch", output)
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


class PathResolutionTest(unittest.TestCase):
    def test_installation_alias_precedence_and_persistence(self):
        with tempfile.TemporaryDirectory(prefix="librpa-alias-") as temporary:
            root = Path(temporary).resolve()
            project = root / "CMakeLists.txt"
            project.write_text(f'''cmake_minimum_required(VERSION 3.18)
project(Alias NONE)
include("{MODULE}")
librpa_thirdparty_path(${{NEW_NAME}} resolved ${{OLD_NAME}})
file(WRITE "${{CMAKE_BINARY_DIR}}/resolved.txt" "${{resolved}}")
''')
            for new, old in (("ELPA_PATH", "EXTERNAL_ELPA_DIR"), ("GREENX_PATH", "EXTERNAL_GREENX_DIR")):
                cases = (
                    ({old: "legacy-cache"}, {old: "legacy-env"}, "legacy-cache"),
                    ({}, {old: "legacy-env"}, "legacy-env"),
                    ({old: "legacy-cache"}, {new: "new-env"}, "new-env"),
                    ({new: "new-cache", old: "legacy-cache"}, {new: "new-env", old: "legacy-env"}, "new-cache"),
                    ({old: ""}, {}, ""),
                    ({}, {}, ""),
                )
                for number, (cache, environment_values, selected) in enumerate(cases):
                    with self.subTest(new=new, case=number):
                        build = root / f"{new}-{number}"
                        command = ["cmake", "-S", str(root), "-B", str(build), f"-DNEW_NAME={new}", f"-DOLD_NAME={old}"]
                        definitions = [f"-D{k}={root / v if v else ''}" for k, v in cache.items()]
                        environment = fixture_environment()
                        environment.update({k: str(root / v) for k, v in environment_values.items()})
                        expected = str(root / selected) if selected else ""
                        for arguments, env in ((command + definitions, environment), (command, fixture_environment())):
                            result = subprocess.run(arguments, capture_output=True, text=True, env=env)
                            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                            self.assertEqual((build / "resolved.txt").read_text(), expected)
                            self.assertEqual(f"{old} is deprecated" in result.stderr, old in cache or old in environment_values)
                        if not cache and not environment_values:
                            self.assertNotIn(f"{old}:", (build / "CMakeCache.txt").read_text())

    def test_relative_paths_share_one_base(self):
        with tempfile.TemporaryDirectory(prefix="librpa-paths-") as temporary:
            # CMake resolves the invocation directory through symlinks, such
            # as /var -> /private/var on macOS.
            root = Path(temporary).resolve()
            project = root / "project"
            project.mkdir()
            invocation = root / "invocation"
            invocation.mkdir()
            (project / "CMakeLists.txt").write_text(f'''cmake_minimum_required(VERSION 3.18)
project(Paths NONE)
include("{MODULE}")
librpa_thirdparty_path(${{CHECK_VARIABLE}} resolved)
if(NOT resolved STREQUAL "{invocation}/relative")
  message(FATAL_ERROR "Unexpected resolved path: ${{resolved}}")
endif()
''')
            for variable in PATH_VARIABLES:
                for origin in ("untyped", "typed", "environment"):
                    with self.subTest(variable=variable, origin=origin):
                        command = ["cmake", "-S", str(project), "-B", str(root / f"{variable}-{origin}"),
                                   f"-DCHECK_VARIABLE={variable}"]
                        environment = fixture_environment()
                        if origin == "environment":
                            environment[variable] = "relative"
                        else:
                            command.append(f"-D{variable}{':PATH' if origin == 'typed' else ''}=relative")
                        result = subprocess.run(command, cwd=invocation, capture_output=True,
                                                text=True, env=environment)
                        output = " ".join((result.stdout + result.stderr).split())
                        self.assertEqual(result.returncode == 0, origin == "untyped", output)
                        if origin != "untyped":
                            self.assertIn("Supply an absolute path", output)
                            self.assertIn(variable, output)


    def test_environment_paths_persist_and_cache_takes_precedence(self):
        with tempfile.TemporaryDirectory(prefix="librpa-environment-") as temporary:
            root = Path(temporary)
            build = root / "build"
            project = root / "CMakeLists.txt"
            project.write_text(f'''cmake_minimum_required(VERSION 3.18)
project(Paths NONE)
include("{MODULE}")
file(WRITE "${{CMAKE_BINARY_DIR}}/resolved.txt" "")
foreach(variable {' '.join(PATH_VARIABLES)})
  librpa_thirdparty_path(${{variable}} resolved)
  file(APPEND "${{CMAKE_BINARY_DIR}}/resolved.txt" "${{variable}}=${{resolved}}\n")
endforeach()
add_custom_target(check_paths ALL COMMAND "${{CMAKE_COMMAND}}" -E echo "Paths retained")
''')
            command = ["cmake", "-S", str(root), "-B", str(build)]
            environment = fixture_environment()
            environment.update({v: str(root / "environment" / v) for v in PATH_VARIABLES})
            def run(arguments, env):
                result = subprocess.run(arguments, capture_output=True, text=True, env=env)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                return result.stdout + result.stderr
            output = run(command + [f"-D{v}:PATH=" for v in PATH_VARIABLES], environment)
            for variable in PATH_VARIABLES:
                self.assertIn(f"{variable}: using and caching path from environment:", output)
                self.assertIn(f"{variable}:PATH={environment[variable]}", (build / "CMakeCache.txt").read_text())
            saved = (build / "resolved.txt").read_text()
            project.write_text(project.read_text() + "\n# Trigger automatic reconfiguration.\n")
            output = run(["cmake", "--build", str(build)], fixture_environment())
            self.assertIn("Configuring done", output)
            self.assertEqual((build / "resolved.txt").read_text(), saved)
            changed = dict(environment)
            changed.update({v: str(root / "changed" / v) for v in PATH_VARIABLES})
            output = run(command, changed)
            self.assertNotIn("using and caching path from environment", output)
            self.assertEqual((build / "resolved.txt").read_text(), saved)
            run(command + [f"-D{v}:PATH={root}/override/{v}" for v in PATH_VARIABLES], environment)
            resolved = (build / "resolved.txt").read_text()
            for variable in PATH_VARIABLES:
                self.assertIn(f"{variable}={root}/override/{variable}", resolved)
            run(command + [f"-U{v}" for v in PATH_VARIABLES], fixture_environment())
            self.assertEqual((build / "resolved.txt").read_text(), "".join(f"{v}=\n" for v in PATH_VARIABLES))

    def test_relative_header_path_uses_invocation_directory(self):
        with tempfile.TemporaryDirectory(prefix="librpa-headers-") as temporary:
            root = Path(temporary)
            project = root / "project"
            project.mkdir()
            invocation = root / "invocation"
            header = invocation / "relative/vendor/header.hpp"
            header.parent.mkdir(parents=True)
            header.write_text("// header\n")
            digest = tree_hash(header.parent.parent)
            (project / "CMakeLists.txt").write_text(f'''cmake_minimum_required(VERSION 3.18)
project(Headers NONE)
include("{MODULE}")
librpa_thirdparty_declare(Fixture VERSION 1 URL unused ARCHIVE unused ARCHIVE_SHA256 unused
  HASH_SUBDIR include TREE_SHA256 {digest} LICENSE test)
librpa_thirdparty_path(LIBRI_INCLUDE_DIR checked)
if(NOT EXISTS "${{checked}}/vendor/header.hpp")
  message(FATAL_ERROR "Header pre-check failed")
endif()
librpa_thirdparty_resolve_headers(Fixture Fixture::Fixture LIBRI_INCLUDE_DIR
  vendor/header.hpp resolved status)
if(NOT checked STREQUAL resolved OR NOT status STREQUAL "pinned")
  message(FATAL_ERROR "Header checks and resolution disagree")
endif()
''')
            result = subprocess.run(["cmake", "-S", str(project), "-B", str(root / "build"),
                                     "-DLIBRI_INCLUDE_DIR=relative"], cwd=invocation,
                                    capture_output=True, text=True, env=fixture_environment())
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class IncludePrecedenceTest(unittest.TestCase):
    @unittest.skipIf(os.name == "nt", "CPATH is not used by MSVC")
    def test_selected_header_wins_over_cpath(self):
        with tempfile.TemporaryDirectory(prefix="librpa-include-precedence-") as temporary:
            root = Path(temporary).resolve()
            selected = root / "selected"
            ambient = root / "ambient"
            other = root / "other"
            for directory in (selected, ambient, other):
                (directory / "vendor").mkdir(parents=True)
            (selected / "vendor/header.hpp").write_text("#define SELECTED_VALUE 17\n")
            (ambient / "vendor/header.hpp").write_text("#define SELECTED_VALUE 99\n")
            (other / "vendor/other.hpp").write_text("// another imported dependency\n")
            (root / "probe.cpp").write_text(
                "#include <vendor/header.hpp>\n"
                "#include <vendor/other.hpp>\n"
                'static_assert(SELECTED_VALUE == 17, "ambient dependency header was selected");\n'
                "int main() { return 0; }\n")
            (root / "CMakeLists.txt").write_text(f'''cmake_minimum_required(VERSION 3.18)
project(IncludePrecedence CXX)
include("{MODULE}")
librpa_thirdparty_declare(Fixture VERSION 1 URL unused ARCHIVE unused ARCHIVE_SHA256 unused
  HASH_SUBDIR include TREE_SHA256 {tree_hash(selected)} LICENSE test)
librpa_thirdparty_resolve_headers(Fixture Fixture::Fixture LIBRI_INCLUDE_DIR
  vendor/header.hpp resolved status)
add_library(Other::Other INTERFACE IMPORTED GLOBAL)
set_target_properties(Other::Other PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "{other}")
add_executable(probe probe.cpp)
target_link_libraries(probe PRIVATE Fixture::Fixture Other::Other)
''')
            environment = fixture_environment()
            environment["CPATH"] = str(ambient)
            build = root / "build"
            result = subprocess.run(
                ["cmake", "-S", str(root), "-B", str(build), f"-DLIBRI_INCLUDE_DIR={selected}"],
                capture_output=True, text=True, env=environment)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run(["cmake", "--build", str(build), "--verbose"],
                                    capture_output=True, text=True, env=environment)
            output = result.stdout + result.stderr
            self.assertEqual(result.returncode, 0, output)
            # Only the selected dependency loses SYSTEM. Other imported targets keep it.
            self.assertIn(f"-I{selected}", output)
            self.assertIn(f"-isystem {other}", output)


class ELPABuildOptionsTest(unittest.TestCase):
    def test_legacy_aliases_and_explicit_new_values(self):
        with tempfile.TemporaryDirectory(prefix="librpa-elpa-options-") as temporary:
            root = Path(temporary)
            for suffix, old_value, new_value in (("KERNEL", "AVX2", ""), ("OPENMP", "ON", "OFF"),
                                                  ("CONFIGURE_ARGS", "--enable-store-build-config", ""),
                                                  ("LIBS", "-lold", "-lnew")):
                for override in (False, True):
                    with self.subTest(suffix=suffix, override=override):
                        old = f"LIBRPA_BUNDLED_ELPA_{suffix}"
                        new = f"LIBRPA_ELPA_{suffix}"
                        script = root / "options.cmake"
                        script.write_text(f'''cmake_minimum_required(VERSION 3.18)
include("{MODULE.parent}/ELPABuildOptions.cmake")
message(STATUS "RESOLVED=${{{new}}}")
''')
                        command = ["cmake", f"-D{old}={old_value}"]
                        if override:
                            command.append(f"-D{new}={new_value}")
                        result = subprocess.run(command + ["-P", str(script)], capture_output=True, text=True)
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                        self.assertIn(f"{old} is deprecated", result.stderr)
                        self.assertIn(f"RESOLVED={new_value if override else old_value}\n", result.stdout)

    def test_source_version_and_build_settings(self):
        with tempfile.TemporaryDirectory(prefix="librpa-elpa-source-") as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            # A compiler-free OpenMP package lets us inspect generated build rules.
            (root / "FindOpenMP.cmake").write_text('''foreach(lang C CXX Fortran)
  add_library(OpenMP::OpenMP_${lang} INTERFACE IMPORTED)
  set(OpenMP_${lang}_FLAGS -fopenmp)
endforeach()
''')
            (root / "CMakeLists.txt").write_text(f'''cmake_minimum_required(VERSION 3.18)
project(ELPASource NONE)
list(PREPEND CMAKE_MODULE_PATH "{root}")
include("{MODULE.parent}/ELPABuildOptions.cmake")
set(_elpa_source_dir "{source}")
set(CMAKE_INSTALL_LIBDIR lib)
set(CMAKE_INSTALL_INCLUDEDIR include)
set(CMAKE_STATIC_LIBRARY_PREFIX lib)
set(CMAKE_STATIC_LIBRARY_SUFFIX .a)
add_subdirectory("{MODULE.parent.parent}/thirdparty/ELPA" elpa-build)
get_target_property(includes elpa INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(links elpa INTERFACE_LINK_LIBRARIES)
ExternalProject_Get_Property(librpa_bundled_elpa CONFIGURE_COMMAND)
file(WRITE "${{CMAKE_BINARY_DIR}}/settings.txt" "${{includes}}\n${{links}}\n${{CONFIGURE_COMMAND}}")
''')
            for version in ("2024.03.001", "2026.02.001"):
                for openmp in (False, True):
                    with self.subTest(version=version, openmp=openmp):
                        (source / "configure").write_text(f"PACKAGE_VERSION='{version}'\n")
                        build = root / f"build-{version}-{openmp}"
                        result = subprocess.run(["cmake", "-S", str(root), "-B", str(build),
                            "-DLIBRPA_BUNDLED_ELPA_VERSION=ignored", "-DLIBRPA_ELPA_KERNEL=AVX2",
                            f"-DLIBRPA_ELPA_OPENMP={'ON' if openmp else 'OFF'}",
                            "-DLIBRPA_ELPA_CONFIGURE_ARGS=--enable-store-build-config",
                            "-DLIBRPA_ELPA_LIBS=-L/custom/lib -lmath"], capture_output=True, text=True)
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                        self.assertIn("LIBRPA_BUNDLED_ELPA_VERSION is deprecated and ignored", result.stderr)
                        settings = (build / "settings.txt").read_text()
                        library = "elpa_openmp" if openmp else "elpa"
                        self.assertIn(f"include/{library}-{version}", settings)
                        self.assertIn("--enable-avx2-kernels" if version.startswith("2026") else "--enable-avx2;", settings)
                        self.assertIn("--enable-openmp" if openmp else "--disable-openmp", settings)
                        self.assertEqual("OpenMP::OpenMP_Fortran" in settings, openmp)
                        self.assertIn("--enable-store-build-config", settings)
                        self.assertIn("LIBS=-L/custom/lib -lmath", settings)


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
include("{MODULE.parent.parent}/thirdparty/ELPA/CMakeLists.txt")
''')
                        result = subprocess.run(["cmake", "-S", str(root), "-B", str(build)],
                                                capture_output=True, text=True, env=fixture_environment())
                        self.assertNotEqual(result.returncode, 0)
                        output = " ".join((result.stdout + result.stderr).split())
                        self.assertIn("ELPA source builds do not support whitespace", output)
                        self.assertIn(" ".join(str(source if kind == "source" else build).split()), output)
                        self.assertIn("LibRPA checkout (or ELPA_SOURCE_DIR)", output)


class GreenXResolutionTest(unittest.TestCase):
    """Exercise target and package resolution without a compiler or network."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="librpa greenx ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.checkout = self.root / "LibRPA"
        self.module = copy_helper(self.checkout)
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
include("{self.module}")
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
        managed = self.checkout / f"thirdparty/src/greenX-{digest[:12]}"
        shutil.copytree(source, managed)
        body = f'''librpa_thirdparty_declare(greenX VERSION 1234567 URL unused ARCHIVE unused ARCHIVE_SHA256 unused
  TREE_SHA256 {digest} LICENSE test)
'''
        for value in ("ON", "OFF"):
            output = self.configure(body + f"\nset(LIBRPA_USE_EXTERNAL_GREENX {value})")
            self.assertIn("LIBRPA_USE_EXTERNAL_GREENX is deprecated and ignored", output)
            self.assertIn("greenX=managed", output)
        cache = (self.root / "build/CMakeCache.txt").read_text()
        prefix = next(line.split("=", 1)[1] for line in cache.splitlines()
                      if line.startswith("CMAKE_INSTALL_PREFIX:"))
        self.assertFalse(Path(prefix).is_relative_to(self.checkout / "thirdparty"))
        subprocess.run(["cmake", "--install", str(self.root / "build"),
                        "--prefix", str(self.root / "install")], check=True, capture_output=True)
        self.assertTrue((self.root / "install/share/greenx/CMakeLists.txt").is_file())
        self.configure(body)
        self.assertEqual(tree_hash(managed), digest)
        with (managed / "CMakeLists.txt").open("a") as source_file:
            source_file.write("# local edit\n")
        output = self.configure(body)
        self.assertIn("greenX=unpinned", output)
        self.assertIn(f"(source: {managed})", output)
        self.assertTrue((managed / "CMakeLists.txt").read_text().endswith("# local edit\n"))

    def test_parent_target_with_deprecated_option(self):
        for target in ("LibGXMiniMax", "greenX::LibGXMiniMax"):
            with self.subTest(target=target):
                output = self.configure(f'''set(LIBRPA_USE_EXTERNAL_GREENX OFF)
add_library({target} INTERFACE IMPORTED)
''')
                self.assertIn("greenX=target", output)
                self.assertIn("LIBRPA_USE_EXTERNAL_GREENX is deprecated and ignored", output)
                self.assertNotIn("Downloading", output)
                self.assertNotIn("prebuilt installation", output)

    def test_installed_transitive_modules_and_compiler_warning(self):
        body = f'set(GREENX_PATH "{self.prefix}")'
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

    def test_legacy_installation_alias(self):
        output = self.configure(f'set(EXTERNAL_GREENX_DIR "{self.prefix}")')
        self.assertIn("EXTERNAL_GREENX_DIR is deprecated", output)
        self.assertIn(f"greenX=installed({self.prefix})", output)
        output = self.configure(f'set(GREENX_SOURCE_DIR /source)\nset(EXTERNAL_GREENX_DIR "{self.prefix}")', success=False)
        self.assertIn("mutually exclusive", output)

    def test_ambiguous_or_missing_external_source(self):
        output = self.configure('set(GREENX_SOURCE_DIR /source)\nset(GREENX_PATH /prefix)',
                                success=False)
        self.assertIn("mutually exclusive", output)



if __name__ == "__main__":
    unittest.main()
