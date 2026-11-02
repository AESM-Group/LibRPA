#!/usr/bin/env python3
"""Offline integration checks for managed dependency preparation (no compilers needed)."""
import hashlib
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import unittest


MODULE = Path(__file__).resolve().parents[1] / "cmake/LibRPAThirdparty.cmake"


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

    def run_cmake(self, body, success=True):
        result = subprocess.run(self.script(body), capture_output=True, text=True)
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode == 0, success, output)
        return output

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

    def test_offline_archive_and_failed_patch(self):
        body = "set(LIBRPA_THIRDPARTY_DOWNLOAD OFF)\nlibrpa_thirdparty_prepare(Fixture source)\n"
        output = self.run_cmake(body, success=False)
        self.assertIn("Missing third-party component Fixture", output)
        self.assertIn("LibRPAPrepareThirdparty.cmake", output)
        self.assertIn("Alternatively", output)
        shutil.copy2(self.mirror / "fixture.tar.gz", self.storage / "downloads")
        self.patch.write_text(self.patch.read_text().replace("-// upstream", "-// bad context"))
        output = self.run_cmake(body, success=False)
        self.assertIn("patch failed", output)
        self.assertFalse(list((self.storage / "src").glob(".tmp-*")))
        self.assertFalse(self.final.exists())

    def test_concurrent_preparation(self):
        command = self.script("librpa_thirdparty_prepare(Fixture source)\n")
        children = [subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     text=True) for _ in range(2)]
        for child in children:
            output, error = child.communicate(timeout=30)
            self.assertEqual(child.returncode, 0, output + error)
        self.assertEqual(tree_hash(self.final / "include"), self.patched)
        self.assertFalse(list((self.storage / "src").glob(".tmp-*")))


if __name__ == "__main__":
    unittest.main()
