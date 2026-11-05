import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_ci_changes import build_required, needs_build


class BuildPathsTest(unittest.TestCase):
    def test_ignored_paths(self):
        for path in (
            "README.md", "sub/guide.md", "docs/image.png",
            "utilities/check_ci_changes.py", "utilities/test_check_risk_tier.py",
            "utilities/convert_fortran_module_to_stub.py",
        ):
            with self.subTest(path=path):
                self.assertFalse(needs_build(path))

    def test_build_paths(self):
        for path in (
            "src/core/rpa.cpp", ".github/workflows/build.yml", "CMakeLists.txt",
            "utilities/convert_data.py", "utilities/sub/check_data.py",
            "regression_tests/backend/test_x.py", "include/a\nb.h",
        ):
            with self.subTest(path=path):
                self.assertTrue(needs_build(path))


class BuildDiffTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name)
        self.git("init", "-q")
        self.write("README.md", "initial\n")
        self.write("source.cpp", "initial\n")
        self.base = self.commit()

    def git(self, *args):
        return subprocess.check_output(
            ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
            cwd=self.repo, stderr=subprocess.PIPE,
        ).decode().strip()

    def write(self, path, content):
        (self.repo / path).write_text(content)

    def commit(self):
        self.git("add", "-A")
        self.git("commit", "-qm", "test")
        return self.git("rev-parse", "HEAD")

    def push_build(self, head):
        return build_required({"before": self.base, "after": head}, cwd=self.repo)

    def test_documentation_then_code_in_push(self):
        self.write("README.md", "updated\n")
        self.assertFalse(self.push_build(self.commit()))
        self.write("source.cpp", "updated\n")
        self.assertTrue(self.push_build(self.commit()))

    def test_source_deletion(self):
        (self.repo / "source.cpp").unlink()
        self.assertTrue(self.push_build(self.commit()))

    def test_source_renamed_to_ignored_path(self):
        self.git("mv", "source.cpp", "source.md")
        self.assertTrue(self.push_build(self.commit()))

    def test_pr_ignores_changes_only_on_base(self):
        self.write("README.md", "PR docs\n")
        head = self.commit()
        self.git("checkout", "-q", "--detach", self.base)
        self.write("source.cpp", "base-only change\n")
        base = self.commit()
        event = {"pull_request": {"base": {"sha": base}, "head": {"sha": head}}}
        self.assertFalse(build_required(event, cwd=self.repo))
        # A description edit preserves the same diff and build decision.
        event["pull_request"]["body"] = "Tier: H"
        self.assertFalse(build_required(event, cwd=self.repo))

    def test_new_branch(self):
        self.assertTrue(build_required({"before": "0" * 40, "after": self.base}))

    def test_empty_diff(self):
        self.assertFalse(self.push_build(self.base))


if __name__ == "__main__":
    unittest.main()
