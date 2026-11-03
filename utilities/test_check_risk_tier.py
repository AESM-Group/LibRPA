import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_risk_tier import declared_tier, install_lines  # noqa: E402

SCRIPT = Path(__file__).resolve().parent / "check_risk_tier.py"

CMAKE = """\
add_library(rpa_lib a.cpp)
# install(TARGETS commented)
install(FILES
    a.h
    b.h
    DESTINATION include)
set(X "install(not a command)")
"""

BASE_FILES = {
    "docs/guide.md": "guide\n",
    "src/core/chi0.cpp": "int a;\nint b;\n",
    "src/test/test_chi0.cpp": "check 1\ncheck 2\n",
    "include/librpa.h": "void f();\n",
    "regression_tests/refs/case/out.txt": "1.0\n",
    "regression_tests/testsuite.xml": "<test a/>\n",
    "regression_tests/backend/comparisons/cmp_float.py": "tol = 1e-8\n",
    "regression_tests/verification/verify_restart.py": "atol = 1e-9\n",
    "src/CMakeLists.txt": CMAKE,
    ".github/workflows/ci.yml": "on: push\n",
}


class TierTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        self.git("init", "-q")
        for path, text in BASE_FILES.items():
            self.write(path, text)
        self.write_bytes("include/logo.bin", b"\0\1\2")
        self.commit()

    def tearDown(self):
        self.tmp.cleanup()

    def git(self, *args):
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
                       cwd=self.repo, check=True, capture_output=True)

    def write(self, path, text):
        p = self.repo / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)

    def write_bytes(self, path, data):
        p = self.repo / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def commit(self):
        self.git("add", "-A")
        self.git("commit", "-q", "-m", "change")

    def run_script(self, *extra, env=None):
        return subprocess.run([sys.executable, str(SCRIPT), "--base", "HEAD^1", *extra],
                              cwd=self.repo, capture_output=True, text=True,
                              env={**os.environ, **(env or {})})

    def tier(self):
        out = self.run_script()
        self.assertEqual(out.returncode, 0, out.stderr)
        return re.search(r"Minimum tier from the diff: \*\*([LMH])\*\*", out.stdout).group(1), out.stdout

    def test_documentation_is_l(self):
        self.write("docs/guide.md", "new guide\n")
        self.write("README.md", "readme\n")
        self.commit()
        self.assertEqual(self.tier()[0], "L")

    def test_new_test_is_l(self):
        self.write("src/test/test_new.cpp", "check\n")
        self.write("src/test/test_chi0.cpp", "check 1\ncheck 2\ncheck 3\n")
        self.write("regression_tests/testsuite.xml", "<test a/>\n<test b/>\n")
        self.write("regression_tests/backend/comparisons/test_cmp_float.py", "check\n")
        self.commit()
        self.assertEqual(self.tier()[0], "L")

    def test_new_reference_is_m(self):
        self.write("regression_tests/refs/new_case/out.txt", "2.0\n")
        self.commit()
        self.assertEqual(self.tier()[0], "M")

    def test_script_in_docs_is_m(self):
        self.write("docs/conf.py", "project = 'LibRPA'\n")
        self.commit()
        self.assertEqual(self.tier()[0], "M")

    def test_changed_test_is_m(self):
        self.write("src/test/test_chi0.cpp", "check 1\n")
        self.commit()
        self.assertEqual(self.tier()[0], "M")

    def test_code_is_m(self):
        self.write("src/core/chi0.cpp", "int a;\nint b;\nint c;\n")
        self.write("include/librpa.h", "void f();\nvoid g();\n")
        self.write("include/new header.h", "void h();\n")
        self.commit()
        tier, out = self.tier()
        self.assertEqual(tier, "M")
        self.assertIn("`src/core/chi0.cpp`", out)

    def check_h(self, path, text=None, delete=False):
        if delete:
            (self.repo / path).unlink()
        else:
            self.write(path, text)
        self.commit()
        tier, out = self.tier()
        self.assertEqual(tier, "H")
        self.assertIn(f"| `{path}` |", out)

    def test_existing_reference_is_h(self):
        self.check_h("regression_tests/refs/case/out.txt", "1.1\n")

    def test_changed_testsuite_line_is_h(self):
        self.check_h("regression_tests/testsuite.xml", "<test a2/>\n")

    def test_comparator_is_h(self):
        self.check_h("regression_tests/backend/comparisons/cmp_float.py", "tol = 1e-8\n# note\n")

    def test_verification_script_is_h(self):
        self.check_h("regression_tests/verification/verify_restart.py", "atol = 1e-9\natol = 1e-6\n")

    def test_process_files_are_h(self):
        for path in ("AGENTS.md", "CONTRIBUTING.md"):
            self.check_h(path, "rules\n")

    def test_changed_public_header_is_h(self):
        self.check_h("include/librpa.h", "void f(int);\n")

    def test_deleted_public_header_is_h(self):
        self.check_h("include/librpa.h", delete=True)

    def test_changed_binary_in_include_is_h(self):
        self.write_bytes("include/logo.bin", b"\0\1\3")
        self.commit()
        self.assertIn("| `include/logo.bin` |", self.tier()[1])

    def test_github_is_h(self):
        self.check_h(".github/workflows/ci.yml", "on: pull_request\n")

    def test_line_inside_install_is_h(self):
        self.check_h("src/CMakeLists.txt", CMAKE.replace("    b.h\n", ""))

    def test_line_outside_install_is_m(self):
        # A changed target, a commented install(), and install( in a string
        text = CMAKE.replace("a.cpp)", "a.cpp b.cpp)").replace("commented", "old") \
            .replace("not a command", "still not")
        self.write("src/CMakeLists.txt", text)
        self.commit()
        self.assertEqual(self.tier()[0], "M")

    def test_path_with_space(self):
        self.write("include/my header.h", "void h();\n")
        self.commit()
        self.check_h("include/my header.h", "void h(int);\n")

    def test_declared_tier(self):
        self.write("include/librpa.h", "void f(int);\n")
        self.commit()
        for body, returncode in (("Tier: H\n", 0), ("Tier: M\n", 1), ("Tier:\nReason:\n", 1)):
            out = self.run_script("--body-env", "BODY", "--github",
                                  env={"BODY": body, "GITHUB_STEP_SUMMARY": ""})
            self.assertEqual(out.returncode, returncode, body)
            self.assertEqual("::error title=Risk tier::" in out.stdout, returncode == 1, body)

    def test_size_warning(self):
        self.write("src/core/big.cpp", "int x;\n" * 1001)
        self.write("regression_tests/testcases/big/input.txt", "1\n" * 5000)
        self.commit()
        out = self.run_script("--github", env={"GITHUB_STEP_SUMMARY": ""})
        self.assertIn("Added lines of production code: 1001", out.stdout)
        self.assertIn("::warning title=Pull request size::", out.stdout)

    def test_later_base_commit_is_not_in_diff(self):
        self.git("branch", "-M", "main")
        self.git("checkout", "-q", "-b", "topic")
        self.write("docs/guide.md", "topic guide\n")
        self.commit()
        self.git("checkout", "-q", "main")
        self.write(".github/workflows/ci.yml", "on: pull_request\n")
        self.commit()
        self.git("checkout", "-q", "topic")
        out = subprocess.run([sys.executable, str(SCRIPT), "--base", "main"],
                             cwd=self.repo, capture_output=True, text=True)
        self.assertIn("**L**", out.stdout)

    def test_bad_base_fails(self):
        out = subprocess.run([sys.executable, str(SCRIPT), "--base", "no-such-branch"],
                             cwd=self.repo, capture_output=True, text=True)
        self.assertEqual(out.returncode, 2)
        self.assertIn("no-such-branch", out.stderr)


class ParserTest(unittest.TestCase):
    def test_install_lines(self):
        self.assertEqual(install_lines(CMAKE), {3, 4, 5, 6})
        # install and ( on two lines, and an install() in a bracket comment
        self.assertEqual(install_lines("a\ninstall\n(FILES\n x\n y\n)\n"), {2, 3, 4, 5, 6})
        self.assertEqual(install_lines("a\n#[[\ninstall(FILES bad.h\n]]\ninstall(FILES real.h)\n"), {5})

    def test_declared_tier(self):
        template = "<!-- L, M, or H. Tier: H -->\n\nTier:\nReason:\n"
        self.assertIsNone(declared_tier(template))
        self.assertEqual(declared_tier("Tier: m\nReason: x\n"), "M")
        self.assertEqual(declared_tier("**Risk tier:** H. This changes `.github/`.\n"), "H")
        # The highest tier, other marks around the tier, and no tier from a code block
        self.assertEqual(declared_tier("Tier: M\nTier: H\n"), "H")
        for body in ("Tier:\u00a0H\n", "Tier: `H`\n", "Tier: (H)\n"):
            self.assertEqual(declared_tier(body), "H", body)
        self.assertEqual(declared_tier("Tier: M\n```\nTier: H\n```\n"), "M")


if __name__ == "__main__":
    unittest.main()
