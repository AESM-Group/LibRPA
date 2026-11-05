import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.error import URLError
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_ci_changes import build_required, needs_build, scheduled_build_required


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


class EditedPrBuildTest(unittest.TestCase):
    def setUp(self):
        self.event = {"action": "edited", "pull_request": {
            "number": 92, "head": {"sha": "head"}, "body": "Tier: H",
        }}
        self.environ = {
            "GITHUB_REPOSITORY": "owner/repo", "GITHUB_SHA": "merge",
            "GITHUB_RUN_ID": "42",
        }
        self.run = {
            "id": 41, "head_sha": "head", "status": "completed",
            "conclusion": "success", "pull_requests": [],
            "referenced_workflows": [{
                "path": "owner/repo/.github/workflows/ci-oneapi.yml@merge",
                "sha": "merge", "ref": "refs/pull/92/merge",
            }],
        }
        self.job = {
            "name": "oneAPI / ONEAPI", "status": "completed", "conclusion": "success",
        }
        self.api = Mock(side_effect=lambda path, **kwargs: (
            {"workflow_runs": [self.run]} if "scheduler.yml" in path
            else {"jobs": [self.job]}
        ))
        diff = patch("check_ci_changes.build_required", return_value=True)
        self.diff = diff.start()
        self.addCleanup(diff.stop)

    def decision(self):
        return scheduled_build_required(
            self.event, environ=self.environ, get_json=self.api,
        )

    def test_description_edit_reuses_successful_build(self):
        self.assertFalse(self.decision())

    def test_tier_correction_builds_when_no_success_exists(self):
        self.api.return_value = {"workflow_runs": []}
        self.api.side_effect = None
        self.assertTrue(self.decision())

    def test_unsuccessful_or_skipped_job_does_not_skip_build(self):
        for conclusion in ("skipped", "failure", "cancelled", None):
            with self.subTest(conclusion=conclusion):
                self.job["conclusion"] = conclusion
                self.assertTrue(self.decision())

    def test_other_job_success_does_not_skip_build(self):
        self.job["name"] = "Checks / checks"
        self.assertTrue(self.decision())

    def test_other_head_or_current_run_does_not_skip_build(self):
        self.run["head_sha"] = "other-head"
        self.assertTrue(self.decision())
        self.run["head_sha"] = "head"
        self.run["id"] = 42
        self.assertTrue(self.decision())

    def test_changed_base_or_other_pr_does_not_skip_build(self):
        self.environ["GITHUB_SHA"] = "new-merge"
        self.assertTrue(self.decision())
        self.environ["GITHUB_SHA"] = "merge"
        self.event["pull_request"]["number"] = 93
        self.assertTrue(self.decision())

    def test_missing_workflow_revision_does_not_skip_build(self):
        self.run["referenced_workflows"] = []
        self.assertTrue(self.decision())

    def test_api_failure_keeps_build_enabled(self):
        self.api.side_effect = URLError("unavailable")
        self.assertTrue(self.decision())

    def test_non_edit_events_do_not_query_history(self):
        for action in ("opened", "synchronize", "reopened"):
            with self.subTest(action=action):
                self.event["action"] = action
                self.assertTrue(self.decision())
        self.event = {"before": "old", "after": "new"}
        self.assertTrue(self.decision())
        self.api.assert_not_called()

    def test_documentation_only_does_not_query_history(self):
        self.diff.return_value = False
        self.assertFalse(self.decision())
        self.api.assert_not_called()

    def test_run_and_job_pagination_finds_earlier_success(self):
        self.api.side_effect = [
            {"workflow_runs": [dict(self.run, id=42)] * 100},
            {"workflow_runs": [self.run]},
            {"jobs": [dict(self.job, name="other")] * 100},
            {"jobs": [self.job]},
        ]
        self.assertFalse(self.decision())
        self.assertIn("page=2", self.api.call_args_list[1].args[0])
        self.assertIn("page=2", self.api.call_args_list[3].args[0])


if __name__ == "__main__":
    unittest.main()
