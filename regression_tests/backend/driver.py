import os
import pathlib
import shlex
import shutil
import tarfile
from collections import OrderedDict
from typing import Tuple

from .utils import run_librpa
from .validate import Validate


__all__ = ["TestDriver"]

PASS_FAIL = {True: "PASS", False: "FAIL"}
EXITCODE_FILE = "librpa.exitcode"


def _has_mpi_task_option(args):
    task_options = {"-n", "-np", "--np", "--ntasks"}
    task_option_prefixes = ("-np=", "--np=", "--ntasks=")
    return any(arg in task_options or arg.startswith(task_option_prefixes)
               for arg in args)


def _build_run_command(exec: pathlib.Path, mpiexec: str, ntasks: int):
    args = shlex.split(mpiexec)
    if not args:
        raise ValueError("--mpiexec cannot be empty")

    launcher = pathlib.Path(args[0]).name
    if launcher in ["mpirun", "mpiexec"] and not _has_mpi_task_option(args[1:]):
        args.extend(["-np", str(ntasks)])

    args.append(str(exec))
    return args


def _format_command(args):
    return " ".join(shlex.quote(str(arg)) for arg in args)


def _prepare_librpa_workspace(src: pathlib.Path, dst: pathlib.Path, copy_files=()):
    """Prepare one test case and return the directory where LibRPA should run."""
    src_librpa = src / "librpa"
    dst_librpa = dst / "librpa"
    dst_librpa.mkdir(parents=True, exist_ok=True)

    if (src_librpa / "librpa.in").is_file() and (src / "dataset.tar.gz").is_file():
        shutil.copy2(src_librpa / "librpa.in", dst_librpa)
        for file in copy_files:
            path = pathlib.PurePosixPath(file)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("copy_files entries must stay inside librpa/: {}".format(file))
            source = src_librpa / pathlib.Path(file)
            target = dst_librpa / pathlib.Path(file)
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                if target.exists():
                    if target.is_dir():
                        shutil.rmtree(target)
                    else:
                        target.unlink()
                shutil.copytree(source, target)
            else:
                shutil.copy2(source, target)
        with tarfile.open(src / "dataset.tar.gz", "r:gz") as tar:
            tar.extractall(path=dst)
        return dst_librpa

    raise FileNotFoundError(
        "unsupported test input layout in {}; expected librpa/librpa.in + dataset.tar.gz"
        .format(src)
    )


def _disable_message(tc: dict):
    labels = tc.get("labels", {})
    disable = labels.get("disable", False)
    if disable is True:
        return ""
    if isinstance(disable, str):
        msg = disable.strip()
        if msg:
            return msg
    return None


def _disable_testcase(tc: dict, message):
    if _disable_message(tc) is not None:
        return
    tc.setdefault("labels", {})["disable"] = message


def _validate_scope_filter(testcases, option: str, values):
    selected = {pathlib.Path(value).name for value in values or []}
    known = {tc["directory"] for tc in testcases}
    unknown = sorted(selected - known)
    if unknown:
        raise ValueError("Unknown test case directory in {:s}: {:s}".format(
            option, ", ".join(unknown)))
    return selected


def _check_build_run_filter(tc: dict,
                            ntasks: int, nthreads: int,
                            use_libri: bool) -> Tuple[bool, bool]:
    # Filter build options
    build = tc["build"]
    if build["require_libri"] and not use_libri:
        return True, False

    # Filter runtime options
    run = tc["run"]
    if ntasks in run["ntasks_disable"]:
        return False, True
    if nthreads in run["nthreads_disable"]:
        return False, True
    nt = run["ntasks_enable"]
    if len(nt) > 0 and ntasks not in nt:
        return False, True
    nt = run["nthreads_enable"]
    if len(nt) > 0 and nthreads not in nt:
        return False, True

    return False, False


class TestDriver:

    def __init__(self, dir_input: str, dir_ref: str, workspace: str, groups: dict):
        self._dir_input = pathlib.Path(dir_input)
        self._dir_ref = pathlib.Path(dir_ref)
        self._workspace = pathlib.Path(workspace)
        self._dir_testcase = self._workspace / "testcases"

        # Check
        if not self._dir_input.exists():
            raise FileNotFoundError("Input directory does not exist")
        if not self._dir_ref.exists():
            raise FileNotFoundError("Reference directory does not exist")

        self._groups = groups
        self._ntasks = None
        self._nthreads = None
        # Testcases that are qualified after initialized with build and runtime conditions
        self._testcases_filtered = None

    def is_initialized(self):
        return self._testcases_filtered is not None

    def reset(self):
        self._ntasks = None
        self._nthreads = None
        self._testcases_filtered = None

    def list(self):
        if self.is_initialized():
            selected = {id(tc) for tc in self._testcases_filtered}
            groups = OrderedDict(
                (g, [tc for tc in gtcs if id(tc) in selected])
                for g, gtcs in self._groups.items()
            )
            title = "Selected test cases"
        else:
            groups = self._groups
            title = "Configured test cases"

        print()
        print(title)
        count = 0
        for group, testcases in groups.items():
            if not testcases:
                continue
            print()
            print("Test group: {}".format(group))
            for tc in testcases:
                count += 1
                note = ""
                if not self.is_initialized():
                    msg = _disable_message(tc)
                    if msg is not None:
                        note = " [disabled"
                        if msg:
                            note += ": {}".format(msg)
                        note += "]"
                print("- {:s}: {:s}{:s}".format(
                    tc["directory"], tc["name"], note))

        print()
        if count == 0:
            print("No test cases selected")
        else:
            print("Total test cases: {:d}".format(count))

    def initialize(self, ntasks: int, nthreads: int, use_libri: bool,
                   only=None, exclude=None):
        self._testcases_filtered = []
        self._ntasks = ntasks
        self._nthreads = nthreads

        if only and exclude:
            raise ValueError("--only and --exclude cannot be used together")

        all_testcases = [tc for gtcs in self._groups.values() for tc in gtcs]
        only = _validate_scope_filter(all_testcases, "--only", only)
        exclude = _validate_scope_filter(all_testcases, "--exclude", exclude)
        if only:
            for tc in all_testcases:
                if tc["directory"] in only:
                    tc.setdefault("labels", {})["disable"] = False
                else:
                    _disable_testcase(tc, "not selected by --only")
        elif exclude:
            for tc in all_testcases:
                if tc["directory"] in exclude:
                    _disable_testcase(tc, "excluded by --exclude")

        print("Initializing workspace targeting:")
        print("- MPI tasks :", ntasks)
        print("- threads   :", nthreads)
        print("- LibRI     ?", use_libri)

        # Filter test cases
        skip_due_to_disable = []
        skip_due_to_build = []
        skip_due_to_run = []

        for g, gtcs in self._groups.items():
            for tc in gtcs:
                if _disable_message(tc) is not None:
                    skip_due_to_disable.append(tc)
                    continue
                filter_build, filter_run = _check_build_run_filter(
                    tc, ntasks, nthreads, use_libri)
                if filter_build:
                    skip_due_to_build.append(tc)
                    continue
                if filter_run:
                    skip_due_to_run.append(tc)
                    continue
                self._testcases_filtered.append(tc)

        if skip_due_to_disable:
            print()
            print("Disabled following test cases")
            for tc in skip_due_to_disable:
                msg = _disable_message(tc)
                reminder = " [{:s}]".format(msg) if msg else ""
                print("- {:s}: {:s}{:s}".format(tc["directory"], tc["name"], reminder))

        if skip_due_to_build:
            print()
            print("Skipped following test cases due to executable build condition")
            for tc in skip_due_to_build:
                print("- {:s}: {:s}".format(tc["directory"], tc["name"]))

        if skip_due_to_run:
            print()
            print("Skipped following test cases due to runtime condition")
            for tc in skip_due_to_run:
                print("- {:s}: {:s}".format(tc["directory"], tc["name"]))

    def run(self, exec: str, mpiexec: str, force, verbose=False):
        if self._workspace.exists() and not force:
            raise FileExistsError("Workspace directory exists, please remove")
        self._workspace.mkdir(parents=True, exist_ok=True)

        if self._testcases_filtered is None:
            raise ValueError("initialize() needs to be called before running")

        # Resolve the absolute path of executable to test
        exec = pathlib.Path(exec).resolve()

        os.environ["OMP_NUM_THREADS"] = str(self._nthreads)
        args = _build_run_command(exec, mpiexec, self._ntasks)

        print()
        if not self._testcases_filtered:
            print("No test case to run")
            return
        for tc in self._testcases_filtered:
            dname = tc["directory"]
            src = self._dir_input / dname
            dst = self._dir_testcase / dname
            run_dir = _prepare_librpa_workspace(src, dst, tc["run"].get("copy_files", ()))
            out = pathlib.Path("librpa.out")
            err = pathlib.Path("librpa.err")
            print("Running {} [{}]".format(tc["name"], dname))
            if verbose:
                print("Command: {}".format(_format_command(args)))
            return_code = run_librpa(args, run_dir, out, err)
            (run_dir / EXITCODE_FILE).write_text("{:d}\n".format(return_code))
        print("Finished test calculations")
        print()

    def _run_failure(self, dname):
        run_dir = self._dir_testcase / dname / "librpa"
        exitcode = run_dir / EXITCODE_FILE
        if not exitcode.is_file():
            return None
        return_code = int(exitcode.read_text().strip())
        if return_code == 0:
            return None
        return "calculation failed with exit code {:d}; see {}".format(
            return_code, run_dir / "librpa.err")

    def analyze(self):
        status = 0
        good_all = []
        for tc in self._testcases_filtered:
            # name = tc["name"]
            dname = tc["directory"]
            run_failure = self._run_failure(dname)
            tc["run_failure"] = run_failure
            if run_failure is not None:
                good_all.append(False)
                tc["results"] = []
                continue
            test = self._dir_testcase / dname
            refr = self._dir_ref / dname
            results = []
            for v in tc["validates"]:
                entry = Validate(v["name"],
                                 v["file"],
                                 v["comparison"],
                                 v["headers"],
                                 v["rows"],
                                 v["regex"],
                                 v["occurences"],
                                 v["binary_extract"],
                                 file_test=v.get("file_test"),
                                 file_refr=v.get("file_refr"),
                                 collective=v.get("collective", False),
                                 binary_extract_preset=v.get("binary_extract_preset"),
                                 )
                good, msg = entry.evaluate(test, refr)
                good_all.append(good)
                results.append([good, msg])
            tc["results"] = results
        if not all(good_all):
            return 1
        return status

    def result_counts(self):
        tests_run = len(self._testcases_filtered)
        failed = sum(
            bool(tc.get("run_failure")) or
            not all(result[0] for result in tc.get("results", []))
            for tc in self._testcases_filtered
        )
        return tests_run, tests_run - failed, failed

    def print(self):
        for g, gtcs in self._groups.items():
            gtcs_active = [tc for tc in gtcs if _disable_message(tc) is None]
            if not gtcs_active:
                continue
            print()
            print("Test group: {}".format(g))
            for tc in gtcs_active:
                results = tc.get("results", None)
                print()
                if results:
                    s = "Validate results for {} [directory: {}]: {}"
                    good_all = PASS_FAIL.get(all(x[0] for x in results))
                    print(s.format(tc["name"], tc["directory"], good_all))
                    for v, e in zip(tc["validates"], results):
                        print("- {:4s}: {:s}, {:s}".format(PASS_FAIL[e[0]], v["name"].strip(), e[1]))
                elif tc.get("run_failure"):
                    s = "Validate results for {} [directory: {}]: FAIL"
                    print(s.format(tc["name"], tc["directory"]))
                    print("- FAIL: calculation, {:s}".format(tc["run_failure"]))
                else:
                    s = "No results to validate for {} [directory: {}]"
                    print(s.format(tc["name"], tc["directory"]))
