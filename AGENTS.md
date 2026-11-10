# LibRPA

LibRPA computes many-body perturbation theory results with numerical atomic
orbitals. It is a C++ library with MPI and OpenMP parallelism, C and Fortran
interfaces, and a standalone driver.

The worst failure is a plausible number that is wrong. Compilation does not
show that a result is correct.

## Build

Initialize the submodules before the first build:

```bash
git submodule update --init --recursive
```

Configure and build with tests:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DLIBRPA_USE_LIBRI=ON -DLIBRPA_ENABLE_TEST=ON
cmake --build build -j 8
```

## Test

Run these checks in the repository root. They need no build:

```bash
python utilities/check_librpa_options.py
python -m unittest discover -s utilities -p 'test_*.py'
python -m unittest discover -s regression_tests/backend/comparisons -p 'test_cmp_*.py'
PYTHONPATH="$PWD/regression_tests:$PWD" python -m unittest discover -s regression_tests/backend -p 'test_*.py'
```

Run the unit tests after a build:

```bash
cmake --build build --target test
ctest --test-dir build -R <test name>
```

After a change to `binding/fortran/librpa_f03.f90`, generate the stub module
again:

```bash
(cd binding/fortran && ../../utilities/convert_fortran_module_to_stub.py librpa_f03.f90 librpa_f03_stubs.f90)
```

Run the regression tests in `regression_tests/`. Use a new workspace for each
run, or add `-f`.

```bash
./run_regression.py full ../build/chi0_main.exe -d workspace_np4 --use-libri -n 4
./run_regression.py full ../build/chi0_main.exe -d workspace_one --use-libri -n 4 --only <test directory>
```

## Rules

- Read `docs/develop/develop_tips.md` for the architecture rules, the runtime
  option rules, CI, the risk tiers, and the evidence standards.
- Use `REVIEW.md` for each code review, which includes local agent reviews such
  as `/code-review` in Claude Code.
- Do not change references, tolerances, or comparators to make a test pass. Give
  the physical reason for each change to an existing reference.
- Give a bug fix a test that fails without the fix.
- Do not write to stdout from several ranks without coordination, because this
  can break the tables that the regression tests parse.
  - Use `lib_printf_root` for text that is the same on all ranks.
  - Use `lib_printf_coll` for text that differs by rank. All ranks in the
    communicator must reach the call, because it gathers the text and then calls
    a barrier.
  - Use `printf_comm_root` or `printf_comm_coll` for another communicator.
- Give the command and its output when you say that a check passes.

## Commits and pull requests

- Use a Conventional Commits title, such as `fix(gw): restore occupations`.
- Keep one topic in each pull request.
- Complete `.github/pull_request_template.md`. Give at least the tier from
  `python utilities/check_risk_tier.py --base <base branch>`.
