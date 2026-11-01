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

`.github/workflows/ci.yml` has the compiler settings and the other build
configurations.

## Test

Run these checks in the repository root:

```bash
cmake --build build --target test
ctest --test-dir build -R <test name>
python utilities/check_librpa_options.py
python -m unittest discover -s regression_tests/backend/comparisons -p 'test_cmp_*.py'
PYTHONPATH="$PWD/regression_tests:$PWD" python -m unittest discover -s regression_tests/backend -p 'test_*.py'
```

Run the regression tests in `regression_tests/`. Use a new workspace for each
run, or add `-f`.

```bash
./run_regression.py full ../build/chi0_main.exe -d workspace_np4 --use-libri -n 4
./run_regression.py full ../build/chi0_main.exe -d workspace_one --use-libri -n 4 --only <test directory>
```

## Rules

- Read `docs/develop/develop_tips.md` for the architecture rules, the runtime
  option rules, and the risk tiers.
- Use `REVIEW.md` for each code review, which includes local `/code-review` runs.
- Do not change references, tolerances, or comparators to make a test pass. Give
  the physical reason for each change to an existing reference.
- Give a bug fix a test that fails without the fix.
- Write to stdout from rank 0 only, with `lib_printf_root` or `lib_printf_coll`.
  Put per-rank output in `global::ofs_myid`. Output from other ranks can break
  the tables that the regression tests parse.
- Give the command and its output when you say that a check passes.

## Commits and pull requests

- Use a Conventional Commits title, such as `fix(gw): restore occupations`.
- Keep one topic in each pull request.
- Complete `.github/pull_request_template.md`.
