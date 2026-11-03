# Development Tips

## Git collaborative workflow

The recommended workflow is:

1. Fork the repository.
2. Create a topic branch for your changes.
3. Rebase your branch on `master` when it is ready for review.
4. Resolve any conflicts locally.
5. Open a pull request and complete the template.
6. Address review comments. Resolve each review thread with a fix or a reason.
7. Merge with a squash merge.

A squash merge keeps each pull request as one commit on `master`, so that one
`git revert` undoes it. Use a Conventional Commits title for the pull request,
because this title becomes the commit title.

Keep a pull request small enough to read. If it adds more than 1000 lines of
production code, split it or give the reason in the description. Do not count
dependency snapshots, data, or generated files. Keep each test in the pull
request whose behavior it protects.

If the complete work adds more than 3000 lines, open a design issue first with
the Design issue template. The issue gives the physical definition, the inputs,
the file formats, the validation plan, and the planned pull requests.

The `Risk tier` workflow counts the added lines of production code and gives a
warning above these limits. It does not count `thirdparty/`, `regression_tests/`,
documentation, tests, and the generated Fortran stub module.

### Continuous integration

These workflows in `.github/workflows/` run on each pull request:

| Workflow | What it does |
| :------- | :----------- |
| `checks.yml` | Checks that need no build: the runtime option definitions, the Fortran stub module, and the tests of the Python scripts |
| `ci.yml` | Builds LibRPA and runs the unit tests and the regression tests |
| `deploy-ghpages-doc.yml` | Builds the documentation |
| `risk-tier.yml` | Gives the minimum risk tier and the size of the change. See "Risk tiers" |

`ci.yml` does not run when a change has only these files:

- Files in `docs/` and `.md` files.
- The scripts in `utilities/` that only `checks.yml` uses:
  `check_*.py`, `test_*.py`, and `convert_fortran_module_to_stub.py`.

The other workflows run for each change. If a script in `utilities/` becomes
part of a `ci.yml` test, remove it from the `paths-ignore` lists in `ci.yml`.

When you push a new commit to a pull request, the runs for the earlier commit
stop. On `master`, a run that started does not stop, and a new run waits in a
queue until it ends. If more commits arrive during this time, only the latest
one stays in the queue.

### Risk tiers

The change sets the tier, not the author. A reviewer or an agent can raise the
tier, but cannot lower it.

| Tier | Change | Review |
| :--- | :----- | :----- |
| L | Documentation, or new tests that add no files to `regression_tests/refs/`, with no H condition | The author can merge after CI passes |
| M | Other changes with no H condition | One reviewer other than the author examines the intent, the evidence, and the agent findings |
| H | At least one H condition | One area owner other than the author examines the change and its numerical evidence |

`.github/CODEOWNERS` gives the area owners. GitHub requests a review from the
owners of the changed files.

The H conditions are in two groups. The paths and the lines of the diff show
the conditions in the first group:

- A change to an existing file in `regression_tests/refs/`.
- A removed or changed line in `regression_tests/testsuite.xml`.
- A change to a comparator in `regression_tests/backend/comparisons/`.
- A change to a script in `regression_tests/verification/`.
- A removed or changed line in `include/` or `binding/fortran/`.
- A change to `.github/`, `thirdparty/`, or `src/mpi/`.
- A change to `REVIEW.md`, `CLAUDE.md`, `AGENTS.md`, or `CONTRIBUTING.md`.
- An added, removed, or changed `install(` command in a CMake file.

A person or an agent must examine the change for the conditions in the second
group:

- A change to an installed file outside `include/`, such as a CMake package
  file.
- A change to an input, output, or restart file format.
- A change to a default value.
- A change to a convention: basis, matrix order, transpose or conjugate,
  normalization, units, k or q mapping, or parallel data ownership.
- A change to the value of an existing computed result, such as an energy, a
  self-energy, a screened interaction, or a density. A new quantity that does
  not change the existing results does not meet this condition.

If a change alters an existing computed result, the description gives the
equation that the code implements: an equation in a paper, or a derivation in
`docs/theory/`. If no paper gives the derivation, write it in `docs/theory/`.
The reviewer compares the code with this equation, not with the old code.

`utilities/check_risk_tier.py` applies the first group and gives the minimum
tier. If you change the first group, change the script and this list together.
Run the script before you open a pull request, with the base branch as `--base`:

```bash
python utilities/check_risk_tier.py --base upstream/master
```

The `Risk tier` workflow runs the script on each pull request and writes the
result to the job summary. The workflow fails if the description has no tier, or
gives a tier lower than the minimum tier. Correct the `Tier:` line, and the
workflow runs again. Do not merge a pull request while this check fails.

The reviewer examines the change for the second group. If the change meets a
condition in this group, the author changes the tier to H.

A maintainer can merge an M pull request without a reviewer if all these
conditions are true:

- No reviewer responded in 24 hours.
- The pull request has no more than 300 changed lines of production code.
- The pull request has no open Important finding.

A reviewer then examines the merged change in the next 7 days.

### Numerical evidence

Give evidence that supports the claim. Each type of evidence has limits.

| Evidence | It supports | It does not support |
| :------- | :---------- | :------------------ |
| Build and link | The interfaces build in the tested configuration | Correct results |
| Small model or analytic comparison | A specific formula, convention, limit, or branch | All production paths |
| Driver or API regression | Behavior on the tested path and inputs | Material convergence or general accuracy |
| Comparison of MPI layouts, symmetry paths, or restarts | Agreement between the compared paths | The absence of an error that the paths share |
| Material benchmark or convergence study | A result for the tested systems and accuracy | Systems that were not tested |

The standard depends on the type of change:

- A refactor keeps the behavior, so existing references do not change. If it
  changes the order of operations, give the floating-point difference that it
  allows.
- A method change needs independent evidence. Agreement with the old results
  does not show that the new method is correct.

### Regression tests and verification scripts

A regression test compares the output of a case with stored reference values.
Put the input in `regression_tests/testcases/<case>`, the reference values in
`regression_tests/refs/<case>`, and the test in `regression_tests/testsuite.xml`.
Use a comparator in `regression_tests/backend/comparisons/`, or add one.

A script in `regression_tests/verification/` runs several calculations in one
job and compares the results with each other, for example a continuous run with
a restarted run, two MPI layouts, or a value with the same value calculated
again from other output. The script uses the inputs in
`regression_tests/testcases/`, which the regression tests also use, so a change
to these inputs affects both. The script does not keep reference values. If a
check needs reference values, make it a regression test. The tolerances in these scripts are numerical gates, so a
change to a script is an H condition.

### Agent review

The author runs the agent review before asking for a review, for example with
`/code-review high <pull request number>` in Claude Code. `AGENTS.md` tells the
agent to use `REVIEW.md`, and `CLAUDE.md` imports `AGENTS.md`. The agent review is advisory and is not a CI check.

Record the status of each agent review in the template: completed, failed, or
not run. A failed
review with no findings does not show that the change is correct. If the review
did not complete, the reviewer examines the full change.

Use a maximum of three rounds of agent review and fixes. This limit does not
mean that the pull request passes. After the third round, the author examines
the approach with a maintainer.

### Numerical errors

A revert does not correct results that users already calculated. If a fix
corrects a wrong number, record these items in the `Fixed` section of
`CHANGELOG.md`:

- The affected versions.
- The input conditions that cause the error.
- The size of the error, if it is known.

## Fortran binding

When updating the Fortran binding, make sure that the `LibrpaOptions` derived
type in `binding/fortran/librpa_f03.f90` matches the definition in
`include/librpa_options.h`.

After modifying `librpa_f03.f90`, run the stub-conversion script so that the
stub module stays synchronized with the main binding module.

```bash
cd binding/fortran
../../utilities/convert_fortran_module_to_stub.py librpa_f03.f90 librpa_f03_stubs.f90
```

## Code style

LibRPA does not enforce a strict project-wide code style. However, the repository
does provide a `.clang-format` file for formatting C and C++ code with
`clang-format`.

The goal is to avoid spending review time on personal formatting preferences,
such as whether an opening brace should be placed on a new line. The
`.clang-format` settings naturally encode some preferences, but the main purpose
is to make formatting mechanical and consistent.

Use `clang-format` for new C and C++ code. Avoid reformatting old code unless you
are making substantial changes in the same area, or unless the commit is
explicitly dedicated to formatting cleanup. This keeps functional diffs focused
and easier to review. Modern editors with language-server support should be able
to pick up the repository formatting rules automatically.

For naming, follow these general conventions:

- Use `UpperCamelCase` for classes and structs.
- Use `snake_case` for variables, object instances, functions, and namespaces.
- Use `SCREAMING_SNAKE_CASE` for constants.

## Profiling

When profiling the driver, set `output_level = critical` to reduce output
overhead while keeping essential results and the timing report visible.

## Source code structure of LibRPA

```
.
|-- binding
|   `-- fortran       : Fortran bindings and binding tests
|-- cmake             : CMake find modules and build helper scripts
|-- docs              : Sphinx/MyST user and developer documentation
|-- driver            : Command-line driver, input parsing, and data readers
|   |-- reader        : Shared input readers and the experimental LibBSE file interface
|   `-- tasks         : Driver task implementations
|-- examples
|   `-- build         : Example build scripts for common platforms
|-- include           : Public C and C++ API headers
|-- regression_tests  : Regression suite definitions, test cases, and references
|-- src               : Source code for the library target
|   |-- api           : Public API implementation and handler/dataset glue
|   |-- core          : Physics objects and algorithms: basis, RPA, EXX, GW, symmetry
|   |   `-- qsgw      : Fixed-basis QSGW task, checkpoints and numerical operations
|   |-- elpa          : ELPA eigensolver integration
|   |-- gpu           : Optional CUDA/HIP device and linear-algebra adapters
|   |-- interface     : Thin interfaces to external codes, e.g. BLAS, LAPACK, and ScaLAPACK 
|   |-- io            : Filesystem, ELSI/GW I/O, and stream-printing helpers
|   |-- math          : Utilities of matrix, vector, interpolation, fitting, etc.
|   |-- mpi           : MPI, BLACS, and k-point/process-grid helpers
|   |-- utils         : Constants, errors, profiling, memory, and configured build information
|   `-- test          : C++ unit and MPI tests for library components
|-- thirdparty        : Bundled external libraries used when not supplied by the user
`-- utilities         : Standalone conversion, consistency-check, and maintenance tools
```

The public API path is `include/librpa*.h(pp)`, implemented under `src/api`.
The driver path is `driver/main.cpp` with tasks implemented under `driver/tasks`
and input preparation in `driver/read_data.cpp`. Shared input readers live in
`driver/reader` and receive their handler, options, and parsing state explicitly.
The `librpa_file_reader` target contains only these readers and links to `rpa_lib`;
it is available with `LIBRPA_ENABLE_DRIVER=OFF`. It is always a static library
with position-independent code and is not installed. The driver may use its private
headers, and a file reader API is exposed through `include/file_reader.hpp`.

`include/file_reader.hpp` is an experimental interface for LibBSE built from a
LibRPA source tree. It returns `librpa_int::Dataset` and includes headers under
`src/`, so it is an exception to the rules on public headers and on including
`src/core` below. It is not installed and is not part of the stable public API.
Do not add new users of it; new external programs should use the public API.

## A few C++ guidelines

General coding guidelines:

- Avoid forward declarations for concrete classes and structs; include the
  header that provides the necessary definition.
- Prefer RAII, standard containers, references, and `const` correctness; make
  ownership and mutation explicit.
- Treat MPI layout, rank ownership, and collectives as part of the program
  behavior; check both serial and representative multi-rank cases when they may
  be affected.
- Use clear, explicit control flow for indexing, basis mappings, matrix layouts,
  and symmetry logic. Add a short comment when the physical or parallel
  assumption is not obvious from the code.

Below are a few specific guidelines for LibRPA:

- Library components (non-test code under `src/`) should remain host-agnostic:
  they must not assume that input data follows the convention of any particular
  host program. The host should communicate its conventions through the public
  API, for example via `set_basis_convention`; LibRPA internals should use only the
  parsed convention values when interpreting input data. The standalone driver follows
  the same rule by reading conventions from input files and passing them through
  the API. For user convenience, input files may still provide producer presets,
  such as a program name, as shorthand for a known set of conventions.
- When adding code in `src/core`, consider whether reusable pieces belong in
  utility components such as `src/math`, `src/mpi`, `src/io`, or `src/utils`;
  implement those pieces in the appropriate place and assemble the core
  algorithm under `src/core`.
- Source files outside `src/core` should not include headers from `src/core`;
  C APIs and dataset instances in `src/api` are the exceptions.
- Implement public behavior in the C API first; the C++ API
  (`src/api/librpa.cpp`) and Fortran bindings (`binding/fortran`) should wrap
  that C layer.
- Files in `src/interface` should not include internal headers outside
  `src/interface`.
- In principle, do not read input files in library code (`src/`). Restart
  check-point files generated by the library itself are the only exception.
- Use `src/io/stl_io_helper.h` when printing STL containers.
- Driver code (`driver/`) should not directly modify internal objects owned by
  a `LibrpaHandler`. Input parsing and output retrieval should go through
  public APIs. The only exception is an early task prototype; in that case,
  helper functions that mutate internal objects should remain `static` in the
  prototype task source, `driver/tasks/{proto}.cpp`.

## Adding new runtime options

- Prefer `output_` or `use_` prefixes for `LibrpaSwitch` options when they
  describe the behavior clearly; use another verb prefix when it is more precise,
  such as `read_sigc_mat_rf`.
- Option with a few available values can either be an `int` type starting with `option_`,
  or a dedicated `enum` type defined in `librpa_enums.h`.
- Try to keep the option name short, favorably no longer than 25 characters.
- Doxygen docstrings in `driver/driver.h` and `include/librpa_options.h` are
  the source of truth for runtime-parameter documentation. Put the explanation,
  default value, status, version information, and deprecation note there.
- Add the option to the appropriate block in
  `docs/user_guide/runtime_parameters.yml`; this controls where the generated
  user-guide table places the keyword.
- Use `utilities/check_librpa_options.py` to cross-check the consistency between the C
  struct, Fortran interface and high-level wrapper of `LibrpaOptions`.
