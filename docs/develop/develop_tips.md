# Development Tips

## Git collaborative workflow

The recommended workflow is:

1. Fork the repository.
2. Create a topic branch for your changes.
3. Rebase your branch on `master` when it is ready for review.
4. Resolve any conflicts locally.
5. Open a merge request.
6. Address review comments.
7. Merge with a fast-forward merge.

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
