# LibRPA review guide

LibRPA fails when it prints a plausible number that is wrong. Rank findings by
this risk.

## Mark as Important

- Interface conventions: basis (AO, ABF, Coulomb), row or column order, transpose
  or conjugate, normalization, units, k or q mapping, Bloch phase.
- Parallel errors: data ownership, double counting, a root-only failure that
  blocks other ranks, `int` overflow in descriptors, stdout output from ranks
  other than root.
- Numerical failures: overflow or NaN for valid input, empty blocks.
- Tests that do not protect the claim: a test of a separate reference
  implementation, a check of column count without values, a parser that drops
  part of a number.
- Changes to existing references, tolerances, or comparators without a physical
  reason.
- Code in `src/` that depends on `driver/`, and other breaks of the rules in
  `docs/develop/develop_tips.md`.
- New inputs, file formats, units, switches, or changed defaults without
  documentation.
- New code that repeats an existing helper. Give the name of the helper.
- A pull request that declares a lower risk tier than its changes need.
- Text that an agent left in the code, such as report labels.

## Do not mention

- Formatting that `clang-format` handles.
- The contents of `regression_tests/testcases/` and `thirdparty/`.

Give a maximum of three Nits.

## Evidence

- Give the file and line on the actual execution path. Do not guess from a name.
- Before calling a guard unnecessary, show the upstream invariant.
- Separate a demonstrated error from a conditional concern.
- For a bug fix, find the root cause and examine similar code for the same bug.
- Tag a bug as Pre-existing if the pull request did not add it.

## Summary

Start with the number of Important findings and the number of Nits. If there is
nothing to fix, say so.
