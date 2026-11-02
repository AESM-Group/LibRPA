# LibRPA review guide

LibRPA fails when it prints a plausible number that is wrong. Rank findings by
this risk.

## Mark as Important

- Interface conventions: basis (AO, ABF, Coulomb), row or column order, transpose
  or conjugate, normalization, units, k or q mapping, Bloch phase.
- Parallel errors:
  - Data ownership or double counting.
  - A root-only failure that blocks other ranks.
  - `int` overflow in descriptors.
  - A `lib_printf_coll` call that some ranks do not reach.
- Numerical failures: overflow or NaN for valid input, empty blocks.
- Tests that do not protect the claim: a test of a separate reference
  implementation, a check of column count without values, a parser that drops
  part of a number.
- Changes to existing references, tolerances, or comparators without a physical
  reason.
- A refactor that changes existing references, or a method change whose only
  evidence is agreement with the old results.
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
- For each finding, give the trigger, the consequence, and the change to make.
- Before calling a guard unnecessary, show the upstream invariant.
- Separate a demonstrated error from a conditional concern.
- For a bug fix, find the root cause and examine similar code for the same bug.
- Tag a bug as Pre-existing if the pull request did not add it.
- Check that the evidence supports the claim. See "Numerical evidence" in
  `docs/develop/develop_tips.md`.

## Rounds

- Give all instances of one error class in one finding.
- Do not repeat a finding that is already in a review thread.
- After the second round, examine only the new changes, the open findings, and
  the affected callers.
- Stop after the third round and ask the author to examine the approach. This
  stop does not mean that the pull request passes.

## Summary

Start with the number of Important findings and the number of Nits. If there is
nothing to fix, say so.
