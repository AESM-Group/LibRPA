# Regression tests

This folder contains regression test cases used to verify LibRPA features.
"Regression tests and verification scripts" in `docs/develop/develop_tips.md`
gives what goes in `testcases/` and `refs/`, and what goes in `verification/`.

The LibRI suite includes two ABACUS H2 cases: standalone Hartree compares the
complete complex AO matrix on the k-grid, and QSGW Hartree feedback compares
the AO and KS corrections, spectra and QP energies across two updates. Both use the same
regression runner and reference comparisons as the EXX cases. Their finite
input data and reference snapshots are not material-convergence benchmarks.
`verification/verify_hartree_band.py` additionally constructs a finite SciPy path
control from the same H2 H(R)/S(R) (with zero path Vxc), runs actual grid/path
GW/QSGW updates, and compares Hartree path projection, spectra, QP roots, MPI
and split restart. The generated path input is not an ABACUS NSCF result.

The LibRI CI also runs `verification/verify_qsgw_restart.py --case h2o-band` and
`verification/verify_qsgw_headwing.py --case aims-rotate` with four MPI ranks. These reuse
the bundled datasets to compare three continuous updates with a one-update
checkpoint followed by two resumed updates. They check fixed-basis matrices,
spectra, entering-state QP roots and live optical feedback. The QP comparison
also checks the entering spectrum, the QP equation and the numbered grid/path
energy tables. The native `current_hr` companion still requires an explicitly
supplied native S/r dataset; it is not covered by the aims-rotate CI case.
The live head/wing companion also repeats continuation from the identical
checkpoint. It saves that comparison before checking the independent continuous
trajectory, retaining the strict gates when either comparison fails.
CI fixes the Intel MPI/oneMKL execution branches with `I_MPI_CBWR=2` and
`MKL_CBWR=COMPATIBLE` for these trajectory comparisons; numerical gates and
physical input settings are unchanged.

For each test case `CASE`, the input files are located under `testcases/${CASE}`,
while the reference results are stored under `refs/${CASE}`.

## Run a single test case manually

The test cases in this directory must be run with the LibRPA driver enabled.
After building LibRPA with the driver, you can run a test case as follows:

```bash
# The LibRPA driver executable built
export myprog="/path/to/librpa/build/chi0_main.exe"
export CASE=case_1

cd testcases/${CASE}
tar -zxf dataset.tar.gz
cd librpa
# Adjust OMP_NUM_THREADS and MPI tasks
OMP_NUM_THREADS=1 mpirun -np 4 "$myprog" > librpa.out 2> librpa.err
```

After the run finishes, compare the standard output and any generated files with
the reference results under `refs/${CASE}/librpa`.

## Create a regression test case data

**Step 1**: create reference and testcase folder for the case
```bash
export CASE=case_new
mkdir -p {refs,testcases}/${CASE}/{dataset,librpa}
```

Step 2: prepare dataset
```bash
cd refs/${CASE}
cd dataset
# Create inputs and run dataset preparation program.
# For example, FHI-aims:
#     vim control.in; vim geometry.in; aims.x > aims.out
cd ..
# Compress dataset as dataset.tar.gz. Exclude all outputs, but keep all input files.
# For example, FHI-aims
tar --exclude aims.out -zchf dataset.tar.gz dataset
```

Step 3: prepare LibRPA input and run LibRPA to get reference data
```bash
mkdir librpa
cd librpa
cat > librpa.in << EOF
task = rpa
input_dir = ../dataset
# other parameters
# ...
EOF
mpirun -np 4 chi0_main > librpa.out
cd ..
```

Step 4: archive output to reference folder
```bash
# Copy dataset generation output for future reference
cp -a dataset/aims.out ../../refs/${CASE}/dataset/aims.out.export
cp -a librpa/librpa.out ../../refs/${CASE}/librpa/librpa.out
# Also copy other LibRPA output files if needed
# cp -a librpa/GW_band_spin_1.dat ../../refs/${CASE}/librpa/GW_band_spin_1.dat
```
Additional reference data (band structures and/or benchmark) from other programs
can be added to the reference `dataset` folder.
LibRPA run with different input or runtime configurations can also be placed and named appropriately
under `librpa` folder.

## Run automatic regression test and analysis

The automatic regression driver reads test definitions from `testsuite.xml`,
prepares a fresh workspace, runs selected test cases, and compares generated
outputs with the reference data under `refs/`.

The entry point for automatic test is the Python script `run_regression.py`.
To list the available test cases after applying build and runtime filters:

```bash
python3 run_regression.py list \
  --use-libri \
  -n 4 \
  --nthreads 1
```

To run tests and immediately perform the diff analysis:

```bash
python3 run_regression.py full /path/to/build/chi0_main.exe \
  --use-libri \
  -n 4 \
  --nthreads 1 \
  --mpiexec "mpirun --bind-to none" \
  --verbose \
  -o regression.log
```

The `full` mode writes normal terminal output and also records the same report
to the file selected by `-o/--output` (default: `regression.log`). The LibRPA
stdout/stderr of each case are stored separately in the workspace, for example
`workspace/testcases/${CASE}/librpa/librpa.out` and `librpa.err`.

Useful options:

- `-d WORKSPACE`: choose the run workspace, default `workspace`.
- `-f, --force`: reuse an existing workspace directory.
- `--only CASE [CASE ...]`: run only selected test case directories.
- `--exclude CASE [CASE ...]`: skip selected test case directories.
- `--mpiexec COMMAND`: choose an MPI launcher, including launcher options.
  For `mpirun` and `mpiexec`, `-np ${ntasks}` is added automatically unless
  the command already contains `-n`, `-np`, `--np`, or `--ntasks`.
- `--verbose`: print the exact command used for each test case.

You can also split execution and analysis into two steps. This is useful when
the calculations run on a batch node but the analysis is performed later:

```bash
python3 run_regression.py run /path/to/build/chi0_main.exe \
  --use-libri \
  -n 4 \
  --nthreads 1 \
  --mpiexec "mpirun --bind-to none" \
  --verbose

python3 run_regression.py analyze \
  --use-libri \
  -n 4 \
  --nthreads 1 \
  -o regression.log
```

### Coupled cRPA numerical control

After a LibRI build, run the standalone/shared-G0W0 and tracked current-HR QSGW
companion from this directory with NumPy and the matching MPI runtime:

```bash
python3 verification/verify_crpa_workflows.py --binary /path/to/build/chi0_main.exe \
  --work-dir /path/to/new-crpa-workspace
```

It reuses the committed four-state full-BZ SOC dataset and runs one/four MPI
ranks by default (`--ranks 1` selects one layout). The comparison checks
fractional endpoint probabilities independently from C/S/Phi, U distinct from
bare V and full W, standalone/shared U equality, two current-HR updates and
one-plus-one continuation, and unchanged GW trajectories when cRPA is enabled.
Orbital analysis is enabled for standalone/shared calculations and every
entering/final QSGW state. NumPy independently reconstructs live Mulliken and
Lowdin charge weights from exported C/S, and the complete Gaussian PDOS from
all poles and k weights; continuation and rank comparisons include these
quantities and require the PDOS to change with the electronic state. The
fixture has two radial s functions on one atom, so its grouped weights measure
total charge. These checks cover current-state output and borrowed numerical
views; they do not establish material orbital transport.
The full B space keeps the physical AO target fixed while its current-KS
probabilities change; band-reordering transport has separate component coverage.
CI invokes this companion after the LibRI regressions. Its JSON report contains
numerical differences; these finite-model checks do not establish material
convergence.

### Magnetic coupled cRPA control

The magnetic stage also provides `verification/verify_magnetic_crpa_workflows.py` with the
same `--binary`, `--work-dir` and optional repeatable `--ranks` arguments. It
reuses the full-BZ and paired-IBZ SOC archives, enables all three symmetry
routes, and compares two current-HR updates with weighted/tracked cRPA at
one/four ranks. QP roots and local tensors are compared directly; velocity,
self-energy and Hamiltonian matrices are aligned through the independently
exported complete eigenframes before comparison. Matrix errors retain each
component's native units; head, wing, velocity and energy matrices do not share
one unit. No separate dataset archive
or material-convergence claim is added. CI runs this after the full-grid
cRPA companion.

### Register an automatic test entry

After preparing `testcases/${CASE}` and `refs/${CASE}`, register the case in
`testsuite.xml`. The `directory` attribute must match the testcase/reference
directory name:

```xml
<group name="RPA correlation energy" prefix="RPA">
  <testcase name="Water molecule in box (aims input)" directory="rpa_aims_mole_H2O_libri">
    <build require_libri="true" />
    <run ntasks_disable="none"
         nthreads_disable="none"
         ntasks_enable="none"
         nthreads_enable="none"
    />
    <validate name="Correlation energy"
              file="librpa/librpa.out"
              regex="Total EcRPA: \s+([-+]?\d*\.\d+)"
              comparison="cmp_float.abs_diff(1e-4)"
    />
  </testcase>
</group>
```

Entry fields:

- `<group>` collects related cases. `name` is printed by the test driver.
- `<testcase name="..." directory="...">` gives the display name and the
  directory under both `testcases/` and `refs/`.
- `<labels disable="...">` optionally disables a case. Use `disable="true"` or
  a short reason such as `disable="experimental"`.
- `<labels refhash="...">` records the commit hash used to generate the
  reference data. It is informational metadata for maintainers.
- `<build require_libri="true">` skips the case unless `--use-libri` is passed.
- `<run>` configures runtime setup. Use `ntasks_enable` or `nthreads_enable` to
  allow only selected runtime MPI/threads layouts, and `ntasks_disable` or
  `nthreads_disable` to reject selected layouts. Values can be comma- or
  space-separated integers, or `none`. Use space- or comma-separated
  `copy_files` to copy extra files or directories from `testcases/${CASE}/librpa`
  into the run workspace `librpa` directory.
- `<validate>` adds one comparison. If `file` is omitted, the default is
  `librpa/librpa.out`. `regex` locates the reference/output block, `headers`
  and `rows` select table regions when needed, and `comparison` names a
  comparison function from `backend/comparisons`.

Multiple `<validate>` entries can be attached to one testcase when a run
produces several quantities that should be checked.

Useful `<validate>` attributes:

- `name`: human-readable label printed in the regression report.
- `file`: file pattern to search under both the test workspace and reference
  directory. The default is `librpa/librpa.out`.
- `file_test`, `file_refr`: optional file patterns for the test workspace and
  reference directory. Each one overrides `file` only for that side.
- `regex`: Python regular expression used to find the quantity or table. If the
  matched line has one capture group, that group is compared. If it has multiple
  capture groups, they are joined with spaces before comparison.
- `headers`: number of lines to skip after a matched `regex`, default `0`.
- `rows`: number of lines to extract after `headers`. Use this for table
  comparisons, for example a header regex followed by `headers="2"` and
  `rows="52"`.
- `occurences`: zero-based selector for repeated regex matches. The attribute
  name is intentionally written here as accepted by the current parser. Use
  `occurences="0"` for the first match, `occurences="2"` for the third match,
  or `occurences="0:3"` for matches 0 through 3.
- `comparison`: comparison function and arguments. Common choices are
  `cmp_float.abs_diff(1e-4)` for scalar values and
  `cmp_table.abs_diff(1e-4)` for numeric tables. Table comparisons also accept
  a `columns` argument, for example
  `cmp_table.abs_diff(1e-4, columns="3:8")`.
- `binary_extract`: numeric `struct` layout; for example, `binary_extract="2i(128d)"`
  reads two integers and 128 doubles, selecting only the doubles for comparison.
  Plain-text extraction is used when omitted.

## Reference update guidelines

- Do not edit reference output by hand. Reference data should match the commit
  recorded in the testcase `refhash` label in `testsuite.xml`, so it can be
  reproduced from that source snapshot, input setup, data files, and runtime
  environment.
- Update references sparingly. Code changes may perturb output, but if the
  quantities validated in `testsuite.xml` remain within tolerance, keep the
  existing reference data. That is the point of a regression test.
