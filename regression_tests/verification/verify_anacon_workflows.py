#!/usr/bin/env python3
"""Exercise selected continuation through the H2O GW/QSGW public API paths.

This strict selector control is single-rank; MPI reproducibility is a separate
question because independent upstream numerical builds need not be bitwise equal.

Default and explicit ridge must agree; Padé and AAA must give distinguishable
results on the same non-rational molecular input. Repeat with unequal source
and final resampling counts. The analytic component tests supply independent
formula oracles; this companion checks propagation, not material accuracy.
"""

import argparse
import csv
import json
from pathlib import Path
import re
import sys
import tarfile

import numpy as np

from verify_qsgw_restart import run, stage_input

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "utilities"))
from read_spectral_function import SpectralFunctionBinary


def outputs(directory, task):
    if task == "qsgw":
        data = np.atleast_2d(np.loadtxt(directory / "librpa.d/qsgw_eigenvalues.dat"))
        # Include both the grid and the direct band path.
        result = {"eigenvalues_eV": data[:, 8]}
        with (directory / "librpa.d/qsgw_qp.csv").open() as source:
            rows = list(csv.DictReader(source))
        for name in ("eqp_eV", "re_sigma_c_effective_eV"):
            result[name] = np.array([float(row[name]) for row in rows])
    else:
        rows = []
        for line in (directory / "energy_qp").read_text().splitlines():
            fields = line.split()
            if len(fields) == 4 and fields[0].isdigit():
                rows.append(float(fields[-1]))
        result = {"qp_Ha": np.array(rows)}
        spectral = SpectralFunctionBinary.from_file(directory / "librpa.d/spectral_function_kgrid.dat")
        for name in ("specfunc", "resigc", "imsigc"):
            result[name] = np.array(getattr(spectral, name))
    for name, array in result.items():
        assert array.size and np.all(np.isfinite(array)), (directory, name)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--analyze-only", action="store_true", help="reuse existing run outputs")
    parser.add_argument("--ranks", type=int, choices=(1,), default=1,
                        help="single-rank selector control; MPI reproducibility is separate")
    parser.add_argument("--mpiexec", default="mpirun")
    args = parser.parse_args()
    work = args.work_dir.resolve()
    if not args.analyze_only:
        work.mkdir(parents=True, exist_ok=False)
    cases = Path(__file__).resolve().parents[1] / "testcases"
    report = {}
    for task, fixture in (("g0w0", "g0w0_aims_mole_H2O_libri"),
                          ("qsgw", "qsgw_band_aims_mole_H2O_libri")):
        home = work / task
        if not args.analyze_only:
            home.mkdir()
            with tarfile.open(cases / fixture / "dataset.tar.gz") as archive:
                archive.extractall(home, filter="data")
        template = home / "template.in"
        original = (cases / fixture / "librpa/librpa.in").read_text()
        if not args.analyze_only:
            template.write_text(re.sub(r"(?m)^\s*anacon_method\s*=.*$", "", original))
        task_results = {}
        for resample in (False, True):
            common = {"nfreq": 16, "n_params_anacon": 8,
                      "n_params_anacon_resample": 10,
                      "anacon_nfreq": 12 if resample else -1,
                      "option_qpe_solver": 2, "output_energy_qp": "true",
                      "qsgw_min_iter": 1, "qsgw_max_iter": 1}
            if task == "g0w0":
                common.update(output_gw_spec_func="true", sf_omega_start=-1,
                              sf_omega_end=1, sf_omega_step=0.5)
            results = {}
            for method in ("default", "ridge", "pade", "aaa"):
                controls = {key: str(value) for key, value in common.items()}
                if method != "default":
                    controls["anacon_method"] = method
                directory = home / f"{'resampled' if resample else 'source'}-{method}"
                if not args.analyze_only:
                    stage_input(template, home / "dataset", directory, controls)
                    run(args.binary.resolve(), args.mpiexec, args.ranks, directory)
                results[method] = outputs(directory, task)
            metrics = {}
            for name in results["ridge"]:
                np.testing.assert_allclose(results["default"][name], results["ridge"][name],
                                           rtol=0, atol=1e-10)
                metrics[name] = {
                    method: float(np.max(np.abs(results[method][name] - results["ridge"][name])))
                    for method in ("default", "pade", "aaa")}
            # Guard QP selection independently of the spectral/static outputs.
            for method in ("pade", "aaa"):
                qp_name = "qp_Ha" if task == "g0w0" else "eqp_eV"
                assert metrics[qp_name][method] > 1e-7, (task, resample, method, qp_name)
                other = ("specfunc", "resigc", "imsigc") if task == "g0w0" else ("eigenvalues_eV",)
                assert max(metrics[name][method] for name in other) > 1e-7, (task, resample, method)
            report[f"{task}-resample-{resample}"] = metrics
            task_results[resample] = results["ridge"]
        for name in task_results[False]:
            delta = float(np.max(np.abs(task_results[False][name] - task_results[True][name])))
            if task == "qsgw" and name == "eigenvalues_eV":
                # Existing task behavior: static QSGW uses the source grid;
                # anacon_nfreq resamples only entering-state GW diagnostics.
                np.testing.assert_allclose(task_results[False][name], task_results[True][name],
                                           rtol=0, atol=1e-10)
            else:
                assert delta > 1e-9, (task, name, "resampling was ineffective")
            report[f"{task}-resampling-delta-{name}"] = delta
    (work / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
