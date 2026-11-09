#!/usr/bin/env python3
"""Compare standalone, shared-GW and tracked current-HR cRPA numerically.

Reuse the bundled four-state full-BZ SOC model without another dataset archive.
Fractional response weights must give U distinct from both bare V and full W.
Two current-HR QSGW updates are compared with a one-plus-one restart and with
one/four-rank runs. These are finite-model controls, not material acceptance.
"""

import argparse
import json
from pathlib import Path
import re
import tarfile

import numpy as np

from verify_qsgw_restart import compare, run, stage_input


HA_TO_EV = 27.211386245988  # The bundled model and constants_choice=internal.
PDOS_GRID_EV = np.linspace(-20, 20, 201)
PDOS_SIGMA_EV = .15


def orbital_analysis(directory, data, metadata, overlap, ao_counts, shell_l, kweights):
    """Reconstruct this s-shell fixture's complete charge weights and PDOS from C/S."""
    assert np.all(shell_l == 0), shell_l
    ns, nk, nb = data["E"].shape
    nao, nspinor = int(sum(ao_counts)), metadata["n_spinor"]
    offsets = np.r_[0, np.cumsum(ao_counts)]
    ng = len(ao_counts)  # All radial s shells on one atom form one charge group.
    expected, mulliken, lowdin = [], np.empty((ns, nk, nb, ng)), np.empty((ns, nk, nb, ng))
    for k in range(nk):
        eigenvalues, eigenvectors = np.linalg.eigh(overlap[k])
        assert eigenvalues.min() > 0
        root = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.conj().T
        for spin in range(ns):
            c = data["C"][spin, k].reshape(nspinor, nao, nb)
            sc = np.einsum("ij,sjb->sib", overlap[k], c)
            sqrt_sc = np.einsum("ij,sjb->sib", root, c)
            wm = np.real(c.conj() * sc).sum(axis=0)
            norm = wm.sum(axis=0)
            wl = (abs(sqrt_sc)**2).sum(axis=0) / norm
            for group in range(ng):
                mulliken[spin, k, :, group] = wm[offsets[group]:offsets[group + 1]].sum(axis=0)
                lowdin[spin, k, :, group] = wl[offsets[group]:offsets[group + 1]].sum(axis=0)
            for band in range(nb):
                for group in range(ng):
                    energy = data["E"][spin, k, band]
                    expected.append([
                        spin, k, band, group, energy * HA_TO_EV,
                        (energy - metadata["fermi_ha"]) * HA_TO_EV,
                        data["occupations"][spin, k, band], kweights[k], norm[band],
                        mulliken[spin, k, band, group], lowdin[spin, k, band, group],
                    ])
    actual = np.loadtxt(directory / "crpa_orbital_weights.dat", ndmin=2)
    np.testing.assert_allclose(actual, np.asarray(expected), rtol=0, atol=1e-10)
    # Sum every loaded pole, with k weights and no occupation/spin multiplier.
    relative_energies = (data["E"] - metadata["fermi_ha"]) * HA_TO_EV
    gaussian = np.exp(-.5 * ((PDOS_GRID_EV[None, :, None, None] -
                             relative_energies[:, None, :, :]) / PDOS_SIGMA_EV)**2)
    gaussian *= kweights[None, None, :, None] / (np.sqrt(2 * np.pi) * PDOS_SIGMA_EV)
    pm = np.einsum("sekb,skbg->seg", gaussian, mulliken)
    pl = np.einsum("sekb,skbg->seg", gaussian, lowdin)
    expected_pdos = [[spin, energy, group, pm[spin, e, group], pl[spin, e, group]]
                     for spin in range(ns) for e, energy in enumerate(PDOS_GRID_EV)
                     for group in range(ng)]
    actual_pdos = np.loadtxt(directory / "crpa_pdos.dat", ndmin=2)
    np.testing.assert_allclose(actual_pdos, np.asarray(expected_pdos), rtol=0, atol=1e-10)
    assert np.max(pl) > 1e-3, pl
    return {"orbital_Mulliken": mulliken, "orbital_Lowdin": lowdin,
            "orbital_PDOS_Mulliken": pm, "orbital_PDOS_Lowdin": pl}


def snapshot(directory):
    with np.load(directory / "crpa_model.npz") as archive:
        data = {key: archive[key].copy() for key in archive.files}
    metadata = json.loads(data["metadata_json"].tobytes().decode())
    with np.load(directory / metadata["static_file"]) as archive:
        overlap = archive["S"].copy()
        ao_counts, shell_l, kweights = (archive[key].copy()
                                       for key in ("ao_counts", "shell_l", "k_weights"))
    coefficients = data["C"]
    probabilities = np.empty_like(data["A_weights"])
    for spin in range(coefficients.shape[0]):
        for k in range(coefficients.shape[1]):
            # The cRPA model uses component-major rows, unlike native H/S/r.
            metric = np.kron(np.eye(metadata["n_spinor"]), overlap[k])
            c = coefficients[spin, k]
            norm = np.real(np.diag(c.conj().T @ metric @ c))
            phi = data["output_Phi"][spin, k]
            np.testing.assert_allclose(phi.conj().T @ metric @ phi,
                                       np.eye(phi.shape[1]), rtol=0, atol=1e-10)
            projector = phi.conj().T @ metric @ c
            np.testing.assert_allclose(projector, data["output_P"][spin, k],
                                       rtol=0, atol=1e-10)
            probabilities[spin, k] = np.sum(abs(projector)**2, axis=0) / norm
    np.testing.assert_allclose(probabilities, data["A_weights"], rtol=0, atol=1e-10)
    assert np.any((probabilities > 1e-4) & (probabilities < 1-1e-4)), probabilities
    numerical = {key: value for key, value in data.items()
                 if key in ("nu", "E", "occupations", "A_weights", "output_Phi")
                 or key.startswith(("v_", "U_", "W_"))}
    for key, value in numerical.items():
        assert np.all(np.isfinite(value)), (directory, key)
    for key in data:
        if key.startswith("U_"):
            assert np.max(abs(data[key] - data["v_" + key[2:]])) > 1e-6, key
    if (directory / "crpa_orbital_weights.dat").exists():
        numerical.update(orbital_analysis(directory, data, metadata, overlap, ao_counts,
                                          shell_l, kweights))
    return numerical


def compare_interactions(reference, actual):
    errors = {}
    assert reference.keys() == actual.keys(), (reference.keys(), actual.keys())
    # Eigenvector gauges may differ. Probabilities and local interaction tensors
    # in the transported physical frame must retain the same numerical values.
    for key in reference:
        assert reference[key].shape == actual[key].shape, key
        if key == "output_Phi":
            continue
        tolerance = (1e-10 if key in ("nu", "E", "occupations", "A_weights")
                     or key.startswith("orbital_") else 1e-10 / HA_TO_EV)
        np.testing.assert_allclose(reference[key], actual[key], rtol=0, atol=tolerance,
                                   err_msg=key)
        errors[key] = float(np.max(abs(reference[key] - actual[key])))
    return errors


def crpa_states(directory):
    return {int(path.name.removeprefix("state_")): snapshot(path)
            for path in sorted((directory / "librpa.d/crpa").glob("state_*"))}


def run_rank(binary, mpiexec, ranks, root, template, dataset):
    home = root / f"np{ranks}"
    home.mkdir()
    common = {
        "g0w0_hr_unit": "Ry", "tfgrids_freq_min": "0.4", "tfgrids_freq_max": "4.0",
        "use_fullcoul_eps": "true", "use_fullcoul_wc": "true",
        "crpa_species_labels": "X", "crpa_correlated_species": "X",
        "crpa_parent_orbitals": "s", "crpa_output_orbitals": "s",
        "crpa_overlap_file": "srs1_nao.csr", "crpa_orbital_bands": "0 1 2 3",
        "crpa_response_mode": "orbital_weights", "crpa_output_analysis": "true",
        "crpa_pdos_reference": "fermi", "crpa_pdos_min_ev": "-20", "crpa_pdos_max_ev": "20",
        "crpa_pdos_step_ev": ".2", "crpa_pdos_sigma_ev": str(PDOS_SIGMA_EV),
        "crpa_output_model": "true", "crpa_output_raw": "true",
        "crpa_output_static": "false", "crpa_output_format": "npz",
    }
    standalone, shared = home / "standalone", home / "shared"
    for directory, controls in (
        (standalone, {"task": "crpa"}),
        (shared, {"task": "g0w0", "crpa_enabled": "true"}),
    ):
        stage_input(template, dataset, directory, {**common, **controls})
        run(binary, mpiexec, ranks, directory)
    alone = snapshot(standalone / "librpa.d")
    together = crpa_states(shared)
    assert set(together) == {0}, together.keys()
    for values in (alone, together[0]):
        assert {"orbital_Mulliken", "orbital_Lowdin", "orbital_PDOS_Mulliken",
                "orbital_PDOS_Lowdin"} <= values.keys()
    # The shared build intentionally returns U only; require every other result
    # key, including every site/spin tensor, rather than comparing an intersection.
    equality = compare_interactions({key: value for key, value in alone.items()
                                     if not key.startswith("W_")}, together[0])
    for key in alone:
        if key.startswith("U_"):
            assert np.max(abs(alone[key] - alone["W_" + key[2:]])) > 1e-6, key

    # cRPA's standalone all-node continuation must not inherit the GW fit count.
    continuation = []
    for count in (-1, 2):
        directory = home / f"static-nparams-{count}"
        stage_input(template, dataset, directory, {
            **common, "task": "crpa", "crpa_output_static": "true",
            "crpa_real_frequencies_ev": "1.0", "n_params_anacon": str(count),
        })
        run(binary, mpiexec, ranks, directory)
        values = np.loadtxt(directory / "librpa.d/crpa_continuation_summary.dat", ndmin=2)
        assert values.size and np.all(np.isfinite(values)), directory
        continuation.append(values)
    np.testing.assert_allclose(continuation[0], continuation[1], rtol=0, atol=1e-10)

    continuous, initial, resumed, uncoupled = (
        home / name for name in ("continuous", "initial", "resumed", "uncoupled"))
    qsgw = {
        **common, "task": "qsgw", "crpa_enabled": "true", "crpa_track_orbitals": "true",
        "qsgw_update_hartree": "false",  # Fixed-Hartree control; no bare metric in this model.
        "qsgw_headwing_velocity": "current_hr", "qsgw_output_qp": "true",
        "qsgw_write_iteration_matrices": "true", "qsgw_cut_mode": "0",
        "qsgw_checkpoint_file": "checkpoint.bin",
    }
    for directory, end in ((continuous, 2), (initial, 1), (resumed, 2), (uncoupled, 2)):
        settings = {**qsgw, "qsgw_min_iter": str(end), "qsgw_max_iter": str(end)}
        if directory == resumed:
            settings["qsgw_restart_file"] = str(initial / "librpa.d/checkpoint.bin")
        if directory == uncoupled:
            settings["crpa_enabled"] = "false"
        stage_input(template, dataset, directory, settings)
        run(binary, mpiexec, ranks, directory)
    states, restored = crpa_states(continuous), crpa_states(resumed)
    assert set(states) == {0, 1, 2}, states.keys()
    assert set(restored) == {1, 2}, restored.keys()
    restart = {state: compare_interactions(states[state], restored[state]) for state in restored}
    assert np.max(abs(states[2]["E"] - states[0]["E"])) > 1e-6
    assert np.max(abs(states[2]["A_weights"] - states[0]["A_weights"])) > 1e-6
    assert np.max(abs(states[2]["orbital_PDOS_Lowdin"] - states[0]["orbital_PDOS_Lowdin"])) > 1e-6
    return {"shared": together[0], "qsgw": states}, {
        "standalone_shared": equality, "crpa_restart": restart,
        "static_count_independence": float(np.max(abs(continuation[0] - continuation[1]))),
        "maximum_U_minus_V_ha": max(float(np.max(abs(value - alone["v_" + key[2:]])))
                                       for key, value in alone.items() if key.startswith("U_")),
        "maximum_U_minus_W_ha": max(float(np.max(abs(value - alone["W_" + key[2:]])))
                                       for key, value in alone.items() if key.startswith("U_")),
        "qsgw_weight_change": float(np.max(abs(states[2]["A_weights"] - states[0]["A_weights"]))),
        "qsgw_orbital_pdos_change_per_ev": float(np.max(abs(
            states[2]["orbital_PDOS_Lowdin"] - states[0]["orbital_PDOS_Lowdin"]))),
        "qsgw_eigenvalue_change_ha": float(np.max(abs(states[2]["E"] - states[0]["E"]))),
        "qsgw_restart": compare(continuous, resumed, 2, 2),
        "coupled_uncoupled_qsgw": compare(continuous, uncoupled, 1, 2),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--mpiexec", default="mpirun")
    parser.add_argument("--ranks", type=int, action="append")
    args = parser.parse_args()
    root = args.work_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    fixture = Path(__file__).resolve().parents[1] / "testcases/g0w0_scipy_fourstate_soc_full_native_headwing"
    with tarfile.open(fixture / "dataset.tar.gz") as archive:
        archive.extractall(root, filter="data")
    dataset = root / "dataset"
    # The fixture's H is in eV. QSGW's native baseline consumes Ry. Convert only
    # the extracted numerical values; leave the committed fixture unchanged.
    hamiltonian = dataset / "hrs1_nao.csr"
    hamiltonian.write_text(re.sub(
        r"\(([-+0-9.eE]+),([-+0-9.eE]+)\)",
        lambda match: f"({float(match[1])*2/HA_TO_EV:.17e},{float(match[2])*2/HA_TO_EV:.17e})",
        hamiltonian.read_text()))
    first, report = None, {}
    for ranks in args.ranks or (1, 4):
        numerical, metrics = run_rank(args.binary.resolve(), args.mpiexec, ranks, root,
                                      fixture / "librpa/librpa.in", dataset)
        if first is None:
            first = numerical
        else:
            metrics["rank_comparison"] = {
                "shared": compare_interactions(first["shared"], numerical["shared"]),
                "qsgw": {state: compare_interactions(first["qsgw"][state], values)
                          for state, values in numerical["qsgw"].items()},
            }
        report[ranks] = metrics
        print(json.dumps({ranks: metrics}, indent=2), flush=True)
    (root / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
