# Using LibRPA driver with FHI-aims dataset

This tutorial will guide you through the process of setting up and running a calculation using FHI-aims for geometry and SCF computations and LibRPA for RPA energy computation.
We will use an H2O molecule as the test case (the setup files can be found [here](https://github.com/AESM-Group/LibRPA/tree/master/regression_tests/testcases/rpa_aims_mole_H2O)).

## 1. **Prerequisites**
Before starting, ensure that you have:
- [FHI-aims](https://fhi-aims.org/get-the-code-menu/get-the-code) installed for running geometry optimizations and generating inputs for LibRPA.
- LibRPA [installed](../../../user_guide/install.md) and ready for use.

## 2. **FHI-aims Input Files**
These files are required to run the initial SCF and geometry optimization using FHI-aims:
- **`control.in`**: This file contains control parameters for FHI-aims. Below is an example:
  ```text
  xc    pbe
  total_energy_method rpa
  frequency_points 60

  k_grid 1 1 1

  output librpa
  ```

- **`geometry.in`**: Contains the geometry of the system. For H2O:
  ```text
  lattice_vector    20.00000     0.00000     0.00000
  lattice_vector     0.00000    20.00000     0.00000
  lattice_vector     0.00000     0.00000    20.00000
  atom        -0.07712649        0.00000000        1.49704522  O
  atom         0.82002231        0.00000000        1.86358518  H
  atom         0.07269418       -0.00000000        0.53972961  H
  ```
Run FHI-aims, and after the calculation is completed, the dataset files for LibRPA driver will be exported.

## 3. **LibRPA Input File**

**`librpa.in`** specifies the parameters for the RPA calculation. Example:

```text
task = rpa
nfreq = 16
```

## 4. **Running the RPA Calculation with LibRPA**
After obtaining the output files from FHI-aims and setting up the parameters in `librpa.in`, you can use LibRPA to calculate the RPA correlation energy:
```bash
mpirun -np 4 /path/to/LibRPA/build/chi0_main.exe > LibRPA.out
```

## 5. **LibRPA Output**
After the LibRPA calculation is completed, an output file (`LibRPA.out`) will be generated, which contains the results of the RPA correlation energy as well as the contributions from each k-point (in Hartree).

```text
RPA correlation energy (Hartree)
| Weighted contribution from each k:
| (         0,         0,         0): (-0.306259,0)
| Total EcRPA:       -0.306258516
```
