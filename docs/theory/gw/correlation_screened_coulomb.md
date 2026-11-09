# Correlation part of the screened Coulomb interaction

At each momentum and imaginary frequency, LibRPA constructs the screened
interaction from a Hermitian independent-particle response, for example the
one in the [conventional RPA](../rpa/conventional.md#response). This note covers
`option_dielect_func = 0` with `replace_w_head = false`. The
transform of $W^{\mathrm{c}}$ to real space is in
[this note](../numerical_methods/wr_real_space.md).

| Symbol | Definition |
| :--- | :--- |
| $V$ | Coulomb matrix used in the dielectric matrix |
| $V_\mathrm{t}$ | Cutoff Coulomb matrix |
| $\chi_0$ | Independent-particle response |
| $\tilde{\varepsilon}$ | Symmetrized dielectric matrix |

On the retained Coulomb subspace, the Dyson equation is

$$
W = V + V\chi_0 W,
$$

which gives $W = \varepsilon^{-1}V$ with the dielectric matrix
$\varepsilon = I - V\chi_0$. This matrix is not Hermitian. The similarity
transformation with $V^{1/2}$ gives the Hermitian symmetrized dielectric matrix

$$
\tilde{\varepsilon} = V^{-1/2}\varepsilon V^{1/2} = I-V^{1/2}\chi_0 V^{1/2},
$$

and the screened interaction becomes $W = V^{1/2}\tilde{\varepsilon}^{-1}V^{1/2}$.
LibRPA uses $V$ for the screening and the cutoff Coulomb matrix for the
external factors. Subtraction of the cutoff bare interaction gives

$$
W^{\mathrm{c}} = V_\mathrm{t}^{1/2}(\tilde{\varepsilon}^{-1}-I)V_\mathrm{t}^{1/2}.
$$

When the two Coulomb matrices are equal, this expression reduces to
$W^{\mathrm{c}}=W-V$.
