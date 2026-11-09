# Correlation part of the screened Coulomb interaction

At each momentum and imaginary frequency, LibRPA constructs the screened
interaction from a Hermitian independent-particle response. This note covers
`option_dielect_func = 0` with `replace_w_head = false`.

| Symbol | Definition |
| :--- | :--- |
| $V$ | Coulomb matrix used in the dielectric matrix |
| $V_\mathrm{t}$ | Cutoff Coulomb matrix |
| $\chi_0$ | Independent-particle response |
| $\tilde{\varepsilon}$ | Symmetrized dielectric matrix |

On the retained Coulomb subspace, the Dyson equation is

$$
W = V + V\chi_0 W.
$$

Substitution of $W=V^{1/2}XV^{1/2}$ gives

$$
X = I + V^{1/2}\chi_0 V^{1/2} X,
\qquad
X = (I-V^{1/2}\chi_0 V^{1/2})^{-1}.
$$

LibRPA uses $V$ for the screening and the cutoff Coulomb matrix for the
external factors. Subtraction of the cutoff bare interaction gives

$$
\tilde{\varepsilon} = I-V^{1/2}\chi_0 V^{1/2},
\qquad
W_c = V_\mathrm{t}^{1/2}(\tilde{\varepsilon}^{-1}-I)V_\mathrm{t}^{1/2}.
$$

When the two Coulomb matrices are equal, this expression reduces to
$W_c=W-V$. At zero response it gives $W_c=0$.
