# Correlation part of the screened Coulomb interaction

At each momentum and imaginary frequency, LibRPA constructs the screened
interaction from a Hermitian independent-particle response. This note covers
`option_dielect_func = 0` with `replace_w_head = false`.

| Symbol | Definition | Dense-path variable |
| :--- | :--- | :--- |
| $V$ | Coulomb matrix used in the dielectric matrix | `Vq_all`, before diagonalization |
| $T$ | Positive square root of the cutoff Coulomb matrix | `sqrtVqcut_all` |
| $S$ | Positive square root of $V$ | `sqrtVq_all` |
| $\chi_0$ | Independent-particle response | `chi0fq_all` |
| $\epsilon$ | Symmetrized dielectric matrix | `eps_fq`, before inversion |

On the retained Coulomb subspace, the Dyson equation is

$$
W = V + V\chi_0 W.
$$

Substitution of $V=S^2$ and $W=SXS$ gives

$$
X = I + S\chi_0 S X,
\qquad
X = (I-S\chi_0 S)^{-1}.
$$

LibRPA uses $V$ for the screening and the cutoff Coulomb matrix for the
external factors. Subtraction of the cutoff bare interaction gives

$$
\epsilon = I-S\chi_0 S,
\qquad
W_c = T(\epsilon^{-1}-I)T.
$$

When the two Coulomb matrices are equal, this expression reduces to
$W_c=W-V$. At zero response it gives $W_c=0$.

The dense implementation uses column-major storage for all factors in these
products. Matrix square roots preserve that storage order. A row-major factor
would violate the multiplication interface even when its mathematical entries
are Hermitian (`src/core/epsilon.cpp`, `compute_Wc_freq_q`).

The `test_dense_wc_layout` test constructs positive-definite $S$ and $T$,
forms their squares with scalar products, and evaluates the formula with an
explicit two-by-two inverse. It checks every entry from both production paths
against this result. The cases include real matrices at Gamma, complex
conjugate matrices at opposite momenta, and zero response
(`src/test/test_epsilon.cpp`).
