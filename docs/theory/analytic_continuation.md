# Rational continuation of the correlation self-energy

The shared interface evaluates a complex rational fit and its analytic frequency
derivative. The self-energy samples, reference energies and GW/QSGW equations
are unchanged. Selecting ridge changes the fit evaluated on the real axis; it
does not constrain poles or residues.

## Ridge objective and derivative

Let $s_x=\max_i|z_i|$, $s_y=\max_i|\Sigma_i|$, $u_i=z_i/s_x$ and
$y_i=\Sigma_i/s_y$. A scale smaller than $10^{-30}$ is replaced by one.
For $N$ samples and requested coefficient count $p$, use $p=\min(p,N)$
(negative means $N$), $n_a=\lceil p/2\rceil$ and $n_b=p-n_a$.
Define $P(u)=\sum_{j=0}^{n_a-1}a_j u^j$ and
$Q(u)=1+\sum_{k=1}^{n_b}b_k u^k$. The code minimizes

$$
\sum_i|P(u_i)-y_iQ(u_i)|^2
+\lambda\sum_{j=1}^{n_a-1}j^2|a_j|^2
+\lambda w_b\sum_{k=1}^{n_b}k^2|b_k|^2.
$$

For explicitly selected ridge, the defaults are $\lambda=10^{-12}$ and $w_b=10$. All samples enter the
linearized objective; $a_0$ is unpenalized. This differs from a nonlinear
minimization of $|P/Q-y|$. The historical normal equations, partial-pivot
elimination and small diagonal-shift retries are retained.

The continued value and derivative are

$$
\widehat\Sigma(z)=s_y\frac{P(z/s_x)}{Q(z/s_x)},\qquad
\widehat\Sigma'(z)=\frac{s_y}{s_x}\frac{P'Q-PQ'}{Q^2}(z/s_x).
$$

The evaluator rejects $|Q|<10^{-12}$ and non-finite results. It does not clip a
denominator or fall back to Padé. A positive coefficient penalty does not
establish causality or bound the real-axis error.

## Padé, AAA and their consumers

Padé keeps the existing Thiele and barycentric fallback construction. AAA uses
greedy support selection and the smallest right singular vector of the complex
Loewner matrix $L_{ij}=(y_i-y_j)/(z_i-z_j)$, with barycentric form

$$
R(z)=\frac{\sum_jw_jy_j/(z-z_j)}{\sum_jw_j/(z-z_j)}.
$$

The quotient derivative is analytic, including its support-point limit.
Zero-weight supports are absent from the rational sums and remain included in
the residual check. An explicit support cap may stop before tolerance is met.

The existing GW solvers consume the selected value and derivative with the
same units and energy reference. Optional imaginary-grid resampling fits the
source grid, evaluates the intermediate grid, and fits the intermediate samples
with the same selected method. Spectral functions use the selected fit as well.

QSGW keeps its fixed-reference construction: diagonal elements are evaluated
at reference energies relative to the Fermi energy. Mode B off-diagonal terms
use zero relative frequency, then combine forward and conjugate reverse ordered
elements to form a Hermitian matrix. Mode A uses the two endpoint energies.
Each ordered element is fitted independently. Method selection changes neither
this staticization convention nor diagonalization, cut policy or mixing.

The standalone QSGW task keeps its existing static-update behavior: it fits
correlation matrices directly on the source frequency grid. `anacon_nfreq`
and `anacon_tfgrids_type` resample its entering-state GW quasiparticle
diagnostics, not its static Hamiltonian update. The low-level matrix builder
supports explicitly supplied resampling nodes; that capability does not imply
that the standalone task supplies them.
