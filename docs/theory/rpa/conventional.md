# Conventional k-space RPA

The conventional method evaluates the independent response at positive
imaginary frequencies by a direct sum over band transitions. It needs no
imaginary-time Green's function and no time-to-frequency transform. The Coulomb
matrices and the RPA energy routines are the same as in the space-time method.
For input options, see the
[user guide](../../user_guide/conventional_rpa.md).

## Transition density

Let $\phi_{Ii}$ be atomic orbital $i$ on atom $I$, and $P_{\mu}$ an auxiliary
function on atom $I$. The local RI coefficients $C^{\mu}_{Ii,Jj}(\mathbf{R})$
expand the product of $\phi_{Ii}$ in cell $\mathbf{0}$ and $\phi_{Jj}$ in cell
$\mathbf{R}$, with one term for the auxiliary functions on each atom. The Bloch
states are

$$
\psi_{sn\mathbf{k}}(\mathbf{r}) = \frac{1}{\sqrt{N_k}}
\sum_{I i \mathbf{R}} c_{sn\mathbf{k},Ii}\, e^{i\mathbf{k}\cdot\mathbf{R}}
\phi_{Ii}(\mathbf{r}-\mathbf{R}).
$$

For an initial state $(n,\mathbf{k})$ and a final state
$(m,\mathbf{k}+\mathbf{q})$ in spin channel $s$, the coefficient of $P_\mu$ on
atom $I$ is

$$
D^{\mu}_{snm}(\mathbf{k},\mathbf{q}) =
\sum_{J\mathbf{R}ij} C^{\mu}_{Ii,Jj}(\mathbf{R})
\left[
 c^*_{sn\mathbf{k},Ii}\, c_{sm,\mathbf{k}+\mathbf{q},Jj}\,
 e^{i(\mathbf{k}+\mathbf{q})\cdot\mathbf{R}}
 + c^*_{sn\mathbf{k},Jj}\, c_{sm,\mathbf{k}+\mathbf{q},Ii}\,
 e^{-i\mathbf{k}\cdot\mathbf{R}}
\right].
$$

The two terms come from the two orderings of the orbital pair, and both are
necessary when $I=J$. The wave vectors here are in physical units. LibRPA stores
them in units of $2\pi$, so the code adds this factor to the phases.

## Response

The response is

$$
\chi_{0,\mu\nu}(\mathbf{q},i\omega) =
\sum_{s\mathbf{k}nm}
\frac{w_{sn\mathbf{k}}-w_{sm,\mathbf{k}+\mathbf{q}}}
{\epsilon_{sn\mathbf{k}}-\epsilon_{sm,\mathbf{k}+\mathbf{q}}+i\omega}
D^{\mu}_{snm}(\mathbf{k},\mathbf{q})
D^{\nu*}_{snm}(\mathbf{k},\mathbf{q}).
$$

The stored weights $w$ include the spin multiplicity and $1/N_k$. Energies and
frequencies are in Hartree.

Time reversal maps the transition $(n\mathbf{k}\to m,\mathbf{k}+\mathbf{q})$ to
$(m,-\mathbf{k}-\mathbf{q}\to n,-\mathbf{k})$. The two transitions have the same
$D$ and complex-conjugate energy factors, so the imaginary part of the factor
cancels in the sum, and $\chi_0$ is Hermitian. The code keeps the real part of
the factor. It also accumulates the imaginary part at the first frequency, and
stops if that part exceeds $10^{-9}$ of the largest response element. The energy
routines build the lower atom triangle from this Hermitian property.

## Cost

Pairs with $w_{sn\mathbf{k}}=w_{sm,\mathbf{k}+\mathbf{q}}$ make no contribution
and are skipped. For an insulator, the cost of the response then scales as
$N_q N_k N_s N_{\rm occ} N_{\rm unocc} N_{\rm aux}^2 N_\omega$.

The Fourier transform of the RI coefficients to $\mathbf{k}+\mathbf{q}$ and to
$-\mathbf{k}$ adds a term that does not depend on the bands. It is one pass over
the local RI coefficients for each pair of k and q points, and one for each k
point. The code makes a transform only when a transition needs it.

The code processes one k point, one q point and one initial band at a time, and
keeps only the transition densities of the current band and the two transformed
RI coefficients. Each rank holds the response of its atom blocks at all of its q
points and frequencies, as in the space-time method, and the imaginary part at
the first frequency. Each rank collects the RI coefficients of the atoms in its
blocks with MPI. A block held on several ranks is used once.

With the `atompair` and `libri` routings, MPI ranks share the upper-triangular
auxiliary atom blocks, and each rank loops over all q points, k points and bands.
The MPI speedup then cannot exceed the number of atom pairs. With the `rtau`
routing, ranks share the q points, and each rank computes all atom blocks of its
q points. The MPI speedup then cannot exceed the number of q points.
