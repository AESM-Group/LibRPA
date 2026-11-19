# Dielectric factorization for W

On the imaginary axis the symmetrized matrix is
`epsilon = I - sqrt(V) chi0 sqrt(V)`. For a Hermitian negative-semidefinite
response and positive-semidefinite Coulomb matrix it is positive definite,
including filtered Coulomb channels where it becomes identity. Cholesky is
therefore always attempted first. W requires Hermitian inputs; a general
non-Hermitian response is outside this contract. POTRF reads one triangle
and cannot diagnose a non-Hermitian input.

The former Cholesky runtime option has been removed from the C API, Fortran
bindings and driver. The options layout now contains 84 fields; applications
and bindings compiled against the previous layout must be rebuilt together.

`solve_dielectric_system` factors first and checks INFO on the owning BLACS
communicator. Successful POTRF is followed by POTRS. At Gamma, the physical
head/head-wing overwrite can change definiteness, so positive POTRF INFO
permits restoration and GETRF/GETRS retry. Negative INFO is an argument error
and never triggers retry. A failed LU is reported; the solver never silently
switches to block-pivot or no-pivot LU. RPA correlation retains its separate
bpiv calculation policy.

The RHS is untouched until a factorization succeeds. Solves apply the inverse
to the needed RHS (sqrt(Vc) or identity), rather than computing an extra inverse
before multiplying that RHS. CPU LAPACK, distributed ScaLAPACK, and CPU/GPU
DDLA use the caller-selected descriptor and backend-local pivots.

POTRF is destructive. The raw head/wing rewrite reuses the now-dead G scratch
as its original-matrix backup, so it introduces no new full device allocation.
The head-only path owns one backup per Gamma q in the selected backend and
reuses it across frequencies. The strict-2D CPU body retains the original body
and uses factor/RHS temporaries for the identity solve; this adds quadratic
local scratch to that Gamma branch. No ordinary-q backup is allocated.

`test_wc_cholesky` uses independent known-X complex systems and a full raw
head/wing indefinite-body fixture. It covers successful Cholesky, restoration
before LU retry, non-Gamma rejection, singular LU rejection, pivot swaps,
and empty-owner layouts. The one-rank connector tests cover split solves,
leading dimensions and native submatrix offsets.
