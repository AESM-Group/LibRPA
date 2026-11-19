# Dielectric factorization for W

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
