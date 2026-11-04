# Real-space screened interaction from an irreducible q grid

The GW correlation self-energy contracts the screened interaction in real space,
so LibRPA transforms $W^{\mathrm{c}}$ from reciprocal space to the Born-von
Karman lattice before the contraction. Symmetry lets the screening be evaluated
on the irreducible q points only. This note gives the equation the restore
implements and states why the reduced and the full Brillouin-zone k-point lists
must reach the same result.

## The transform

For lattice vectors $R$ and reciprocal q points,

\[
W^{\mathrm{c}}(R)=\frac{1}{N_k}\sum_{q}W^{\mathrm{c}}(q)\,e^{-i q\cdot R},
\]

with $N_k$ the number of Born-von Karman cells. When screening is evaluated only
at the $N_q^{\mathrm{IBZ}}$ irreducible points $q^{\mathrm{rep}}$, the sum over
the full mesh is recovered by grouping the remaining q points into the stars
$S(q^{\mathrm{rep}})$,

\[
W^{\mathrm{c}}(R)=\sum_{q^{\mathrm{rep}}}
\frac{|S(q^{\mathrm{rep}})|}{N_k}
\sum_{q\in S(q^{\mathrm{rep}})}\frac{1}{|S(q^{\mathrm{rep}})|}
W^{\mathrm{c}}(q)\,e^{-i q\cdot R},
\]

where $W^{\mathrm{c}}(q)$ for $q\ne q^{\mathrm{rep}}$ is obtained from the
representative by the space-group rotation that maps $q^{\mathrm{rep}}$ onto
$q$. The inner sum visits every member of the star, so the star weight
$|S(q^{\mathrm{rep}})|/N_k$ emerges from adding one $1/N_k$ term per member; the
outer sum then runs over representatives alone and reproduces the full mesh sum.
`SymmetryQPointView::weights` stores the same factor $|S|/N_k$.

The restore therefore needs one representative per k-star, and it must visit
each star once. Visiting a star once per member would apply the star weight
$|S|/N_k$ a further $|S|$ times.

## The two accepted k-point lists

A reader may hand over either form:

- a reduced list holding one k point per star, which is what the ABACUS reader
  exports;
- a full list holding every member k point of the Born-von Karman cell, which
  is what the FHI-aims reader exports.

Both forms describe the same star decomposition. The reduced list already has
one entry per star, so its grid mapping can be used as it stands. The full list
repeats each star once per member, so one recorded representative must be
selected per star before the sum above is formed. Neither form changes the
equation; they differ only in which entries of the mapping table are read.

Detecting the form by comparing the number of k-stars with the length of the k
list is only valid for the reduced form and silently rejects the full form.
The restore instead asks the shared representative builder, which returns one
index per k-star for a full list and an empty list for a reduced one. A full
list whose representatives cannot be resolved is reported rather than treated
as reduced, because the reduced layout would repeat the stars and inflate
$W^{\mathrm{c}}(R)$ without a diagnostic.

## Reference

The transform is the discrete Fourier sum of the screened interaction used by
the periodic GW space-time method. The auxiliary-basis (LRI) representation of
the screened interaction that this code follows is the one described in the
LibRI method literature; consult the manuscript you cite for that method rather
than this note, which records the convention the code implements.
