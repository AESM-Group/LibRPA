#include "utils_matrix_mpi.h"

#include <cassert>
#include <complex>
#include <stdexcept>

#include "../utils/base_utility.h"
#include "complexmatrix.h"
#include "scalapack_connector.h"

namespace librpa_int
{

void allreduce_matrix(const matrix &mat_send, matrix &mat_recv, MPI_Comm mpi_comm)
{
    assert(mat_send.nr==mat_recv.nr);
    assert(mat_send.nc==mat_recv.nc);
    int mat_size=mat_recv.nr*mat_recv.nc;
    MPI_Allreduce(mat_send.c, mat_recv.c, mat_size, MPI_DOUBLE, MPI_SUM, mpi_comm);
}

void reduce_matrix(const matrix &mat_send, matrix &mat_recv, int root, MPI_Comm mpi_comm)
{
    assert(mat_send.nr==mat_recv.nr);
    assert(mat_send.nc==mat_recv.nc);
    int mat_size = mat_recv.nr * mat_recv.nc;
    MPI_Reduce(mat_send.c, mat_recv.c, mat_size, MPI_DOUBLE, MPI_SUM, root, mpi_comm);
}

void allreduce_ComplexMatrix(const ComplexMatrix &cmat_send, ComplexMatrix &cmat_recv, MPI_Comm mpi_comm)
{
    assert(cmat_send.nr==cmat_recv.nr);
    assert(cmat_send.nc==cmat_recv.nc);
    MPI_Allreduce(cmat_send.c, cmat_recv.c, cmat_recv.size, MPI_DOUBLE_COMPLEX, MPI_SUM, mpi_comm);
}

void reduce_ComplexMatrix(const ComplexMatrix &cmat_send, ComplexMatrix &cmat_recv, int root, MPI_Comm mpi_comm)
{
    assert(cmat_send.nr==cmat_recv.nr);
    assert(cmat_send.nc==cmat_recv.nc);
    MPI_Reduce(cmat_send.c, cmat_recv.c, cmat_recv.size, MPI_DOUBLE_COMPLEX, MPI_SUM, root, mpi_comm);
}

void broadcast_matrix(matrix &mat, int root, MPI_Comm mpi_comm)
{
    int dims[2];
    dims[0] = mat.nr;
    dims[1] = mat.nc;
    MPI_Bcast(dims, 2, MPI_INT, root, mpi_comm);

    int myrank;
    MPI_Comm_rank(mpi_comm, &myrank);
    if (myrank != root)
    {
        mat.create(dims[0], dims[1]);
    }

    size_t count = as_size(dims[0]) * as_size(dims[1]);
    MPI_Bcast(mat.c, count, mpi_datatype<double>::value, root, mpi_comm);
}

void broadcast_ComplexMatrix(ComplexMatrix &cmat, int root, MPI_Comm mpi_comm)
{
    int dims[2];
    dims[0] = cmat.nr;
    dims[1] = cmat.nc;
    MPI_Bcast(dims, 2, MPI_INT, root, mpi_comm);

    int myrank;
    MPI_Comm_rank(mpi_comm, &myrank);
    if (myrank != root)
    {
        cmat.create(dims[0], dims[1]);
    }

    size_t count = as_size(dims[0]) * as_size(dims[1]);
    MPI_Bcast(cmat.c, count, mpi_datatype<std::complex<double>>::value, root, mpi_comm);
}

Matz collect_blacs_matrix_root(const Matz &local, const ArrayDesc &distributed_descriptor)
{
    if (!distributed_descriptor.is_initialized())
    {
        throw std::invalid_argument("Distributed matrix descriptor is not initialized");
    }
    if (distributed_descriptor.m() <= 0 || distributed_descriptor.n() <= 0)
    {
        throw std::invalid_argument("Distributed matrix must have positive global dimensions");
    }
    if (local.major() != MAJOR::COL)
    {
        throw std::invalid_argument("Distributed matrix must use column-major local storage");
    }
    if (local.nr() != distributed_descriptor.m_loc() ||
        local.nc() != distributed_descriptor.n_loc())
    {
        throw std::invalid_argument("Local matrix shape does not match its BLACS descriptor");
    }

    ArrayDesc root_descriptor(distributed_descriptor.ictxt());
    root_descriptor.init(distributed_descriptor.m(), distributed_descriptor.n(),
                         distributed_descriptor.m(), distributed_descriptor.n(),
                         distributed_descriptor.irsrc(), distributed_descriptor.icsrc());

    Matz source_dummy(1, 1, MAJOR::COL);
    const cplxdb *source = local.nr() > 0 && local.nc() > 0 ? local.ptr() : source_dummy.ptr();
    Matz transfer_buffer = root_descriptor.is_src() ? Matz(distributed_descriptor.m(),
                                                           distributed_descriptor.n(), MAJOR::COL)
                                                    : Matz(1, 1, MAJOR::COL);
    ScalapackConnector::pgemr2d_f(distributed_descriptor.m(), distributed_descriptor.n(), source, 1,
                                  1, distributed_descriptor.desc, transfer_buffer.ptr(), 1, 1,
                                  root_descriptor.desc, distributed_descriptor.ictxt());

    if (root_descriptor.is_src())
    {
        return transfer_buffer;
    }
    return Matz(0, 0, MAJOR::COL);
}

} /* end of namespace librpa_int */

