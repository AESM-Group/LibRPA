#ifdef NDEBUG
#undef NDEBUG
#endif

#include <mpi.h>

#include <cassert>
#include <complex>
#include <iostream>

#include "../core/qsgw/distributed_matrix.h"
#include "../io/global_io.h"
#include "../math/lapack_connector.h"
#include "../math/utils_matrix_m_mpi.h"
#include "../math/utils_matrix_mpi.h"
#include "../mpi/global_mpi.h"
#include "../mpi/utils_blacs.h"

using librpa_int::ArrayDesc;
using librpa_int::BlacsCtxtHandler;
using librpa_int::collect_blacs_matrix_root;
using librpa_int::cplxdb;
using librpa_int::init_local_mat;
using librpa_int::MAJOR;
using librpa_int::Matz;
using librpa_int::ScalapackConnector;
using librpa_int::qsgw::broadcast_spin_k_matrix_map;
using librpa_int::qsgw::SpinKMatrixMap;

namespace
{

void assert_close(const cplxdb actual, const cplxdb expected)
{
    assert(std::abs(actual - expected) < 1.0e-13);
}

Matz make_global_matrix(const int rows, const int columns)
{
    Matz result(rows, columns, MAJOR::COL);
    for (int row = 0; row < rows; ++row)
    {
        for (int column = 0; column < columns; ++column)
        {
            result(row, column) = cplxdb(10.0 * row + column + 0.25, row - 2.0 * column + 0.5);
        }
    }
    return result;
}

void test_distributed_complex_matrix_is_collected_exactly_once_on_root(BlacsCtxtHandler& blacs)
{
    blacs.set_square_grid();
    assert(blacs.nprocs == 4);
    assert(blacs.nprows == 2);
    assert(blacs.npcols == 2);

    constexpr int rows = 5;
    constexpr int columns = 4;
    ArrayDesc distributed_desc(blacs);
    distributed_desc.init(rows, columns, 2, 2, 0, 0);
    ArrayDesc root_desc(blacs);
    root_desc.init(rows, columns, rows, columns, 0, 0);

    Matz global(1, 1, MAJOR::COL);
    if (root_desc.is_src())
    {
        global = make_global_matrix(rows, columns);
    }
    auto local = init_local_mat<cplxdb>(distributed_desc, MAJOR::COL);
    ScalapackConnector::pgemr2d_f(rows, columns, global.ptr(), 1, 1, root_desc.desc, local.ptr(), 1,
                                  1, distributed_desc.desc, distributed_desc.ictxt());

    const auto collected = collect_blacs_matrix_root(local, distributed_desc);
    if (root_desc.is_src())
    {
        assert(collected.nr() == rows);
        assert(collected.nc() == columns);
        const auto expected = make_global_matrix(rows, columns);
        for (int row = 0; row < rows; ++row)
        {
            for (int column = 0; column < columns; ++column)
            {
                assert_close(collected(row, column), expected(row, column));
            }
        }
    }
    else
    {
        assert(collected.nr() == 0);
        assert(collected.nc() == 0);
    }
    distributed_desc.barrier();
}

void test_spin_k_matrix_map_is_broadcast_with_layout(const librpa_int::MpiCommHandler& communicator)
{
    SpinKMatrixMap values;
    if (communicator.myid == 0)
    {
        Matz first(2, 2, MAJOR::ROW);
        first(0, 0) = {1.0, 0.0};
        first(0, 1) = {2.0, 3.0};
        first(1, 0) = {4.0, 5.0};
        first(1, 1) = {6.0, 0.0};
        values[0][1] = std::move(first);

        Matz second(1, 3, MAJOR::COL);
        second(0, 0) = {-1.0, 0.5};
        second(0, 1) = {-2.0, 1.5};
        second(0, 2) = {-3.0, 2.5};
        values[2][4] = std::move(second);
    }
    else
    {
        Matz poison(1, 1, MAJOR::ROW);
        poison(0, 0) = 99.0;
        values[9][9] = std::move(poison);
    }

    broadcast_spin_k_matrix_map(values, 0, communicator);
    assert(values.size() == 2);
    assert(values.count(9) == 0);
    const auto& first = values.at(0).at(1);
    assert(first.nr() == 2 && first.nc() == 2);
    assert(first.major() == MAJOR::ROW);
    assert_close(first(0, 1), {2.0, 3.0});
    assert_close(first(1, 0), {4.0, 5.0});
    const auto& second = values.at(2).at(4);
    assert(second.nr() == 1 && second.nc() == 3);
    assert(second.major() == MAJOR::COL);
    assert_close(second(0, 2), {-3.0, 2.5});
}

}  // namespace

int main(int argc, char** argv)
{
    int provided = 0;
    // The matrix collectives run on the initializing thread.
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    assert(provided >= MPI_THREAD_FUNNELED);
    librpa_int::global::init_global_mpi(MPI_COMM_WORLD);
    librpa_int::global::init_global_io();

    BlacsCtxtHandler blacs(librpa_int::global::mpi_comm_global);
    blacs.init();
    test_distributed_complex_matrix_is_collected_exactly_once_on_root(blacs);
    test_spin_k_matrix_map_is_broadcast_with_layout(librpa_int::global::mpi_comm_global_h);
    blacs.exit();

    const int rank = librpa_int::global::myid_global;
    librpa_int::global::finalize_global_io();
    librpa_int::global::finalize_global_mpi();
    MPI_Finalize();
    if (rank == 0)
    {
        std::cout << "test_qsgw_distributed_matrix: all tests passed\n";
    }
    return 0;
}
