#include "../io/stl_io_helper.h"
#include "../math/utils_matrix_m.h"
#include "../math/utils_matrix_m_mpi.h"
#include "../mpi/global_mpi.h"
#include "../mpi/utils_blacs.h"
#include "../src/io/global_io.h"
#include "mpi_test_config.h"
#include "testutils.h"

using namespace std;
using namespace librpa_int;
using namespace librpa_int::global;

static BlacsCtxtHandler blacs_ctxt_h;

template <typename T>
void test_pgemm(const T &m1_lb, const T &m1_ub)
{
    typedef T type;
    blacs_ctxt_h.set_square_grid();
    assert(blacs_ctxt_h.nprocs == 4);
    assert(blacs_ctxt_h.nprows == 2);
    assert(blacs_ctxt_h.npcols == 2);

    const int m = 2, n = 6, k = 4;
    const int mb = 1, nb = 3, kb = 2;
    // size 1 matrices to avoid nullptr in pgemr2d
    matrix_m<T> m1(1, 1, MAJOR::COL), m2(1, 1, MAJOR::COL), prod(1, 1, MAJOR::COL), prod_lapack(1, 1, MAJOR::COL);
    int pid_src, irsrc, icsrc;
    pid_src = 2;
    blacs_ctxt_h.get_pcoord(pid_src, irsrc, icsrc);
    // printf("pid_src irsrc icsrc: %d %d %d\n", pid_src, irsrc, icsrc);

    // prepare the real data in the source process
    if (blacs_ctxt_h.myid == pid_src)
    {
        m1.resize(m, k);
        m2.resize(k, n);
        prod.resize(m, n);
        m1.randomize(m1_lb, m1_ub);
        m2.randomize();
        // printf("Mat1:\n%s", str(m1).c_str());
        // printf("Mat2:\n%s", str(m2).c_str());
        prod_lapack = m1 * m2;
        // printf("Product:\n%s", str(prod_lapack).c_str());
    }
    blacs_ctxt_h.barrier();

    auto pair_desc_m1 = prepare_array_desc_mr2d_src_and_all(blacs_ctxt_h, m, k, mb, kb, irsrc, icsrc);
    auto pair_desc_m2 = prepare_array_desc_mr2d_src_and_all(blacs_ctxt_h, k, n, kb, nb, irsrc, icsrc);
    matrix_m<T> m1_local = init_local_mat<T>(pair_desc_m1.second, MAJOR::COL);
    matrix_m<T> m2_local = init_local_mat<T>(pair_desc_m2.second, MAJOR::COL);
    // printf("mat1_local addr %p\n%s", m1_local.ptr(), str(m1_local).c_str());
    // printf("mat2_local addr %p\n%s", m2_local.ptr(), str(m2_local).c_str());
    // distribute data
    ScalapackConnector::pgemr2d_f(m, k,
                                  m1.ptr(), 1, 1, pair_desc_m1.first.desc,
                                  m1_local.ptr(), 1, 1, pair_desc_m1.second.desc,
                                  blacs_ctxt_h.ictxt);
    ScalapackConnector::pgemr2d_f(k, n,
                                  m2.ptr(), 1, 1, pair_desc_m2.first.desc,
                                  m2_local.ptr(), 1, 1, pair_desc_m2.second.desc,
                                  blacs_ctxt_h.ictxt);

    // initialize product matrix
    auto pair_desc_prod = prepare_array_desc_mr2d_src_and_all(blacs_ctxt_h, m, n, mb, nb, irsrc, icsrc);
    auto prod_local = init_local_mat<type>(pair_desc_prod.second, MAJOR::COL);

    // carry out distributed multiplication
    ScalapackConnector::pgemm_f('N', 'N', m, n, k, 1.0,
                                m1_local.ptr(), 1, 1, pair_desc_m1.second.desc,
                                m2_local.ptr(), 1, 1, pair_desc_m2.second.desc,
                                0.0,
                                prod_local.ptr(), 1, 1, pair_desc_prod.second.desc);
    // printf("prod_local on proc %d:\n%s", blacs_ctxt_h.myid, str(prod_local).c_str());
    // collect data back to source
    ScalapackConnector::pgemr2d_f(m, n,
                                  prod_local.ptr(), 1, 1, pair_desc_prod.second.desc,
                                  prod.ptr(), 1, 1, pair_desc_prod.first.desc,
                                  blacs_ctxt_h.ictxt);

    if (blacs_ctxt_h.myid == pid_src)
    {
        T thres = 1e-14;
        // small threshold when float is used
        if (std::is_same<typename to_real<type>::type, float>::value)
            thres = 1e-6;
        assert(fequal_array(m*n, prod_lapack.ptr(), prod.ptr(), false, thres));
    }

    blacs_ctxt_h.exit();
}

template <typename T>
void test_invert_scalapack()
{
    typedef T type;
    typedef typename to_real<type>::type real_type;
    // a looser threshold than gemm
    const T thres = std::is_same<real_type, float>::value ? 1e-4 : 1e-12;

    blacs_ctxt_h.set_square_grid();
    assert(blacs_ctxt_h.nprocs == 4);
    assert(blacs_ctxt_h.nprows == 2);
    assert(blacs_ctxt_h.npcols == 2);

    const int n = 8;
    matrix_m<T> mat(1, 1, MAJOR::COL);

    int pid_src, irsrc, icsrc;
    pid_src = 2;
    blacs_ctxt_h.get_pcoord(pid_src, irsrc, icsrc);
    // Initialize a well-conditioned matrix at the source process.
    // The random elements are in [-1, 1], so adding n to the diagonal makes
    // the matrix strictly diagonally dominant and keeps the error of
    // inv(A) A well below thres. Without the shift, about 0.1% of the
    // random matrices are ill-conditioned enough to fail the check.
    if (blacs_ctxt_h.myid == pid_src)
    {
        mat = random<T>(n, n, -1.0, 1.0, MAJOR::COL);
        for (int i = 0; i < n; i++)
        {
            mat(i, i) += T(n);
        }
    }

    ArrayDesc desc_mat(blacs_ctxt_h), desc_mat_fb_src(blacs_ctxt_h);
    desc_mat.init_1b1p(n, n, irsrc, icsrc);
    desc_mat_fb_src.init(n, n, n, n, irsrc, icsrc);
    // create local matrix and distribute the source to the process grid
    matrix_m<T> mat_loc = init_local_mat<T>(desc_mat, MAJOR::COL);
    ScalapackConnector::pgemr2d_f(n, n,
                                  mat.ptr(), 1, 1, desc_mat_fb_src.desc,
                                  mat_loc.ptr(), 1, 1, desc_mat.desc,
                                  blacs_ctxt_h.ictxt);
    auto mat_loc_orig = mat_loc.copy();
    invert_scalapack(mat_loc, desc_mat);
    auto mat_times_invmat = multiply_scalapack(mat_loc, desc_mat, mat_loc_orig, desc_mat, desc_mat);
    ScalapackConnector::pgemr2d_f(n, n,
                                  mat_times_invmat.ptr(), 1, 1, desc_mat.desc,
                                  mat.ptr(), 1, 1, desc_mat_fb_src.desc,
                                  blacs_ctxt_h.ictxt);
    // printf("mat * invmat on proc %d\n%s", blacs_ctxt_h.myid, str(mat_times_invmat).c_str());
    if (blacs_ctxt_h.myid == pid_src)
    {
        matrix_m<T> identity(n, n, MAJOR::COL);
        identity.zero_out();
        identity.set_diag(1.0);
        assert(fequal_array(n*n, mat.ptr(), identity.ptr(), false, thres));
    }

    blacs_ctxt_h.exit();
}

template <typename T>
void test_power_hemat_blacs_square_grid(const T &m_lb, const T &m_ub, bool force_real = false)
{
    typedef T type;
    typedef typename to_real<type>::type real_type;
    const T thres =
        std::is_same<typename to_real<type>::type, float>::value ? 1e-5 : 1e-14;
    blacs_ctxt_h.set_square_grid();
    assert(blacs_ctxt_h.nprocs == 4);
    assert(blacs_ctxt_h.nprows == 2);
    assert(blacs_ctxt_h.npcols == 2);

    const int n = 6, nb = n / 2;
    matrix_m<T> mat(1, 1, MAJOR::COL), mat_gather(1, 1, MAJOR::COL);
    int pid_src, irsrc, icsrc;
    pid_src = 2;
    blacs_ctxt_h.get_pcoord(pid_src, irsrc, icsrc);
    if (blacs_ctxt_h.myid == pid_src)
    {
        vector<real_type> evs {0.1, 0.2, 0.5, 1.0, 2.0, 4.0};
        if (force_real && mat.is_complex)
            mat = random_sy_selected_ev(n, evs, MAJOR::COL).to_complex();
        else
            mat = random_he_selected_ev(n, evs, MAJOR::COL);
        mat_gather = mat;
    }
    auto pair_desc_m = prepare_array_desc_mr2d_src_and_all(blacs_ctxt_h, n, n, nb, nb, irsrc, icsrc);
    matrix_m<T> mat_loc = init_local_mat<T>(pair_desc_m.second, MAJOR::COL);
    matrix_m<T> eig_loc = init_local_mat<T>(pair_desc_m.second, MAJOR::COL);
    // printf("eig_loc\n%s", str(eig_loc).c_str());
    ScalapackConnector::pgemr2d_f(n, n,
                                  mat.ptr(), 1, 1, pair_desc_m.first.desc,
                                  mat_loc.ptr(), 1, 1, pair_desc_m.second.desc,
                                  blacs_ctxt_h.ictxt);
    // printf("mat_loc of PID %d before\n%s", blacs_ctxt_h.myid, str(mat_loc).c_str());
    auto mat_loc_back = mat_loc;

    real_type *W = new real_type [n];

    size_t n_filtered;

    // Take cubic root
    if (force_real)
        power_hemat_blacs_real<real_type>(mat_loc, pair_desc_m.second, eig_loc, pair_desc_m.second,
                                          n_filtered, W, 1.0 / 3.0, -1.0e5);
    else
        power_hemat_blacs<real_type>(mat_loc, pair_desc_m.second,
                                    eig_loc, pair_desc_m.second, n_filtered, W, 1.0/3.0, -1.0e5);
    assert(n_filtered == 0);
    // if (blacs_ctxt_h.myid == pid_src)
    // {
    //     printf("Eigenvalues: ");
    //     for (int i = 0; i < n; i++)
    //         printf("%f ", W[i]);
    //     printf("\n");
    // }
    blacs_ctxt_h.barrier();
    // printf("mat_loc of PID %d middle\n%s", blacs_ctxt_h.myid, str(mat_loc).c_str());
    // Take cubic
    if (force_real)
        power_hemat_blacs_real<real_type>(mat_loc, pair_desc_m.second, eig_loc, pair_desc_m.second,
                                          n_filtered, W, 3.0, -1.0e5);
    else
        power_hemat_blacs<real_type>(mat_loc, pair_desc_m.second,
                                    eig_loc, pair_desc_m.second, n_filtered, W, 3.0, -1.0e5);
    assert(n_filtered == 0);
    // printf("eig_loc\n%s", str(eig_loc).c_str());
    // printf("mat_loc of PID %d after\n%s", blacs_ctxt_h.myid, str(mat_loc).c_str());

    ScalapackConnector::pgemr2d_f(n, n,
                                  mat_loc.ptr(), 1, 1, pair_desc_m.second.desc,
                                  mat_gather.ptr(), 1, 1, pair_desc_m.first.desc,
                                  blacs_ctxt_h.ictxt);

    if (blacs_ctxt_h.myid == pid_src)
    {
        bool print = true;
        if (print)
        {
            if (force_real)
                printf("use power_hemat_blacs_real\n");
            else
                printf("use power_hemat_blacs\n");
            printf("mat global at pid_src %d\n%s", pid_src, str(mat).c_str());
            printf("mat gathered at pid_src %d\n%s", pid_src, str(mat_gather).c_str());
        }
        assert(fequal_array(n*n, mat.ptr(), mat_gather.ptr(), print, thres));
    }

    delete [] W;

    blacs_ctxt_h.exit();
}

template<typename T>
void test_collect_block_from_IJ_storage()
{
    // typedef T type;
    // typedef typename to_real<type>::type real_type;
    blacs_ctxt_h.set_square_grid();
    // assert(blacs_ctxt_h.nprocs == 4);
    // assert(blacs_ctxt_h.nprows == 2);
    // assert(blacs_ctxt_h.npcols == 2);
    map<int, map<int, matrix_m<T>>> IJmap;

    AtomicBasis ab(vector<size_t>{1, 2, 2, 1});

    for (int i = 0; i < ab.n_atoms; i++)
    {
        IJmap[i][i].resize(ab.get_atom_nb(i), ab.get_atom_nb(i), MAJOR::COL).randomize(0, 1, !is_complex<T>(), is_complex<T>());
        for (int j = i + 1; j < ab.n_atoms; j++)
        {
            IJmap[i][j].resize(ab.get_atom_nb(i), ab.get_atom_nb(j), MAJOR::COL).randomize(-1, 0, false, false);
            IJmap[j][i] = transpose(IJmap[i][j], is_complex<T>());
        }
    }

    if (blacs_ctxt_h.myid == 0)
    {
        for (int I = 0; I < ab.n_atoms; I++)
            for (int J = 0; J < ab.n_atoms; J++)
                cout << I << " " << J << endl << str(IJmap[I][J]);
    }
    blacs_ctxt_h.barrier();
    T alpha = 1.0;

    ArrayDesc desc_fb(blacs_ctxt_h);
    desc_fb.init(ab.nb_total, ab.nb_total, ab.nb_total, ab.nb_total, 0, 0);
    auto mat_loc = init_local_mat<T>(desc_fb, IJmap[0][0].major());

    for (int I = 0; I < ab.n_atoms; I++)
        for (int J = 0; J < ab.n_atoms; J++)
            collect_block_from_IJ_storage(mat_loc, desc_fb, ab, ab, I, J, alpha, IJmap[I][J].ptr(), MAJOR::COL);

    // if (blacs_ctxt_h.myid == 0)
    // {
    //     cout << "Use collect_block_from_IJ_storage" << endl << str(mat_loc);
    // }
    blacs_ctxt_h.barrier();

    auto mat_loc_syhe = init_local_mat<T>(desc_fb, IJmap[0][0].major());
    for (int I = 0; I < ab.n_atoms; I++)
        for (int J = I; J < ab.n_atoms; J++)
        {
            // printf("myid %d I %d J %d\n", blacs_ctxt_h.myid, I, J);
            collect_block_from_IJ_storage_syhe(mat_loc_syhe, desc_fb, ab, I, J, is_complex<T>(), alpha, IJmap[I][J].ptr(), MAJOR::COL);
        }
    // if (blacs_ctxt_h.myid == 0)
    // {
    //     cout << "Use collect_block_from_IJ_storage_syhe" << endl << str(mat_loc_syhe);
    // }

    map<int, map<int, matrix_m<T>>> IJmap_collect;
    if (blacs_ctxt_h.myid == 0)
    {
        for (int I = 0; I < ab.n_atoms; I++)
            for (int J = 0; J < ab.n_atoms; J++)
                cout << I << " " << J << endl << str(IJmap_collect[I][J]);
    }
    // cout << "mapping the 2D block to IJ" << endl;
    map_block_to_IJ_storage(IJmap_collect, ab, ab, mat_loc_syhe, desc_fb, MAJOR::COL);
    // cout << "done map" << endl;
    if (blacs_ctxt_h.myid == 0)
    {
        for (int I = 0; I < ab.n_atoms; I++)
            for (int J = 0; J < ab.n_atoms; J++)
                cout << I << " " << J << endl << str(IJmap_collect[I][J]);
    }

    blacs_ctxt_h.exit();
}

template <typename T>
void test_collect_block_from_ALL_IJ_Tensor_sparse_zero_missing()
{
    blacs_ctxt_h.set_square_grid();

    AtomicBasis ab(std::vector<size_t>{1, 1});
    std::array<double, 3> cell{0.0, 0.0, 0.0};
    std::map<int, std::map<std::pair<int, std::array<double, 3>>, RI::Tensor<T>>> tmap;

    auto data = std::make_shared<std::valarray<T>>(T(7), 1);
    tmap[0][{0, cell}] = RI::Tensor<T>({1, 1}, data);

    ArrayDesc desc_fb(blacs_ctxt_h);
    desc_fb.init(ab.nb_total, ab.nb_total, ab.nb_total, ab.nb_total, 0, 0);
    auto mat_loc = init_local_mat<T>(desc_fb, MAJOR::ROW);

    collect_block_from_ALL_IJ_Tensor_sparse_zero_missing(
        mat_loc, desc_fb, ab, cell, true, T(1), tmap, MAJOR::ROW);

    if (blacs_ctxt_h.myid == 0)
    {
        assert(mat_loc(0, 0) == T(7));
        assert(mat_loc(0, 1) == T(0));
        assert(mat_loc(1, 0) == T(0));
        assert(mat_loc(1, 1) == T(0));
    }

    blacs_ctxt_h.exit();
}

template <typename T>
void test_local_mat_from_ap_dist()
{
    blacs_ctxt_h.set_square_grid(true, librpa_int::CTXT_LAYOUT::R);
    assert(blacs_ctxt_h.nprocs == 4);
    assert(blacs_ctxt_h.nprows == 2);
    assert(blacs_ctxt_h.npcols == 2);

    const size_t m = 4;
    const size_t n = m;

    ArrayDesc ad(blacs_ctxt_h);
    ad.init_1b1p(m, n, 0, 0);
    assert(ad.initialized());
    assert(ad.mb() == 2);
    assert(ad.nb() == 2);

    librpa_int::AtomicBasis ab;
    // 3 atoms, atom 0 and 2 with 1 and atom 1 with 2
    // Process indices
    // | 0   0 | 0   1 |
    // | 0   3 | 3   1 |
    // |-------|-------|
    // | 0   3 | 3   1 |
    // | 2   2 | 2   3 |
    ab.set(std::vector<size_t>{1, 2, 1});
    matrix_m<T> global(ab.nb_total, ab.nb_total, MAJOR::COL);
    if (myid_global == 0)
    {
        if (is_complex<T>())
        {
             global.randomize(0, 1, false, true);
        }
        else
        {
             global.randomize(0, 1, true, false);
        }
    }
    // broadcast
    MPI_Bcast(global.ptr(), m * n, mpi_datatype<T>::value, 0, ad.comm());
    auto mat_loc_ref = get_local_mat<T>(global, ad);

    const std::unordered_map<int, std::vector<atpair_t>> map_proc_IJs_avail{
        {0, {{0, 0}, {1, 0}, {0, 1}}},
        {1, {{0, 2}, {1, 2}}},
        {2, {{2, 0}, {2, 1}}},
        {3, {{1, 1}, {2, 2}}}};
    unordered_map<atpair_t, matrix_m<T>, atpair_hash> IJmap;
    for (const auto &IJ: map_proc_IJs_avail.at(myid_global))
    {
        const auto &I = IJ.first;
        const auto &J = IJ.second;
        auto mat = matrix_m<T>(ab.get_atom_nb(I), ab.get_atom_nb(J), MAJOR::COL);
        for (int i = 0; i < mat.nr(); i++)
        {
            for (int j = 0; j < mat.nc(); j++)
            {
                mat(i, j) = global(ab.get_part_range()[I] + i, ab.get_part_range()[J]+ j);
            }
        }
        IJmap[IJ] = std::move(mat);
    }
    const auto mat_loc = get_local_mat_from_ap_dist<T>(IJmap, map_proc_IJs_avail, ab, ab, ad, MAJOR::COL);
    // for (int i = 0; i < 4; i++)
    // {
    //     blacs_ctxt_h.barrier();
    //     if (myid_global == i)
    //     {
    //         std::cout << "myid " << i << " local matrix reference" << std::endl;
    //         std::cout << mat_loc_ref;
    //         std::cout << "myid " << i << " local matrix" << std::endl;
    //         std::cout << mat_loc << std::endl;
    //     }
    // }
    assert(fequal_array(mat_loc.size(), mat_loc.ptr(), mat_loc_ref.ptr(), false));

    blacs_ctxt_h.exit();
}

void test_local_mat_from_ap_dist_he()
{
    blacs_ctxt_h.set_square_grid(true, librpa_int::CTXT_LAYOUT::R);
    assert(blacs_ctxt_h.nprocs == 4);
    assert(blacs_ctxt_h.nprows == 2);
    assert(blacs_ctxt_h.npcols == 2);

    const size_t m = 4;
    const size_t n = m;

    ArrayDesc ad(blacs_ctxt_h);
    ad.init_1b1p(m, n, 0, 0);
    assert(ad.initialized());
    assert(ad.mb() == 2);
    assert(ad.nb() == 2);

    librpa_int::AtomicBasis ab;
    // 3 atoms, atom 0 and 2 with 1 and atom 1 with 2
    // Process indices
    // | 0   0 | 0   1 |
    // | 0   3 | 3   1 |
    // |-------|-------|
    // | 0   3 | 3   1 |
    // | 2   2 | 2   3 |
    ab.set(std::vector<size_t>{1, 2, 1});

    typedef complex<double> T;
    MPI_Datatype dtype = mpi_datatype<T>::value;

    matrix_m<T> global(ab.nb_total, ab.nb_total, MAJOR::COL);

    if (myid_global == 0)
    {
         global.randomize({0, 0}, {1, 1}, false, true);
    }
    // broadcast
    MPI_Bcast(global.ptr(), m * n, dtype, 0, ad.comm());
    const auto mat_loc_ref = get_local_mat<T>(global, ad);

    // only the upper half
    const std::unordered_map<int, std::vector<atpair_t>> map_proc_IJs_avail{
        {0, {{0, 0}, {0, 1}}},
        {1, {{0, 2}, {1, 2}}},
        {2, {}},
        {3, {{1, 1}, {2, 2}}}};
    unordered_map<atpair_t, matrix_m<T>, atpair_hash> IJmap;
    for (const auto &IJ: map_proc_IJs_avail.at(myid_global))
    {
        const auto &I = IJ.first;
        const auto &J = IJ.second;
        auto mat = matrix_m<T>(ab.get_atom_nb(I), ab.get_atom_nb(J), MAJOR::COL);
        for (int i = 0; i < mat.nr(); i++)
        {
            for (int j = 0; j < mat.nc(); j++)
            {
                mat(i, j) = global(ab.get_part_range()[I] + i, ab.get_part_range()[J]+ j);
            }
        }
        IJmap[IJ] = std::move(mat);
    }
    const auto mat_loc = get_local_mat_from_ap_dist_sy<T>(IJmap, 'u', map_proc_IJs_avail, ab, ad, true, MAJOR::COL);
    // for (int i = 0; i < 4; i++)
    // {
    //     blacs_ctxt_h.barrier();
    //     if (myid_global == i)
    //     {
    //         std::cout << "myid " << i << " local matrix reference" << std::endl;
    //         std::cout << mat_loc_ref;
    //         std::cout << "myid " << i << " local matrix" << std::endl;
    //         std::cout << mat_loc << std::endl;
    //     }
    // }
    blacs_ctxt_h.barrier();
    assert(fequal_array(mat_loc.size(), mat_loc.ptr(), mat_loc_ref.ptr(), false));

    blacs_ctxt_h.exit();
}

template <typename T>
void test_ap_map_from_blacs_dist()
{
    blacs_ctxt_h.set_square_grid(true, librpa_int::CTXT_LAYOUT::R);
    assert(blacs_ctxt_h.nprocs == 4);
    assert(blacs_ctxt_h.nprows == 2);
    assert(blacs_ctxt_h.npcols == 2);

    const size_t m = 4;
    const size_t n = m;

    ArrayDesc ad(blacs_ctxt_h);
    ad.init_1b1p(m, n, 0, 0);
    assert(ad.initialized());
    assert(ad.mb() == 2);
    assert(ad.nb() == 2);

    librpa_int::AtomicBasis ab;
    // 3 atoms, atom 0 and 2 with 1 and atom 1 with 2
    ab.set(std::vector<size_t>{1, 2, 1});
    matrix_m<T> global(ab.nb_total, ab.nb_total, MAJOR::COL);
    if (myid_global == 0)
    {
        if (is_complex<T>())
        {
             global.randomize(0, 1, false, true);
        }
        else
        {
             global.randomize(0, 1, true, false);
        }
    }
    // broadcast
    MPI_Bcast(global.ptr(), m * n, mpi_datatype<T>::value, 0, ad.comm());
    auto mat_loc = get_local_mat<T>(global, ad);

    // if (myid_global == 0) std::cout << "global matrix" << std::endl << global;

    // Test different IJ distribution cases

    // Case one
    // Process indices
    // | 0   0 | 0   1 |
    // | 0   3 | 3   1 |
    // |-------|-------|
    // | 0   3 | 3   1 |
    // | 2   2 | 2   3 |
    {
        const std::unordered_map<int, std::vector<atpair_t>> map_proc_IJs_require{
            {0, {{0, 0}, {1, 0}, {0, 1}}},
            {1, {{0, 2}, {1, 2}}},
            {2, {{2, 0}, {2, 1}}},
            {3, {{1, 1}, {2, 2}}}};
        const auto IJmap = get_ap_map_from_blacs_dist<T>(mat_loc, map_proc_IJs_require, ab, ab, ad);
        const std::vector<std::map<atpair_t, matrix_m<T>>> IJmap_ref_all {
            {{{0, 0}, { {{global(0, 0)}} }}, {{0, 1}, { {{global(0, 1), global(0, 2)}} }}, {{1, 0}, { {{global(1, 0)}, {global(2, 0)}} }}}, // proc 0
            {{{0, 2}, { {{global(0, 3)}} }}, {{1, 2}, { {{global(1, 3)}, {global(2, 3)}} }}}, // proc 1
            {{{2, 0}, { {{global(3, 0)}} }}, {{2, 1}, { {{global(3, 1), global(3, 2)}} }}, }, // proc 2
            {{{2, 2}, { {{global(3, 3)}} }}, {{1, 1}, { {{global(1, 1), global(1, 2)}, {global(2, 1), global(2, 2)}} }}, }, // proc 0
        };
        const auto &IJmap_ref = IJmap_ref_all[myid_global];
        for (int i = 0; i < 4; i++)
        {
            blacs_ctxt_h.barrier();
            if (myid_global == i)
            {
                assert(IJmap.size() == IJmap_ref.size());
                for (const auto &IJ_mat: IJmap)
                {
                    const auto &IJ = IJ_mat.first;
                    const auto &mat = IJ_mat.second;
                    assert(IJmap_ref.count(IJ));
                    const auto &mat_ref = IJmap_ref.at(IJ);
                    assert(mat.size() == mat_ref.size());
                    assert(fequal_array(mat.size(), mat.ptr(), mat_ref.ptr(), false));
                }
            }
        }
    }

    // Case two
    // Process indices
    // | 0   1 | 1   2 |
    // | 3   0 | 0   1 |
    // |-------|-------|
    // | 3   0 | 0   1 |
    // | 2   3 | 3   0 |
    {
        const std::unordered_map<int, std::vector<atpair_t>> map_proc_IJs_require{
            {0, {{0, 0}, {1, 1}, {2, 2}}},
            {1, {{0, 1}, {1, 2}}},
            {2, {{0, 2}, {2, 0}}},
            {3, {{1, 0}, {2, 1}}}};
        const auto IJmap = get_ap_map_from_blacs_dist<T>(mat_loc, map_proc_IJs_require, ab, ab, ad);
        const std::vector<std::map<atpair_t, matrix_m<T>>> IJmap_ref_all {
            {{{0, 0}, { {{global(0, 0)}} }}, {{1, 1}, { {{global(1, 1), global(1, 2)}, {global(2, 1), global(2, 2)}} }}, {{2, 2}, { {{global(3, 3)}} }}, }, // proc 0
            {{{0, 1}, { {{global(0, 1), global(0, 2)}} }}, {{1, 2}, { {{global(1, 3)}, {global(2, 3)}} }}}, // proc 1
            {{{0, 2}, { {{global(0, 3)}} }}, {{2, 0}, { {{global(3, 0)}} }}, }, // proc 2
            {{{1, 0}, { {{global(1, 0)}, {global(2, 0)}} }}, {{2, 1}, { {{global(3, 1), global(3, 2)}} }}, }, // proc 3
        };
        const auto &IJmap_ref = IJmap_ref_all[myid_global];
        for (int i = 0; i < 4; i++)
        {
            blacs_ctxt_h.barrier();
            if (myid_global == i)
            {
                assert(IJmap.size() == IJmap_ref.size());
                for (const auto &IJ_mat: IJmap)
                {
                    const auto &IJ = IJ_mat.first;
                    const auto &mat = IJ_mat.second;
                    assert(IJmap_ref.count(IJ));
                    const auto &mat_ref = IJmap_ref.at(IJ);
                    assert(mat.size() == mat_ref.size());
                    assert(fequal_array(mat.size(), mat.ptr(), mat_ref.ptr(), false));
                }
            }
        }
    }

    blacs_ctxt_h.exit();
}

template <typename T>
void test_restore_local_mat(const std::vector<size_t> &nbs, MAJOR major, const bool print = false)
{
    blacs_ctxt_h.set_square_grid(true, librpa_int::CTXT_LAYOUT::R);
    const auto &nprocs = blacs_ctxt_h.nprocs;
    assert(blacs_ctxt_h.nprows == blacs_ctxt_h.npcols);

    librpa_int::AtomicBasis ab(nbs);
    const auto m = ab.nb_total;
    const auto n = m;
    ArrayDesc ad(blacs_ctxt_h);
    ad.init_1b1p(m, n, 0, 0);
    assert(ad.initialized());

    // initialize global matrix
    matrix_m<T> global(ab.nb_total, ab.nb_total, major);
    if (myid_global == 0)
    {
        if (is_complex<T>())
        {
             global.randomize(0, 1, false, true);
        }
        else
        {
             global.randomize(0, 1, true, false);
        }
    }
    // broadcast
    MPI_Bcast(global.ptr(), m * n, mpi_datatype<T>::value, 0, ad.comm());
    if (print && myid_global == 0) std::cout << "global matrix" << std::endl << global;
    const auto mat_loc_ref = get_local_mat<T>(global, ad);

    const auto &pairs = generate_atom_pair_from_nat(ab.n_atoms, true);
    assert(pairs.size() == ab.n_atoms * ab.n_atoms);
    // distribute atom pairs to processes - a naive distribution by order
    std::unordered_map<int, std::set<atpair_t>> IJs;
    for (size_t i = 0; i < pairs.size(); i++)
    {
        IJs[i % nprocs].insert(pairs[i]);
    }
    if (print && myid_global == 0) std::cout << IJs << std::endl;

    IndexScheduler sched;
    sched.init(IJs, ab, ab, ad, major == MAJOR::ROW);

    const auto IJmap = get_ap_map_from_blacs_dist_scheduler(mat_loc_ref, sched, ab, ab, ad);
    for (int i = 0; i < size_global; i++)
    {
        blacs_ctxt_h.barrier();
        if (myid_global == i)
        {
            if (print)
            {
                std::cout << "myid " << i << " AP matrices reference" << std::endl;
            }
            for (const auto &IJ_mat: IJmap)
            {
                const auto &IJ = IJ_mat.first;
                const auto &mat = IJ_mat.second;
                if (print) std::cout << "IJ " << IJ << std::endl;
                const auto mat_ref = get_ap_block_from_global(global, IJ, ab, ab);
                if (print)
                {
                    fequal_array(mat_ref.size(), mat_ref.ptr(), mat.ptr(), print);
                }
                else
                {
                    assert(fequal_array(mat_ref.size(), mat_ref.ptr(), mat.ptr(), print));
                }
            }
        }
    }
    // blacs_ctxt_h.exit();
    // return;
    const auto mat_loc = get_local_mat_from_ap_dist_scheduler(IJmap, sched, ab, ab, ad, major);

    for (int i = 0; i < size_global; i++)
    {
        blacs_ctxt_h.barrier();
        if (print && myid_global == i)
        {
            std::cout << "myid " << i << " local matrix reference" << std::endl;
            std::cout << mat_loc_ref;
            std::cout << "myid " << i << " local matrix" << std::endl;
            std::cout << mat_loc << std::endl;
            assert(fequal_array(mat_loc.size(), mat_loc.ptr(), mat_loc_ref.ptr(), false));
            // fequal_array(mat_loc.size(), mat_loc.ptr(), mat_loc_ref.ptr(), true);
        }
    }

    blacs_ctxt_h.exit();
}

template <typename T>
void test_restore_local_mat_scheduler(const std::vector<size_t> &nbs, MAJOR major, const bool print = false)
{
    blacs_ctxt_h.set_square_grid(true, librpa_int::CTXT_LAYOUT::R);
    const auto &nprocs = blacs_ctxt_h.nprocs;
    assert(blacs_ctxt_h.nprows == blacs_ctxt_h.npcols);

    librpa_int::AtomicBasis ab(nbs);
    const auto m = ab.nb_total;
    const auto n = m;
    ArrayDesc ad(blacs_ctxt_h);
    ad.init_1b1p(m, n, 0, 0);
    assert(ad.initialized());

    // initialize global matrix
    matrix_m<T> global(ab.nb_total, ab.nb_total, major);
    if (myid_global == 0)
    {
        global.randomize(0, 1, false, false);
    }
    // broadcast
    MPI_Bcast(global.ptr(), m * n, mpi_datatype<T>::value, 0, ad.comm());
    if (print && myid_global == 0) std::cout << "global matrix" << std::endl << global;
    const auto mat_loc_ref = get_local_mat<T>(global, ad);

    const auto &pairs = generate_atom_pair_from_nat(ab.n_atoms, true);
    assert(pairs.size() == ab.n_atoms * ab.n_atoms);
    // distribute atom pairs to processes - a naive distribution by order
    std::unordered_map<int, std::vector<atpair_t>> IJs;
    for (size_t i = 0; i < pairs.size(); i++)
    {
        IJs[i % nprocs].emplace_back(pairs[i]);
    }
    if (print && myid_global == 0) std::cout << IJs << std::endl;

    const auto IJmap = get_ap_map_from_blacs_dist(mat_loc_ref, IJs, ab, ab, ad);
    for (int i = 0; i < size_global; i++)
    {
        blacs_ctxt_h.barrier();
        if (myid_global == i)
        {
            if (print)
            {
                std::cout << "myid " << i << " AP matrices reference" << std::endl;
            }
            for (const auto &IJ_mat: IJmap)
            {
                const auto &IJ = IJ_mat.first;
                const auto &mat = IJ_mat.second;
                if (print) std::cout << "IJ " << IJ << std::endl;
                const auto mat_ref = get_ap_block_from_global(global, IJ, ab, ab);
                if (print)
                {
                    fequal_array(mat_ref.size(), mat_ref.ptr(), mat.ptr(), true);
                }
                else
                {
                    assert(fequal_array(mat_ref.size(), mat_ref.ptr(), mat.ptr(), false));
                }
            }
        }
    }
    const auto mat_loc = get_local_mat_from_ap_dist(IJmap, IJs, ab, ab, ad, major);

    for (int i = 0; i < size_global; i++)
    {
        blacs_ctxt_h.barrier();
        if (myid_global == i)
        {
            if (print)
            {
                std::cout << "myid " << i << " local matrix reference" << std::endl;
                std::cout << mat_loc_ref;
                std::cout << "myid " << i << " local matrix" << std::endl;
                std::cout << mat_loc << std::endl;
                fequal_array(mat_loc.size(), mat_loc.ptr(), mat_loc_ref.ptr(), true);
            }
            else
            {
                assert(fequal_array(mat_loc.size(), mat_loc.ptr(), mat_loc_ref.ptr(), false));
            }
        }
    }

    blacs_ctxt_h.exit();
}

template <typename T>
void test_restore_ap_map(const std::vector<size_t> &nbs, MAJOR major, const bool print = false)
{
    blacs_ctxt_h.set_square_grid(true, librpa_int::CTXT_LAYOUT::R);

    const auto &nprocs = blacs_ctxt_h.nprocs;
    assert(blacs_ctxt_h.nprows == blacs_ctxt_h.npcols);

    librpa_int::AtomicBasis ab(nbs);
    const auto m = ab.nb_total;
    const auto n = m;
    ArrayDesc ad(blacs_ctxt_h);
    ad.init_1b1p(m, n, 0, 0);
    assert(ad.initialized());

    // initialize global matrix
    matrix_m<T> global(ab.nb_total, ab.nb_total, major);
    if (myid_global == 0)
    {
        if (is_complex<T>())
        {
             global.randomize(0, 1, false, true);
        }
        else
        {
             global.randomize(0, 1, true, false);
        }
    }
    if (print && myid_global == 0) std::cout << "global matrix" << std::endl << global;

    // broadcast
    MPI_Bcast(global.ptr(), m * n, mpi_datatype<T>::value, 0, ad.comm());
    const auto &pairs = generate_atom_pair_from_nat(ab.n_atoms, true);
    assert(pairs.size() == ab.n_atoms * ab.n_atoms);
    // distribute atom pairs to processes - a naive distribution by order
    std::unordered_map<int, std::vector<atpair_t>> IJs;
    for (size_t i = 0; i < pairs.size(); i++)
    {
        IJs[i % nprocs].emplace_back(pairs[i]);
    }
    std::unordered_map<atpair_t, matrix_m<T>, atpair_hash> IJmap_ref;
    if (IJs.count(myid_global))
    {
        for (const auto &IJ: IJs.at(myid_global))
        {
            IJmap_ref[IJ] = get_ap_block_from_global(global, IJ, ab, ab);
        }
    }
    // for (int i = 0; i < 4; i++)
    // {
    //     blacs_ctxt_h.barrier();
    //     // cout << "part_range: " << ab.get_part_range() << endl;
    //     if (myid_global == i)
    //     {
    //         // std::cout << "myid " << i << std::endl;
    //         for (const auto &IJ: IJs.at(myid_global))
    //         {
    //             // cout << IJ << endl;
    //             IJmap_ref[IJ] = get_ap_block_from_global(global, IJ, ab, ab);
    //             // cout << IJmap_ref[IJ];
    //         }
    //     }
    // }
    const auto mat_loc = get_local_mat_from_ap_dist(IJmap_ref, IJs, ab, ab, ad, major);
    const auto IJmap = get_ap_map_from_blacs_dist(mat_loc, IJs, ab, ab, ad);

    for (int i = 0; i < size_global; i++)
    {
        blacs_ctxt_h.barrier();
        if (myid_global == i)
        {
            // std::cout << "myid " << i << " comparing atom pair mapping data" << std::endl;
            // std::cout << "map size: ref " << IJmap_ref.size() << " - test " << IJmap.size() << std::endl;
            assert(IJmap.size() == IJmap_ref.size());
            for (const auto &IJ_mat: IJmap)
            {
                const auto &IJ = IJ_mat.first;
                const auto &mat = IJ_mat.second;
                // if (!IJmap_ref.count(IJ))
                // {
                //     std::cout << "ref has no " << IJ << " ! " << std::endl;
                //     std::cout << "test" << std::endl << mat;
                //     continue;
                // }
                // std::cout << "comparing common key " << IJ << std::endl;
                // std::cout << "test" << std::endl << mat;
                // std::cout << "ref" << std::endl << mat_ref;
                assert(IJmap_ref.count(IJ));
                const auto &mat_ref = IJmap_ref.at(IJ);
                assert(mat.size() == mat_ref.size());
                assert(fequal_array(mat.size(), mat.ptr(), mat_ref.ptr(), false));
                // fequal_array(mat.size(), mat.ptr(), mat_ref.ptr(), true);
            }
        }
    }

    blacs_ctxt_h.exit();
}

template <typename T>
void test_restore_ap_map_scheduler(const std::vector<size_t> &nbs, MAJOR major, const bool print = false)
{
    blacs_ctxt_h.set_square_grid(true, librpa_int::CTXT_LAYOUT::R);

    const auto &nprocs = blacs_ctxt_h.nprocs;
    assert(blacs_ctxt_h.nprows == blacs_ctxt_h.npcols);

    librpa_int::AtomicBasis ab(nbs);
    const auto m = ab.nb_total;
    const auto n = m;
    ArrayDesc ad(blacs_ctxt_h);
    ad.init_1b1p(m, n, 0, 0);
    assert(ad.initialized());

    // initialize global matrix
    matrix_m<T> global(ab.nb_total, ab.nb_total, major);
    if (myid_global == 0)
    {
        global.randomize(0, 1, false, false);
    }
    if (print && myid_global == 0) std::cout << "global matrix" << std::endl << global;

    // broadcast
    MPI_Bcast(global.ptr(), m * n, mpi_datatype<T>::value, 0, ad.comm());
    const auto &pairs = generate_atom_pair_from_nat(ab.n_atoms, true);
    assert(pairs.size() == ab.n_atoms * ab.n_atoms);
    // distribute atom pairs to processes - a naive distribution by order
    std::unordered_map<int, std::set<atpair_t>> IJs;
    for (size_t i = 0; i < pairs.size(); i++)
    {
        IJs[i % nprocs].insert(pairs[i]);
    }
    std::unordered_map<atpair_t, matrix_m<T>, atpair_hash> IJmap_ref;
    if (IJs.count(myid_global))
    {
        for (const auto &IJ: IJs.at(myid_global))
        {
            IJmap_ref[IJ] = get_ap_block_from_global(global, IJ, ab, ab);
        }
    }
    // for (int i = 0; i < 4; i++)
    // {
    //     blacs_ctxt_h.barrier();
    //     // cout << "part_range: " << ab.get_part_range() << endl;
    //     if (myid_global == i)
    //     {
    //         // std::cout << "myid " << i << std::endl;
    //         for (const auto &IJ: IJs.at(myid_global))
    //         {
    //             // cout << IJ << endl;
    //             IJmap_ref[IJ] = get_ap_block_from_global(global, IJ, ab, ab);
    //             // cout << IJmap_ref[IJ];
    //         }
    //     }
    // }
    IndexScheduler sched;
    sched.init(IJs, ab, ab, ad, major == MAJOR::ROW);
    const auto mat_loc = get_local_mat_from_ap_dist_scheduler(IJmap_ref, sched, ab, ab, ad, major);
    const auto IJmap = get_ap_map_from_blacs_dist_scheduler(mat_loc, sched, ab, ab, ad);

    for (int i = 0; i < size_global; i++)
    {
        blacs_ctxt_h.barrier();
        if (myid_global == i)
        {
            // std::cout << "myid " << i << " comparing atom pair mapping data" << std::endl;
            // std::cout << "map size: ref " << IJmap_ref.size() << " - test " << IJmap.size() << std::endl;
            assert(IJmap.size() == IJmap_ref.size());
            for (const auto &IJ_mat: IJmap)
            {
                const auto &IJ = IJ_mat.first;
                const auto &mat = IJ_mat.second;
                // if (!IJmap_ref.count(IJ))
                // {
                //     std::cout << "ref has no " << IJ << " ! " << std::endl;
                //     std::cout << "test" << std::endl << mat;
                //     continue;
                // }
                // std::cout << "comparing common key " << IJ << std::endl;
                // std::cout << "test" << std::endl << mat;
                // std::cout << "ref" << std::endl << mat_ref;
                assert(IJmap_ref.count(IJ));
                const auto &mat_ref = IJmap_ref.at(IJ);
                assert(mat.size() == mat_ref.size());
                assert(fequal_array(mat.size(), mat.ptr(), mat_ref.ptr(), false));
                // fequal_array(mat.size(), mat.ptr(), mat_ref.ptr(), true);
            }
        }
    }

    blacs_ctxt_h.exit();
}

template <typename T>
void test_restore_local_mat_sy(const std::vector<size_t> &nbs, MAJOR major)
{
    blacs_ctxt_h.set_square_grid(true, librpa_int::CTXT_LAYOUT::R);
    const auto &nprocs = blacs_ctxt_h.nprocs;
    assert(nprocs == 4);
    assert(blacs_ctxt_h.nprows == 2);
    assert(blacs_ctxt_h.npcols == 2);

    librpa_int::AtomicBasis ab(nbs);
    const auto m = ab.nb_total;
    const auto n = m;
    ArrayDesc ad(blacs_ctxt_h);
    ad.init_1b1p(m, n, 0, 0);
    assert(ad.initialized());

    // initialize global matrix
    matrix_m<T> global(ab.nb_total, ab.nb_total, major);
    if (myid_global == 0)
    {
        if (is_complex<T>())
        {
             global.randomize(0, 1, false, true);
        }
        else
        {
             global.randomize(0, 1, true, false);
        }
    }
    // broadcast
    MPI_Bcast(global.ptr(), m * n, mpi_datatype<T>::value, 0, ad.comm());
    const auto mat_loc_ref = get_local_mat<T>(global, ad);

    const auto &pairs = generate_atom_pair_from_nat(ab.n_atoms, false);
    assert(pairs.size() == ab.n_atoms * (ab.n_atoms + 1) / 2);
    // distribute atom pairs to processes - a naive distribution by order
    std::unordered_map<int, std::vector<atpair_t>> IJs;
    for (size_t i = 0; i < pairs.size(); i++)
    {
        IJs[i % nprocs].emplace_back(pairs[i]);
    }
    // if (myid_global == 0) std::cout << IJs << std::endl;

    const auto IJmap = get_ap_map_from_blacs_dist(mat_loc_ref, IJs, ab, ab, ad);
    const auto mat_loc = get_local_mat_from_ap_dist_sy(IJmap, 'u', IJs, ab, ad, true, major);

    for (int i = 0; i < 4; i++)
    {
        blacs_ctxt_h.barrier();
        if (myid_global == i)
        {
            // std::cout << "myid " << i << " local matrix reference" << std::endl;
            // std::cout << mat_loc_ref;
            // std::cout << "myid " << i << " local matrix" << std::endl;
            // std::cout << mat_loc << std::endl;
            assert(fequal_array(mat_loc.size(), mat_loc.ptr(), mat_loc_ref.ptr(), false));
            // fequal_array(mat_loc.size(), mat_loc.ptr(), mat_loc_ref.ptr(), true);
        }
    }

    blacs_ctxt_h.exit();
    blacs_ctxt_h.exit();
}

static void initialize()
{
    init_global_mpi(MPI_COMM_WORLD);
    blacs_ctxt_h.reset_comm(global::mpi_comm_global);
    init_global_io();
}

static void finalize()
{
    finalize_global_io();
    finalize_global_mpi();
}

int main (int argc, char *argv[])
{
    int provided;
    MPI_Init_thread(&argc, &argv, LIBRPA_MPI_THREAD_LEVEL, &provided);
    initialize();

    // test_pgemm<float>(-2, 1);
    // test_pgemm<double>(-2, 1);
    // test_pgemm<complex<float>>({-2, -1}, {1, 0});
    // test_pgemm<complex<double>>({-2, -1}, {1, 0});
    //
    test_power_hemat_blacs_square_grid<complex<double>>(0.0, {1.0, 1.0});
    test_power_hemat_blacs_square_grid<complex<double>>(0.0, {1.0, 1.0}, true);
    // test_power_hemat_blacs_square_grid<complex<float>>(0.0, {1.0, 2.0});

    test_invert_scalapack<float>();
    test_invert_scalapack<double>();
    test_invert_scalapack<complex<float>>();
    test_invert_scalapack<complex<double>>();

    // test_collect_block_from_IJ_storage<double>();
    // test_collect_block_from_IJ_storage<complex<double>>();
    test_collect_block_from_ALL_IJ_Tensor_sparse_zero_missing<double>();
    test_collect_block_from_ALL_IJ_Tensor_sparse_zero_missing<complex<double>>();

    test_local_mat_from_ap_dist<double>();
    test_local_mat_from_ap_dist<complex<double>>();
    test_local_mat_from_ap_dist_he();

    test_ap_map_from_blacs_dist<double>();
    test_ap_map_from_blacs_dist<complex<double>>();

    test_restore_local_mat<double>({1}, MAJOR::COL);
    test_restore_local_mat<double>({2}, MAJOR::COL);
    test_restore_local_mat<double>({4, 3}, MAJOR::COL);
    test_restore_local_mat<double>({1, 2, 1}, MAJOR::COL);
    test_restore_local_mat<double>({1, 2, 1, 2}, MAJOR::COL, false);
    // test_restore_local_mat<double>({2, 4, 1, 3}, MAJOR::COL);
    test_restore_local_mat<complex<double>>({3, 5, 1}, MAJOR::COL);

    test_restore_local_mat<double>({1}, MAJOR::ROW);
    test_restore_local_mat<double>({2}, MAJOR::ROW);
    test_restore_local_mat<double>({4, 3}, MAJOR::ROW);
    test_restore_local_mat<double>({1, 2, 1}, MAJOR::ROW);
    test_restore_local_mat<double>({1, 2, 1, 2}, MAJOR::ROW);
    test_restore_local_mat<double>({2, 4, 1, 3}, MAJOR::ROW);
    test_restore_local_mat<complex<double>>({3, 5, 1}, MAJOR::ROW);

    // large cases
    test_restore_local_mat_scheduler<complex<double>>({418, 643}, MAJOR::ROW);
    test_restore_local_mat_scheduler<complex<double>>(std::vector<size_t>(50, 20), MAJOR::COL);
    test_restore_local_mat_scheduler<double>({200, 150, 401, 300}, MAJOR::ROW);

    test_restore_ap_map<double>({1}, MAJOR::COL);
    test_restore_ap_map<double>({2}, MAJOR::COL);
    test_restore_ap_map<double>({4, 3}, MAJOR::COL);
    test_restore_ap_map<double>({1, 2, 1}, MAJOR::COL);
    test_restore_ap_map<double>({1, 2, 1, 2}, MAJOR::COL);
    test_restore_ap_map<double>({2, 4, 1, 3}, MAJOR::COL);
    test_restore_ap_map<complex<double>>({3, 5, 1}, MAJOR::COL);

    test_restore_ap_map<double>({1}, MAJOR::ROW);
    test_restore_ap_map<double>({2}, MAJOR::ROW);
    test_restore_ap_map<double>({4, 3}, MAJOR::ROW);
    test_restore_ap_map<double>({1, 2, 1}, MAJOR::ROW);
    test_restore_ap_map<double>({1, 2, 1, 2}, MAJOR::ROW);
    test_restore_ap_map<double>({2, 4, 1, 3}, MAJOR::ROW);
    test_restore_ap_map<complex<double>>({3, 5, 1}, MAJOR::ROW);

    // large cases
    test_restore_ap_map_scheduler<complex<double>>({13, 13, 18}, MAJOR::ROW);
    test_restore_local_mat_scheduler<complex<double>>({13, 13, 18}, MAJOR::ROW);
    test_restore_ap_map_scheduler<complex<double>>(std::vector<size_t>(50, 20), MAJOR::ROW);
    test_restore_local_mat_scheduler<complex<double>>({418, 643}, MAJOR::COL);
    test_restore_ap_map_scheduler<double>({200, 150, 401, 300}, MAJOR::COL);

    test_restore_local_mat_sy<double>({1}, MAJOR::COL);
    test_restore_local_mat_sy<double>({2}, MAJOR::ROW);
    // test_restore_local_mat_sy<double>({4, 3}, MAJOR::COL); // FAIL
    // test_restore_local_mat_sy<double>({1, 2, 1}, MAJOR::ROW); // FAIL
    // test_restore_local_mat_sy<double>({2, 4, 1, 3}, MAJOR::COL); // FAIL
    // test_restore_local_mat_sy<complex<double>>({3, 5, 1}, MAJOR::COL); // FAIL

    finalize();
    MPI_Finalize();
    return 0;
}
