#pragma once

#include <algorithm>
#include <array>
#include <stdexcept>
#include <vector>

#include "../core/dielecmodel.h"
#include "../interface/la_connector.h"
#ifdef LIBRPA_USE_DDLA
#include <ddla/ddla_vector.h>
#endif

// Test callers own preparation/transfers just as the production Wc caller does.
namespace headwing_test
{
using namespace librpa_int;
inline void rewrite(Matz &epsilon, const Matz &sqrtv, const Matz &eigenvectors,
    const Matz &direction, const ArrayDesc &head_desc, const Matz &head,
    const Matz &wing, const std::vector<double> &qx, const std::vector<double> &qy,
    const std::vector<double> &qz, const std::vector<double> &rho,
    const ArrayDesc &layout, const BlacsCtxtHandler &grid,
    std::size_t retained, double threshold, bool cholesky, bool gpu,
    diele_func *model = nullptr, int ifreq = 0)
{
    using Z = std::complex<double>;
    const int n = layout.m(), m = 1 + n - static_cast<int>(retained);
    if (retained == 0 || retained > std::size_t(n) || head.nr() != 3 || head.nc() != 3)
        throw std::invalid_argument("invalid head/wing test input");
    (void)threshold;
    ArrayDesc desc(grid), yd(grid), wd(grid), sd(grid), cd(grid);
    desc.init(n, n, layout.mb(), layout.nb(), layout.irsrc(), layout.icsrc());
    yd.init(n, m, layout.mb(), layout.nb(), layout.irsrc(), layout.icsrc());
    wd.init(n, 3, layout.mb(), layout.nb(), layout.irsrc(), layout.icsrc());
    sd.init(m, m, layout.mb(), layout.nb(), layout.irsrc(), layout.icsrc());
    cd.init(3, 3, layout.mb(), layout.nb(), layout.irsrc(), layout.icsrc());
    for (auto *d : {&desc, &yd, &wd, &sd, &cd})
    {
        d->set_linear_algorithm("scalapack");
#ifdef LIBRPA_USE_DDLA
        if (gpu || (layout.use_ddla_linear() && grid.ddla_cpu_handle != nullptr))
        {
            d->set_linear_algorithm("ddla");
            d->set_ddla_handle(gpu ? grid.ddla_handle : grid.ddla_cpu_handle);
        }
#endif
    }
    const auto storage = [](const ArrayDesc &d) {
        return std::max<std::size_t>(1, std::size_t(d.lld()) * d.n_loc());
    };
    const auto count = std::max(storage(desc), storage(wd));
    std::vector<Z> e(storage(desc)), f(count), t(count), y(storage(yd)),
        z(storage(wd)), small(std::max(storage(sd), storage(cd)));
    if (epsilon.size()) std::copy_n(epsilon.ptr(), epsilon.size(), e.data());
    if (sqrtv.size()) std::copy_n(sqrtv.ptr(), sqrtv.size(), f.data());
    ScalapackConnector::pgemr2d_f(n, 1, direction.ptr(), 1, 1, head_desc.desc,
        y.data(), 1, 1, yd.desc, grid.ictxt);
    if (m > 1)
        ScalapackConnector::pgemr2d_f(n, m - 1, eigenvectors.ptr(), 1, int(retained) + 1,
            layout.desc, y.data(), 1, 2, yd.desc, grid.ictxt);
    for (int j = 0; j < wd.n_loc(); ++j)
        for (int i = 0; i < wd.m_loc(); ++i)
            t[i + wd.lld() * j] = wing(wd.indx_l2g_r(i), wd.indx_l2g_c(j));
    std::array<Z, 9> host_head{};
    for (int j = 0; j < 3; ++j)
        for (int i = 0; i < 3; ++i) host_head[i + 3 * j] = head(i, j);
    const auto apply = [&](Z *ep, Z *fp, Z *tp, Z *sp, const Z *yp, Z *zp) {
        LaConnector::pgemm('N', 'N', n, 3, n, Z(1), fp, 1, 1, desc,
            tp, 1, 1, wd, Z(0), zp, 1, 1, wd);
        if (model)
            model->rewrite_eps_abf_space(ep, fp, tp, sp, yp, zp, ifreq,
                desc, yd, wd, sd, cd, cholesky);
        else
            librpa_int::rewrite_eps_abf_space(ep, fp, tp, sp, yp, zp, host_head.data(),
                desc, yd, wd, sd, cd, qx, qy, qz, rho, cholesky);
    };
#if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
    if (gpu)
    {
        const auto handle = grid.ddla_handle;
        ddla::vector<Z> de(handle, e.size()), df(handle, f.size()), dt(handle, t.size()),
            dy(handle, y.size()), dz(handle, z.size()), ds(handle, small.size());
        LaConnector::copy_local(de.data(), e.data(), e.size(), desc);
        LaConnector::copy_local(df.data(), f.data(), f.size(), desc);
        LaConnector::copy_local(dt.data(), t.data(), t.size(), desc);
        LaConnector::copy_local(dy.data(), y.data(), y.size(), desc);
        LaConnector::synchronize(desc);
        apply(de.data(), df.data(), dt.data(), ds.data(), dy.data(), dz.data());
        LaConnector::copy_local(e.data(), de.data(), e.size(), desc);
        LaConnector::synchronize(desc);
    }
    else
#else
    if (gpu) throw std::runtime_error("GPU head/wing test requires a GPU build");
#endif
        apply(e.data(), f.data(), t.data(), small.data(), y.data(), z.data());
    if (epsilon.size()) std::copy_n(e.data(), epsilon.size(), epsilon.ptr());
}
} // namespace headwing_test
