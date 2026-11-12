#include "../core/meanfield_mpi.h"

#include <cmath>
#include <complex>
#include <iostream>
#include <iterator>
#include <map>
#include <stdexcept>
#include <vector>

#include "../mpi/global_mpi.h"
#include "../math/matrix_m.h"
#include "../interface/scalapack/scalapack_connector.h"

#include "../io/stl_io_helper.h"
#include "mpi_test_config.h"
#include "testutils.h"

#if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
#include <ddla/ddla.h>
#endif

using namespace std;
using namespace librpa_int;
using namespace librpa_int::global;

static ArrayDesc make_permanent_wfc_desc(
    const KPointBlacsParallelContext &context, const int nao, const int nstates)
{
    const int block_ao =
        get_capped_blacs_block_size(nao, wfc_gemm_block_size_opt, context.blacs_h);
    const int block_state =
        get_capped_blacs_block_size(nstates, wfc_gemm_block_size_opt, context.blacs_h);
    return context.create_array_desc(nao, nstates, block_ao, block_state);
}

static MeanField init_mf_pbc(int nk, int nb, int nocc, double gap,
                             std::vector<Vector3_Order<double>> &kfrac_list,
                             std::vector<Vector3_Order<int>> &Rs)
{
    assert(nk > 0 && nb > 0);
    const int nspin = 1;
    MeanField mf(nspin, nk, nb, nb);
    const double efermi = 0.0;
    mf.get_efermi() = efermi;
    gap = std::abs(gap);
    for (int ik = 0; ik < nk; ik++)
    {
        for (int ib = 0; ib < nocc; ib++)
        {
            mf.get_weight().at(0)(ik, ib) = 2.0 / nspin / nk;
            mf.get_eigenvals().at(0)(ik, ib) = efermi - 0.5 * gap;
        }
        for (int ib = nocc; ib < nb; ib++)
        {
            mf.get_eigenvals().at(0)(ik, ib) = efermi + 0.5 * gap;
        }
    }

    kfrac_list.clear();
    Rs.clear();
    for (int ik = 0; ik < nk; ik++)
    {
        Vector3_Order<double> kf{double(ik) / nk, 0, 0};
        kfrac_list.emplace_back(kf);
        if (ik % size_global == myid_global)
        {
            ComplexMatrix mat(nb, nb);
            // Every eigenvector matrix is identity
            // Only (0,0,0) has non-zero density matrix. Other R-point are cancelled out.
            mat.set_as_identity_matrix();
            mf.get_eigenvectors()[0][0][ik] = std::move(mat);
        }
    }

    for (int iR = 0; iR < nk; iR++)
    {
        Vector3_Order<int> R{iR, 0, 0};
        if (iR % size_global == myid_global) Rs.emplace_back(R);
    }

    return mf;
}

static void test_dmat_cplx_Rs_kpara(int nk, int nb, int nocc)
{
    // ofs_myid << kfrac_list << endl;
    std::vector<Vector3_Order<double>> kfrac_list;
    std::vector<Vector3_Order<int>> Rs;
    const auto mf = init_mf_pbc(nk, nb, nocc, 1.0, kfrac_list, Rs);

    // cout << mf.get_n_kpoints() << endl;
    // cout << mf.get_n_spinor() << endl;
    // cout << mf.get_n_spins() << endl;

    const auto dm_Rs = get_dmat_cplx_Rs_kpara(0, mf, kfrac_list, Rs, mpi_comm_global_h);
    assert(dm_Rs.size() == Rs.size());
    for (const auto &[R, rmat]: dm_Rs)
    {
        if (R.x == 0 && R.y == 0 && R.z == 0)
        {
            for (int ib = 0; ib < nocc; ib++)
            {
                assert(fequal(rmat(ib, ib), {1.0, 0.0}));
            }
            for (int ib = nocc; ib < nb; ib++)
            {
                assert(fequal(rmat(ib, ib), {0.0, 0.0}));
            }
        }
        else
        {
            for (int ib = 0; ib < nb; ib++)
            {
                assert(fequal(rmat(ib, ib), {0.0, 0.0}));
            }
        }
    }
}

static void test_gf_cplx_Rs_kpara(int nk, int nb, int nocc, double gap, double tau, int n_bands = -1, int n_excluded = 0)
{
    std::vector<Vector3_Order<double>> kfrac_list;
    std::vector<Vector3_Order<int>> Rs;
    const auto mf = init_mf_pbc(nk, nb, nocc, gap, kfrac_list, Rs);

    std::vector<bool> mask(nb, true);
    if (n_bands >= 0) std::fill(mask.begin() + n_bands, mask.end(), false);
    std::fill_n(mask.begin(), n_excluded, false);
    const auto gf_Rs = get_gf_cplx_imagtimes_Rs_kpara(0, mf, kfrac_list, {tau}, Rs, mpi_comm_global_h, mask);
    assert(gf_Rs.size() == 1);
    assert(gf_Rs.at(tau).size() == Rs.size());
    const double coeff = std::exp(- 0.5 * std::abs(gap * tau)) * (tau > 0? 1.0: -1.0);
    for (const auto &[tau, trmat]: gf_Rs)
    {
        for (const auto &[R, rmat]: trmat)
        {
            // ofs_myid << "tau " << tau << " R " << R << endl;
            // print_complex_matrix("", rmat, ofs_myid, true);
            if (R.x == 0 && R.y == 0 && R.z == 0)
            {
                for (int ib = 0; ib < nocc; ib++)
                {
                    if (!mask[ib]) assert(fequal(rmat(ib, ib), {0.0, 0.0}));
                    else if (tau > 0) assert(fequal(rmat(ib, ib), {0.0, 0.0}));
                    else assert(fequal(rmat(ib, ib), {coeff, 0.0}));
                }
                for (int ib = nocc; ib < nb; ib++)
                {
                    if (!mask[ib]) assert(fequal(rmat(ib, ib), {0.0, 0.0}));
                    else if (tau > 0) assert(fequal(rmat(ib, ib), {coeff, 0.0}));
                    else assert(fequal(rmat(ib, ib), {0.0, 0.0}));
                }
            }
            else
            {
                for (int ib = 0; ib < nb; ib++)
                {
                    assert(fequal(rmat(ib, ib), {0.0, 0.0}));
                }
            }
        }
    }
}

// Actual data from FHI-aims: a=3A, k222, light, minimal + 2p in tier1, PBE
static void get_BCC_He_k222_light_min_2p_aims_ref(int &nk, int &nb, std::vector<Vector3_Order<double>> &kfrac_list,
                                                  double &efermi, Matd &eigvals, Matd &eigvecs)
{
    nk = 8;
    nb = 8;
    efermi = 0.00609624851934464917;
    kfrac_list = {
        {0.0, 0.0, 0.0}, {0.0, 0.0, 0.5}, {0.0, 0.5, 0.0}, {0.0, 0.5, 0.5},
        {0.5, 0.0, 0.0}, {0.5, 0.0, 0.5}, {0.5, 0.5, 0.0}, {0.5, 0.5, 0.5},
    };
    // Set eigenvalues
    // Nk * nb
    eigvals = Matd ({
        {-0.626245559348763803E+00, -0.549248421709219947E+00, 0.795715865718289783E+00, 0.795715870153016036E+00, 0.795715870153017368E+00, 0.114519974490557885E+01, 0.114519975015467290E+01, 0.114519975015467534E+01,},
        {-0.582366828421241656E+00, -0.582366828421236327E+00, 0.778608717055217614E+00, 0.778608717055219834E+00, 0.969761906335162127E+00, 0.969761906335163459E+00, 0.969761911169224722E+00, 0.969761911169229496E+00,},
        {-0.582366828421241323E+00, -0.582366828421236993E+00, 0.778608717055214727E+00, 0.778608717055218169E+00, 0.969761906335162793E+00, 0.969761906335163459E+00, 0.969761911169227608E+00, 0.969761911169228386E+00,},
        {-0.573728708427312606E+00, -0.573728708427307832E+00, 0.561440920781375419E+00, 0.561440920781376196E+00, 0.991942055871926631E+00, 0.991942055871928297E+00, 0.111808903061832510E+01, 0.111808903061832710E+01,},
        {-0.582366828368694245E+00, -0.582366828368689249E+00, 0.778608712477487197E+00, 0.778608712477488418E+00, 0.969761911187294712E+00, 0.969761911187297154E+00, 0.969761911187298264E+00, 0.969761911187299264E+00,},
        {-0.573728708371605944E+00, -0.573728708371599949E+00, 0.561440918747909912E+00, 0.561440918747910134E+00, 0.991942060772925349E+00, 0.991942060772929901E+00, 0.111808902789979281E+01, 0.111808902789979303E+01,},
        {-0.573728708371605833E+00, -0.573728708371599283E+00, 0.561440918747909246E+00, 0.561440918747910134E+00, 0.991942060772928014E+00, 0.991942060772929790E+00, 0.111808902789979125E+01, 0.111808902789979303E+01,},
        {-0.566863956270889036E+00, -0.566863956270882818E+00, 0.810843336307190632E+00, 0.810843336307191853E+00, 0.810843340998502415E+00, 0.810843340998505191E+00, 0.810843340998505968E+00, 0.810843340998506190E+00,},
    }, MAJOR::ROW);
    // Set eigenvectors
    eigvecs = Matd ({
        {-0.635395184437369198e+00, -0.747426907197621060e+00, -0.148425330272811403e-15, -0.288193135129330573e-15, -0.657121997774521427e-16,  0.213174851968120430e-15,  0.240912789689757665e-15, -0.256391196735161627e-15,  0.110673002931629419e-15,  0.131735844079276242e-15, -0.101365202837211461e-07, -0.679882091280804679e+00,  0.488725375880342608e-01, -0.131152925637379580e-07,  0.786207917070590234e+00, -0.337952427417521975e-01, -0.834460907048564821e-16, -0.748749936980692225e-16, -0.212348341907902352e-07, -0.488725375880331575e-01, -0.679882091280802570e+00, -0.258182827117184054e-07, -0.337952427417522738e-01, -0.786207917070593121e+00, -0.710834324854303791e-16, -0.584192382463665804e-16, -0.681636400858000813e+00,  0.116329449607769049e-07,  0.204534062770632809e-07,  0.786933928164528673e+00,  0.119944143455211349e-07, -0.263577059878771378e-07, -0.635395184437318683e+00,  0.747426907197663914e+00, -0.306047631033667958e-15, -0.789022149956443225e-16,  0.100542437904474730e-15, -0.264900625458063534e-15, -0.187675891989289404e-15, -0.136534980723661598e-16, -0.231171062640870563e-15, -0.160665159717294532e-29,  0.101365200400602554e-07,  0.679882091280795797e+00, -0.488725375880334975e-01, -0.131152923280523871e-07,  0.786207917070598672e+00, -0.337952427417515036e-01, -0.261173419814486132e-15, -0.454507092264657445e-15,  0.212348350078658308e-07,  0.488725375880335947e-01,  0.679882091280797907e+00, -0.258182822766050813e-07, -0.337952427417524889e-01, -0.786207917070596007e+00, -0.189234527182622590e-15,  0.243421004199512306e-15,  0.681636400857998370e+00, -0.116329451639152745e-07, -0.204534066240976590e-07,  0.786933928164528784e+00,  0.119944146398649087e-07, -0.263577050540936305e-07,},
        {-0.990199482172444689e+00,  0.152341569771065697e-01, -0.701841766337032470e-02,  0.179788946949924561e+00, -0.233989588879152610e-15,  0.207578311829213018e-16,  0.949647758660348072e-16,  0.567558635903070457e-15,  0.314411583696667063e-15, -0.687029374634380357e-16, -0.212031888559672388e-16,  0.279944490004562475e-15, -0.163400005738381198e-07,  0.353468545473865874e-09,  0.103884026836965404e+01, -0.250882516922309486e-01,  0.134458675172929453e-03,  0.873963099696128259e-02, -0.966320354559544681e+00, -0.377222290912238370e-01, -0.326018978121744822e-15,  0.843728020754372367e-15, -0.132267607252230686e-15, -0.734048610829950915e-15, -0.751304153031882358e-16, -0.487890909108235626e-16,  0.343010414762997293e-15,  0.138959353759382230e-15, -0.103905467062790668e+01, -0.135615266706970647e-01, -0.169590116166031761e-07, -0.256830894290934587e-07,  0.152341569771065784e-01,  0.990199482172444689e+00,  0.179788946949925421e+00,  0.701841766337015470e-02,  0.925795602655423746e-16, -0.572779785387826849e-15,  0.340107980558053812e-16, -0.177170273199077789e-15, -0.175398931199396867e-16,  0.231300424132522207e-15, -0.667651231209041343e-15, -0.247306652475209218e-14, -0.259393916462599970e-07, -0.113301033901972209e-07,  0.250882516922304975e-01,  0.103884026836965426e+01,  0.873963099696043778e-02, -0.134458675173031910e-03, -0.377222290912238717e-01,  0.966320354559543682e+00,  0.129879016223473548e-15,  0.707821954409186915e-16, -0.681542827812771132e-15,  0.299291668948013705e-14, -0.310392659834616765e-17,  0.308953977835893369e-15,  0.831324057887873072e-15,  0.384881116007614414e-16, -0.135615266706973249e-01,  0.103905467062790557e+01, -0.301172768851852205e-09,  0.110010898025990421e-07,},
        {0.990308524093064690e+00,  0.401511419472870904e-02, -0.876276209438335975e-03,  0.179923749882735129e+00,  0.212285380418803112e-15,  0.758131383311979808e-16,  0.108530779776711546e-15,  0.105663288461277722e-15, -0.354379265034145431e-04,  0.874059341530878525e-02, -0.967044886406685289e+00, -0.470976415269896542e-02, -0.799664458209222683e-15,  0.226873510527350296e-15,  0.608427540144322061e-15, -0.120032072242435910e-14, -0.105449939920820851e-15, -0.312822988282106089e-16,  0.704405117989277500e-15, -0.194820176245873613e-15, -0.522633204817708856e-07, -0.166170814971126669e-07,  0.103184738386828623e+01,  0.122920705999757768e+00,  0.741320337851100416e-16,  0.528803668193611943e-16, -0.878424739404918159e-15,  0.303672217006303538e-15,  0.103787698209412182e+01, -0.512824882245623281e-01,  0.544514558402038749e-07, -0.227370987255144096e-07, -0.401511419472858934e-02,  0.990308524093064912e+00,  0.179923749882736156e+00,  0.876276209437369192e-03,  0.136857244919773558e-15, -0.456789603422932989e-15, -0.285229229086461692e-15,  0.146373680011976021e-15, -0.874059341530802544e-02, -0.354379265025537611e-04, -0.470976415269894633e-02,  0.967044886406685511e+00, -0.263596278577681681e-15,  0.321998986751637183e-15,  0.141150119201595346e-15, -0.259934655129369555e-15,  0.216315125807477546e-16, -0.230736134194642716e-15,  0.112483865419888315e-14,  0.129789075484486497e-15, -0.290653236772091143e-07, -0.230162317031099196e-09,  0.122920705999756172e+00, -0.103184738386828689e+01, -0.139556037869217773e-16,  0.309356721444783277e-15,  0.255177059549568138e-15, -0.240246612430901235e-15,  0.512824882245613081e-01,  0.103787698209412227e+01,  0.192383012595417774e-07,  0.615753083290430941e-09, },
        {0.101091264553157378e+01,  0.100236223444002369e-01,  0.133296690489577009e-15, -0.109882897423517254e-15, -0.271302438501417007e-15, -0.219579483226183064e-17,  0.172066298880110359e-15,  0.575938237021641693e-16, -0.319164520949966288e-15,  0.389763973534249235e-17,  0.601991530296333788e+00,  0.124354827139159255e+00, -0.110060265065559971e-15, -0.175392116594888844e-15,  0.772935901495790945e+00,  0.101291464371909765e+00, -0.132554745679537837e-15, -0.329815299425215273e-16, -0.124354827139159463e+00,  0.601991530296332122e+00, -0.546974600474473360e-15, -0.330695975699658993e-15,  0.101291464371910264e+00, -0.772935901495793054e+00,  0.761774240795905435e-16,  0.517330410143193204e-17,  0.306716384950991664e-15, -0.317151603203604473e-16, -0.104735153285374372e+01,  0.125148154709254993e-15, -0.570599864028916927e-15,  0.566352284681960181e-15,  0.100236223444002369e-01, -0.101091264553157378e+01, -0.165130114066302124e-15,  0.225056599483962225e-15, -0.106907391459473601e-16, -0.432479411406336835e-15, -0.131707674792485528e-15,  0.190761285119216341e-15, -0.140255776747076362e-16, -0.227180594958384794e-15,  0.124354827139158741e+00, -0.601991530296330013e+00, -0.367350367559626275e-15, -0.237862836389220801e-15,  0.101291464371910098e+00, -0.772935901495794608e+00, -0.688857025875443871e-16,  0.220434248280142009e-15, -0.601991530296328237e+00, -0.124354827139158547e+00, -0.678524170487312186e-15, -0.199075666680435656e-15,  0.772935901495795497e+00,  0.101291464371910139e+00,  0.264464250933522494e-16, -0.312020991859367990e-15, -0.637722128443756577e-16,  0.209410945609469689e-16,  0.112757819512167386e-15,  0.104735153285374327e+01,  0.385836373561019728e-15, -0.421130106562897075e-15, },
        {-0.990315275818137497e+00, -0.165785343658863800e-02, -0.394102509502307838e-01, -0.175556702488141092e+00,  0.159919090516496777e-15, -0.336686094372342024e-16,  0.638014852478640656e-16, -0.306929718338149160e-16,  0.323923418794307479e-15,  0.973210426568806759e-16,  0.228189361039378115e-15, -0.299179882652563415e-15,  0.103582426677735229e+01, -0.822771644819453585e-01,  0.804677025637263489e-02, -0.723392063264858328e-02,  0.110044139634222507e-15,  0.975932482842452862e-16,  0.146857145993868456e-15, -0.236055522939384087e-15, -0.822299937510404255e-01, -0.103318668440309192e+01, -0.372998057443853226e-01, -0.647360127795150869e-01,  0.146324328668810467e-04, -0.874065310643149830e-02,  0.943573104885117142e+00, -0.211820183031309173e+00, -0.305144570597473320e-15,  0.363269550940002135e-15,  0.492094046878242353e-15, -0.660749067466116756e-15, -0.165785343658858509e-02,  0.990315275818137497e+00,  0.175556702488141786e+00, -0.394102509502316303e-01, -0.100539343657872606e-15, -0.378384205540364748e-17,  0.203780342169132846e-15, -0.401757608827353660e-15, -0.490275127457516898e-16,  0.212282735693330979e-15,  0.601490567565597960e-15, -0.293981436989208209e-15,  0.351903485719428028e-02, -0.596784215558673059e-01, -0.138973161121255406e+00,  0.102807153760890957e+01, -0.708593913257273627e-18, -0.171025332574732410e-15, -0.371744570255136517e-15,  0.121559953259577236e-15, -0.106045478849929511e-01, -0.448637151478297330e-01,  0.102910104364301547e+01,  0.136544337705565177e+00, -0.874065310643100737e-02, -0.146324328666355003e-04,  0.211820183031308867e+00,  0.943573104885116365e+00,  0.262411521741187182e-15, -0.289960814805569042e-15,  0.000000000000000000e+00,  0.218332882770432280e-15,},
        {-0.101036513257670557e+01,  0.347440465951327632e-01, -0.382051145281088921e-16, -0.158315196758421178e-15, -0.180077528836175718e-15, -0.333388992876965394e-16, -0.465185560087139592e-16,  0.128513434412210904e-15,  0.377810384684595907e-15, -0.372527402452393659e-16, -0.500553776954217638e-16,  0.103092206906453412e-15, -0.104508697299621245e+01,  0.688364165591181243e-01, -0.676975549601539869e-15, -0.393617284278854964e-14,  0.156700476875449626e-15,  0.196593601348128064e-16, -0.613661447253032066e+00,  0.357428269460637604e-01,  0.535664217495180495e-15, -0.147223445145760541e-15, -0.779159690451184161e+00,  0.244959059902630436e-01, -0.782204010280675860e-16, -0.624245645253091780e-16, -0.357428273299751423e-01, -0.613661453844330573e+00, -0.280476576029852665e-14, -0.162440408949116192e-15,  0.244959058266635207e-01,  0.779159685247448275e+00, -0.347440465951328187e-01, -0.101036513257670579e+01,  0.413479969307056467e-15, -0.125463128870701980e-15,  0.154901180710428086e-15,  0.655326066514134947e-15, -0.186552100489034139e-15, -0.355457982899113210e-16, -0.174419067997601752e-15,  0.150321136698105116e-30,  0.401535983010570853e-16,  0.169709895618099385e-15, -0.688364165591178467e-01, -0.104508697299621334e+01,  0.704887594501439332e-16, -0.469962013441804450e-15, -0.281853475530176047e-16,  0.203365048217887354e-15, -0.357428269460638229e-01, -0.613661447253029402e+00,  0.237430277743661013e-14, -0.230401324600298492e-15, -0.244959059902629395e-01, -0.779159690451185050e+00,  0.398578930015128718e-16, -0.326935312650025221e-15, -0.613661453844328131e+00,  0.357428273299751770e-01, -0.296379955475482316e-15, -0.813344459752158668e-17,  0.779159685247449496e+00, -0.244959058266633646e-01,},
        {-0.101095948336842900e+01, -0.240268349714521136e-02, -0.229207840627946963e-15, -0.133383531120775379e-15,  0.290171825757655398e-15,  0.878258283523218326e-16, -0.558572000737140554e-16,  0.189406260793175445e-15,  0.327896580734073862e-15, -0.103232118464647040e-16, -0.614254919983792669e+00,  0.234267965904009917e-01, -0.407039952790377107e-17, -0.488401753164254496e-15, -0.779365454347476394e+00, -0.167141019506646185e-01, -0.940264896243991271e-16,  0.187823854244476939e-16,  0.349060618645540797e-16,  0.447788159594532260e-15,  0.104716246240442978e+01, -0.199000176427105574e-01,  0.150151604470246934e-15,  0.443328993810943086e-15, -0.794160801634734587e-16,  0.464388843279514543e-16, -0.234267968420267918e-01, -0.614254926581463856e+00,  0.154343959452083530e-15, -0.369573980548268517e-15, -0.167141018390368767e-01,  0.779365449142368827e+00,  0.240268349714525126e-02, -0.101095948336842834e+01,  0.333565728562493970e-15,  0.224931755802606978e-16,  0.992195863399099543e-16,  0.189618107468915265e-15, -0.298622608727988939e-15, -0.691041408527517215e-16,  0.447757066819038911e-16, -0.661302744840426570e-15, -0.234267965904009050e-01, -0.614254919983789449e+00,  0.669105374150053833e-15,  0.492646639993772477e-15,  0.167141019506647122e-01, -0.779365454347479059e+00,  0.334178966097075742e-16,  0.594957471734905521e-16, -0.182030775590589942e-15, -0.273655860896657515e-15, -0.199000176427106372e-01, -0.104716246240442823e+01,  0.101366027310619675e-14, -0.521468084367198679e-15, -0.317544370167873652e-17, -0.318317720939542735e-15, -0.614254926581460525e+00,  0.234267968420266461e-01, -0.722290159642669966e-16,  0.807380567871549949e-15,  0.779365449142370936e+00,  0.167141018390369427e-01,},
        {-0.102637208700827420e+01, -0.122423418140429029e-01,  0.235677004339378160e-15, -0.491833010014734292e-16, -0.158765029794875380e-15,  0.333144469690164116e-16,  0.133563611236909480e-15,  0.204419921617631058e-16,  0.346251593026233643e-15,  0.148036087680791715e-16,  0.159137191670291161e-07,  0.144620190666010882e-08, -0.975228400876956059e+00, -0.318444039310171595e-01,  0.397082527677413352e-01, -0.315532149041125795e-01,  0.146826959396165303e-15,  0.784306466277347479e-16,  0.117275232312321510e-08, -0.247080752971263988e-07, -0.184757568588186753e-01,  0.946281288520041119e+00,  0.213439512845603280e+00, -0.115372416152974500e+00, -0.852194131059891567e-16,  0.733485305142145976e-17,  0.976949638126806574e+00,  0.150417821849355166e-01,  0.139045931415630535e-07,  0.913770118759431861e-08, -0.422573359536139218e-07,  0.125349820825797661e-08, -0.122423418140430052e-01,  0.102637208700827443e+01,  0.318993122291684330e-16, -0.456606947566863748e-15, -0.591511901663085484e-16,  0.619477818060958407e-16, -0.193056463244233339e-15,  0.298138468973190472e-15,  0.672148249416019762e-17,  0.229085191224828799e-15,  0.154152752792816133e-07, -0.507514368667872799e-08,  0.476737795916388346e-01, -0.178700729622267795e+00,  0.304638441633494350e+00, -0.909750257212622149e+00, -0.135595066845229893e-15, -0.216952106952367825e-15,  0.396666537036669488e-07,  0.108347497865953793e-08,  0.311813071938208675e-01, -0.162051093223702092e+00,  0.902617914482577222e+00,  0.335715534894062695e+00,  0.857446398031754366e-17,  0.324286892328574715e-15, -0.150417821849354212e-01,  0.976949638126806241e+00,  0.975376870792955303e-09,  0.230902812800570748e-07,  0.657147393089460471e-08, -0.798885149782741533e-08,},
    }, MAJOR::ROW);
}

static void test_dmat_cplx_Rs_kblacs_para_full_wfc()
{
    if (size_global < 2 || size_global % 2 != 0) return;

    std::vector<Vector3_Order<double>> kfrac_list;
    Matd eigvals, eigvecs;
    double efermi;
    int nk, nb, nao;
    get_BCC_He_k222_light_min_2p_aims_ref(nk, nb, kfrac_list, efermi, eigvals, eigvecs);
    nao = nb;

    const auto set_common_mf_data = [&](MeanField &mf)
    {
        mf.get_efermi() = efermi;
        mf.get_weight()[0].zero_out();
        for (int ik = 0; ik != nk; ++ik)
        {
            mf.get_weight()[0](ik, 0) = mf.get_weight()[0](ik, 1) = 2.0 / nk;
            for (int ib = 0; ib != nb; ++ib)
            {
                mf.get_eigenvals()[0](ik, ib) = eigvals(ik, ib);
            }
        }
    };
    const auto set_wfc = [&](MeanField &mf, int ik)
    {
        auto &eig = mf.get_eigenvectors()[0][0][ik];
        eig.create(nb, nao);
        for (int iao = 0; iao != nao; ++iao)
        {
            for (int ib = 0; ib != nb; ++ib)
            {
                eig.c[ib * nb + iao] = eigvecs(ik, iao * nb + ib);
            }
        }
    };

    MeanField mf_ref(1, nk, nb, nao);
    set_common_mf_data(mf_ref);
    for (int ik = 0; ik != nk; ++ik) set_wfc(mf_ref, ik);

    KPointBlacsProcessShape shape(2, size_global / 2, false);
    KPointBlacsParallelContext context(shape, mpi_comm_global_h.comm, nk);
    const auto desc_wfc = make_permanent_wfc_desc(context, nao, nb);
    const auto desc_wfc_full = context.create_array_desc(nao, nb, nao, nb);
    const auto desc_dm = context.create_array_desc(nao, nb);

    MeanField mf(1, nk, nb, nao);
    set_common_mf_data(mf);
    for (int ik = 0; ik != nk; ++ik)
    {
        if (context.kpoint_blacs_root_global_rank(ik) != myid_global) continue;
        set_wfc(mf, ik);
    }
    redistribute_meanfield_eigvecs_kblacs(
        mf, context, desc_wfc_full, desc_wfc, "test density matrix");

    const std::vector<Vector3_Order<int>> Rs =
        context.comm_kpoint_h.myid == 0
            ? std::vector<Vector3_Order<int>>{{-1, 0, 0}, {0, 0, 0}}
            : std::vector<Vector3_Order<int>>{{1, 0, 0}};
    const auto dm_Rs = get_dmat_cplx_Rs_kblacs_para(
        0, mf, kfrac_list, Rs, context, desc_wfc, desc_dm);
    assert(dm_Rs.size() == Rs.size());

    for (const auto &R: Rs)
    {
        const auto dm_ref = mf_ref.get_dmat_cplx_R(0, 0, 0, kfrac_list, R);
        const auto &rmat = dm_Rs.at(R);
        assert(rmat.nr() == desc_dm.m_loc());
        assert(rmat.nc() == desc_dm.n_loc());
        for (int jloc = 0; jloc != desc_dm.n_loc(); ++jloc)
        {
            const int jglob = desc_dm.indx_l2g_c(jloc);
            for (int iloc = 0; iloc != desc_dm.m_loc(); ++iloc)
            {
                const int iglob = desc_dm.indx_l2g_r(iloc);
                assert(fequal(rmat(iloc, jloc), dm_ref(iglob, jglob)));
            }
        }
    }
}

static void test_gf_cplx_imagtimes_Rs_kblacs_para_full_wfc(bool masked = false)
{
    if (size_global < 2 || size_global % 2 != 0) return;

    std::vector<Vector3_Order<double>> kfrac_list;
    Matd eigvals, eigvecs;
    double efermi;
    int nk, nb, nao;
    get_BCC_He_k222_light_min_2p_aims_ref(nk, nb, kfrac_list, efermi, eigvals, eigvecs);
    nao = nb;

    const auto set_common_mf_data = [&](MeanField &mf)
    {
        mf.get_efermi() = efermi;
        mf.get_weight()[0].zero_out();
        for (int ik = 0; ik != nk; ++ik)
        {
            mf.get_weight()[0](ik, 0) = mf.get_weight()[0](ik, 1) = 2.0 / nk;
            for (int ib = 0; ib != nb; ++ib)
            {
                mf.get_eigenvals()[0](ik, ib) = eigvals(ik, ib);
            }
        }
    };
    const auto set_wfc = [&](MeanField &mf, int ik)
    {
        auto &eig = mf.get_eigenvectors()[0][0][ik];
        eig.create(nb, nao);
        for (int iao = 0; iao != nao; ++iao)
        {
            for (int ib = 0; ib != nb; ++ib)
            {
                eig.c[ib * nb + iao] = eigvecs(ik, iao * nb + ib);
            }
        }
    };

    MeanField mf_ref(1, nk, nb, nao);
    set_common_mf_data(mf_ref);
    for (int ik = 0; ik != nk; ++ik) set_wfc(mf_ref, ik);

    KPointBlacsProcessShape shape(2, size_global / 2, false);
    KPointBlacsParallelContext context(shape, mpi_comm_global_h.comm, nk);
    const auto desc_wfc = make_permanent_wfc_desc(context, nao, nb);
    const auto desc_wfc_full = context.create_array_desc(nao, nb, nao, nb);
    const auto desc_gf = context.create_array_desc(nao, nao);

    MeanField mf(1, nk, nb, nao);
    set_common_mf_data(mf);
    for (int ik = 0; ik != nk; ++ik)
    {
        if (context.kpoint_blacs_root_global_rank(ik) != myid_global) continue;
        set_wfc(mf, ik);
    }
    redistribute_meanfield_eigvecs_kblacs(
        mf, context, desc_wfc_full, desc_wfc, "test Green function");

    const std::vector<Vector3_Order<int>> Rs =
        context.comm_kpoint_h.myid == 0
            ? std::vector<Vector3_Order<int>>{{-1, 0, 0}, {0, 0, 0}}
            : std::vector<Vector3_Order<int>>{{1, 0, 0}};
    std::vector<double> imagtimes{1.0, -0.5};
    std::vector<bool> mask(nb, true);
    if (masked)
    {
        std::fill(mask.begin() + 3, mask.end(), false);
        mask[0] = false;
    }
    const auto gf_ref = mf_ref.get_gf_cplx_imagtimes_Rs(0, 0, 0, kfrac_list, imagtimes, Rs, mask);
    const auto gf_Rs = get_gf_cplx_imagtimes_Rs_kblacs_para(
        0, mf, kfrac_list, imagtimes, Rs, context, desc_wfc, desc_gf, mask);
    assert(gf_Rs.size() == imagtimes.size());

    for (const auto tau: imagtimes)
    {
        const auto &gf_tau = gf_Rs.at(tau);
        assert(gf_tau.size() == Rs.size());
        for (const auto &R: Rs)
        {
            const auto &rmat = gf_tau.at(R);
            assert(rmat.nr() == desc_gf.m_loc());
            assert(rmat.nc() == desc_gf.n_loc());
            for (int jloc = 0; jloc != desc_gf.n_loc(); ++jloc)
            {
                const int jglob = desc_gf.indx_l2g_c(jloc);
                for (int iloc = 0; iloc != desc_gf.m_loc(); ++iloc)
                {
                    const int iglob = desc_gf.indx_l2g_r(iloc);
                    assert(fequal(rmat(iloc, jloc), gf_ref.at(tau).at(R)(iglob, jglob),
                                  cplxdb{1e-12, 0.0}));
                }
            }
        }
    }
}

static void test_dm_gf_kblacs_para_redistributed_full_wfc()
{
    if (size_global < 2) return;

    const int nk = 2;
    const int nb = 144;
    const int nao = 72;
    const int nocc = 6;
    const double efermi = 0.0;
    std::vector<Vector3_Order<double>> kfrac_list{{0.0, 0.0, 0.0}, {0.5, 0.0, 0.0}};
    const std::vector<Vector3_Order<int>> Rs{{0, 0, 0}, {1, 0, 0}};
    const std::vector<double> imagtimes{0.75, -0.5};

    const auto set_common_mf_data = [&](MeanField &mf)
    {
        mf.get_efermi() = efermi;
        mf.get_weight()[0].zero_out();
        for (int ik = 0; ik != nk; ++ik)
        {
            for (int ib = 0; ib != nb; ++ib)
            {
                mf.get_weight()[0](ik, ib) = ib < nocc ? 2.0 / nk : 0.0;
                mf.get_eigenvals()[0](ik, ib) = ib < nocc
                    ? -0.2 - 0.001 * ib
                    : 0.3 + 0.001 * ib;
            }
        }
    };
    const auto set_wfc = [&](MeanField &mf, int ik)
    {
        auto &eig = mf.get_eigenvectors()[0][0][ik];
        eig.create(nb, nao);
        for (int ib = 0; ib != nb; ++ib)
        {
            for (int iao = 0; iao != nao; ++iao)
            {
                const double arg = 0.013 * (ib + 1) * (iao + 2) + 0.17 * (ik + 1);
                eig(ib, iao) = cplxdb{std::sin(arg), 0.25 * std::cos(0.7 * arg)};
            }
        }
    };

    MeanField mf_ref(1, nk, nb, nao);
    set_common_mf_data(mf_ref);
    for (int ik = 0; ik != nk; ++ik) set_wfc(mf_ref, ik);

    KPointBlacsProcessShape shape(1, size_global, false);
    KPointBlacsParallelContext context(shape, mpi_comm_global_h.comm, nk);
    const auto desc_wfc = make_permanent_wfc_desc(context, nao, nb);
    const auto desc_wfc_full = context.create_array_desc(nao, nb, nao, nb);
    const auto desc_dm = context.create_array_desc(nao, nao);

    MeanField mf(1, nk, nb, nao);
    set_common_mf_data(mf);
    for (int ik = 0; ik != nk; ++ik)
    {
        if (context.kpoint_blacs_root_global_rank(ik) != myid_global) continue;
        set_wfc(mf, ik);
    }

    redistribute_meanfield_eigvecs_kblacs(
        mf, context, desc_wfc_full, desc_wfc, "test rectangular WFC");

    const auto dm_Rs = get_dmat_cplx_Rs_kblacs_para(
        0, mf, kfrac_list, Rs, context, desc_wfc, desc_dm);
#if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
    // Match a local LibRI contraction between uses of the same BLACS handle.
    // MPI_COMM_SELF selects device 0, which differs from nonzero world ranks.
    ddla::DdlaHandle_t local_context = nullptr;
    ddla::DDLA_CHECK(ddla::ddlaInit(local_context, ddla::DdlaBackend::GPU));
    ddla::DDLA_CHECK(ddla::ddlaSet(local_context, MPI_COMM_SELF, 1, 1, 'R'));
    ddla::DDLA_CHECK(ddla::ddlaDestroy(local_context));
#endif
    const auto gf_Rs = get_gf_cplx_imagtimes_Rs_kblacs_para(
        0, mf, kfrac_list, imagtimes, Rs, context, desc_wfc, desc_dm);

    for (const auto &R: Rs)
    {
        const auto dm_ref = mf_ref.get_dmat_cplx_R(0, 0, 0, kfrac_list, R);
        const auto &rmat = dm_Rs.at(R);
        assert(rmat.nr() == desc_dm.m_loc());
        assert(rmat.nc() == desc_dm.n_loc());
        for (int jloc = 0; jloc != desc_dm.n_loc(); ++jloc)
        {
            const int jglob = desc_dm.indx_l2g_c(jloc);
            for (int iloc = 0; iloc != desc_dm.m_loc(); ++iloc)
            {
                const int iglob = desc_dm.indx_l2g_r(iloc);
                assert(fequal(rmat(iloc, jloc), dm_ref(iglob, jglob), cplxdb{1e-10, 0.0}));
            }
        }
    }

    const auto gf_ref = mf_ref.get_gf_cplx_imagtimes_Rs(0, 0, 0, kfrac_list, imagtimes, Rs);
    for (const auto tau: imagtimes)
    {
        for (const auto &R: Rs)
        {
            const auto &rmat = gf_Rs.at(tau).at(R);
            for (int jloc = 0; jloc != desc_dm.n_loc(); ++jloc)
            {
                const int jglob = desc_dm.indx_l2g_c(jloc);
                for (int iloc = 0; iloc != desc_dm.m_loc(); ++iloc)
                {
                    const int iglob = desc_dm.indx_l2g_r(iloc);
                    assert(fequal(rmat(iloc, jloc), gf_ref.at(tau).at(R)(iglob, jglob),
                                  cplxdb{1e-10, 0.0}));
                }
            }
        }
    }
}

static SymmetryContext build_two_member_identity_kstar_context()
{
    SymmetryContext ctx;
    ctx.set_available();
    ctx.atom_to_type[0] = 0;
    ctx.input_coord_frac[0] = {0.0, 0.0, 0.0};

    SymmetryOperation identity_operation;
    identity_operation.rotation.Identity();
    identity_operation.translation = {0.0, 0.0, 0.0};
    ctx.rspace_operations.push_back(identity_operation);
    ctx.rsh_rotations.emplace_back();
    ctx.rsh_rotations.back()[0] = ComplexMatrix(1, 1);
    ctx.rsh_rotations.back()[0](0, 0) = {1.0, 0.0};

    SymmetryKAtomRotation atom_rotation;
    atom_rotation.atom_from = 0;
    atom_rotation.atom_to = 0;
    atom_rotation.atom_type = 0;
    atom_rotation.lmax = 0;
    atom_rotation.bloch_rsh_rotations[0] = ComplexMatrix(1, 1);
    atom_rotation.bloch_rsh_rotations[0](0, 0) = {1.0, 0.0};

    SymmetryKStar star;
    star.star_index = 0;
    star.k_ibz = {0.0, 0.0, 0.0};
    star.members.resize(2);
    star.members[0].spatial_isym = 0;
    star.members[0].k_bz = {0.0, 0.0, 0.0};
    star.members[0].atom_rotations.push_back(atom_rotation);
    star.members[1].spatial_isym = 0;
    star.members[1].k_bz = {0.5, 0.0, 0.0};
    star.members[1].atom_rotations.push_back(atom_rotation);
    ctx.kstars.push_back(star);

    return ctx;
}

static void set_one_ao_reduced_kstar_meanfield(MeanField &mf)
{
    mf.get_efermi() = 0.0;
    mf.get_eigenvals()[0](0, 0) = -1.0;
    mf.get_weight()[0](0, 0) = 2.0;
    mf.get_eigenvectors()[0][0][0].create(1, 1);
    mf.get_eigenvectors()[0][0][0](0, 0) = {1.0, 0.0};
}

static SymmetryKStarMember make_one_ao_atom_rotation_member(
    const atom_t atom_from, const atom_t atom_to, const std::complex<double> &rotation)
{
    SymmetryKStarMember member;
    member.spatial_isym = 0;
    member.k_bz = {0.0, 0.0, 0.0};

    SymmetryKAtomRotation atom_rotation;
    atom_rotation.atom_from = atom_from;
    atom_rotation.atom_to = atom_to;
    atom_rotation.atom_type = 0;
    atom_rotation.lmax = 0;
    atom_rotation.bloch_rsh_rotations[0] = ComplexMatrix(1, 1);
    atom_rotation.bloch_rsh_rotations[0](0, 0) = rotation;
    member.atom_rotations.push_back(atom_rotation);
    return member;
}

static SymmetryContext build_two_atom_swap_kstar_context()
{
    SymmetryContext ctx;
    ctx.set_available();
    ctx.atom_to_type = {{0, 0}, {1, 0}};
    ctx.input_coord_frac = {{0, {0.0, 0.0, 0.0}}, {1, {0.5, 0.0, 0.0}}};

    SymmetryOperation identity_operation;
    identity_operation.rotation.Identity();
    identity_operation.translation = {0.0, 0.0, 0.0};
    ctx.rspace_operations.push_back(identity_operation);
    ctx.rsh_rotations.emplace_back();
    ctx.rsh_rotations.back()[0] = ComplexMatrix(1, 1);
    ctx.rsh_rotations.back()[0](0, 0) = {1.0, 0.0};

    SymmetryKStar star;
    star.star_index = 0;
    star.k_ibz = {0.0, 0.0, 0.0};

    auto identity_atom_0 = make_one_ao_atom_rotation_member(0, 0, {1.0, 0.0}).atom_rotations[0];
    auto identity_atom_1 = make_one_ao_atom_rotation_member(1, 1, {1.0, 0.0}).atom_rotations[0];
    SymmetryKStarMember identity_member;
    identity_member.spatial_isym = 0;
    identity_member.k_bz = {0.0, 0.0, 0.0};
    identity_member.atom_rotations = {identity_atom_0, identity_atom_1};

    auto swap_atom_0 = make_one_ao_atom_rotation_member(0, 1, {0.0, 1.0}).atom_rotations[0];
    auto swap_atom_1 = make_one_ao_atom_rotation_member(1, 0, {-1.0, 0.0}).atom_rotations[0];
    SymmetryKStarMember swap_member;
    swap_member.spatial_isym = 0;
    swap_member.k_bz = {0.5, 0.0, 0.0};
    swap_member.atom_rotations = {swap_atom_0, swap_atom_1};

    star.members = {identity_member, swap_member};
    ctx.kstars.push_back(star);
    return ctx;
}

static ComplexMatrix restore_test_wfc_to_member(
    const SymmetryContext &ctx, const SymmetryKStarMember &member,
    const std::vector<SpeciesBasisLayout> &wfc_layouts,
    const std::map<atom_t, size_t> &atom_nw,
    const Vector3_Order<double> &k_ibz, const ComplexMatrix &wfc_ibz)
{
    const auto rotation = build_symmetry_kspace_rotation_matrix(
        ctx, wfc_layouts, member, atom_nw, k_ibz, member.time_reversal, &member.k_bz);
    if (member.time_reversal)
    {
        return conj(wfc_ibz) * rotation;
    }
    return wfc_ibz * conj(rotation);
}

static void set_two_atom_reduced_kstar_meanfield(MeanField &mf)
{
    mf.get_efermi() = 0.0;
    mf.get_eigenvals()[0](0, 0) = -0.4;
    mf.get_eigenvals()[0](0, 1) = 0.7;
    mf.get_weight()[0](0, 0) = 2.0;
    mf.get_weight()[0](0, 1) = 0.0;

    auto &wfc = mf.get_eigenvectors()[0][0][0];
    wfc.create(2, 2);
    wfc(0, 0) = {0.7, -0.2};
    wfc(0, 1) = {-0.4, 0.6};
    wfc(1, 0) = {0.3, 0.5};
    wfc(1, 1) = {-0.8, -0.1};
}

static MeanField build_two_atom_full_bz_meanfield_from_kstar(
    const SymmetryContext &ctx, const std::vector<SpeciesBasisLayout> &wfc_layouts,
    const std::map<atom_t, size_t> &atom_nw)
{
    MeanField mf_ibz(1, 1, 2, 2);
    set_two_atom_reduced_kstar_meanfield(mf_ibz);

    MeanField mf_full(1, 2, 2, 2);
    mf_full.get_efermi() = mf_ibz.get_efermi();
    const auto &star = ctx.kstars.front();
    for (int ik = 0; ik != 2; ++ik)
    {
        mf_full.get_eigenvals()[0](ik, 0) = mf_ibz.get_eigenvals()[0](0, 0);
        mf_full.get_eigenvals()[0](ik, 1) = mf_ibz.get_eigenvals()[0](0, 1);
        mf_full.get_weight()[0](ik, 0) = 1.0;
        mf_full.get_weight()[0](ik, 1) = 0.0;
        auto &wfc_full = mf_full.get_eigenvectors()[0][0][ik];
        wfc_full = restore_test_wfc_to_member(
            ctx, star.members[static_cast<std::size_t>(ik)], wfc_layouts, atom_nw,
            star.k_ibz, mf_ibz.get_eigenvectors()[0][0][0]);
    }
    return mf_full;
}

static void test_dmat_gf_kblacs_reduced_kstar_matches_full_bz_fourier(bool masked = false, bool exclude_lowest = false)
{
    if (size_global < 2) return;

    const auto ctx = build_two_atom_swap_kstar_context();
    AtomicBasis atbasis_wfc(std::vector<size_t>{1, 1});
    atbasis_wfc.set_l_shells({{0}, {0}});
    const auto wfc_layouts = atbasis_wfc.build_species_basis_layouts(ctx.atom_to_type);
    const std::map<atom_t, size_t> atom_nw{{0, 1}, {1, 1}};
    PeriodicBoundaryData pbc;

    const std::vector<Vector3_Order<double>> kfrac_ibz{{0.0, 0.0, 0.0}};
    const std::vector<Vector3_Order<double>> kfrac_full{{0.0, 0.0, 0.0}, {0.5, 0.0, 0.0}};
    const std::vector<Vector3_Order<int>> Rs{{0, 0, 0}, {1, 0, 0}};
    const std::vector<double> taus{-0.25, 0.25};

    MeanField mf_full = build_two_atom_full_bz_meanfield_from_kstar(ctx, wfc_layouts, atom_nw);
    const auto expected_dmat_R0 = mf_full.get_dmat_cplx_R(0, 0, 0, kfrac_full, Rs.front());
    const auto expected_dmat_R1 = mf_full.get_dmat_cplx_R(0, 0, 0, kfrac_full, Rs.back());
    const std::vector<bool> mask{!exclude_lowest, !masked};
    const auto expected_gf =
        mf_full.get_gf_cplx_imagtimes_Rs(0, 0, 0, kfrac_full, taus, Rs, mask);

    KPointBlacsProcessShape shape(1, size_global, false);
    KPointBlacsParallelContext context(shape, mpi_comm_global_h.comm, 1);
    const auto desc_wfc_full = context.create_array_desc(2, 2, 2, 2);
    const auto desc_dm = context.create_array_desc(2, 2);

    MeanField mf(1, 1, 2, 2);
    mf.get_efermi() = 0.0;
    mf.get_eigenvals()[0](0, 0) = -0.4;
    mf.get_eigenvals()[0](0, 1) = 0.7;
    mf.get_weight()[0](0, 0) = 2.0;
    mf.get_weight()[0](0, 1) = 0.0;
    if (context.kpoint_blacs_root_global_rank(0) == myid_global)
    {
        set_two_atom_reduced_kstar_meanfield(mf);
    }

    const auto actual_dmat = get_symmetry_restored_dmat_cplx_Rs_kblacs_para(
        0, 0, 0, mf, kfrac_ibz, Rs, context, desc_wfc_full, desc_dm, ctx, pbc,
        atbasis_wfc);
    const auto actual_gf = get_symmetry_restored_gf_cplx_imagtimes_Rs_kblacs_para(
        0, 0, 0, mf, kfrac_ibz, taus, Rs, context, desc_wfc_full, desc_dm, ctx, pbc,
        atbasis_wfc, mask);

    const std::map<Vector3_Order<int>, ComplexMatrix> expected_dmat{
        {Rs.front(), expected_dmat_R0}, {Rs.back(), expected_dmat_R1}};
    for (const auto &R : Rs)
    {
        const auto &rmat = actual_dmat.at(R);
        for (int jloc = 0; jloc != desc_dm.n_loc(); ++jloc)
        {
            const int jglob = desc_dm.indx_l2g_c(jloc);
            for (int iloc = 0; iloc != desc_dm.m_loc(); ++iloc)
            {
                const int iglob = desc_dm.indx_l2g_r(iloc);
                assert(fequal(rmat(iloc, jloc), expected_dmat.at(R)(iglob, jglob),
                              cplxdb{1e-12, 0.0}));
            }
        }
    }
    for (const auto tau : taus)
    {
        for (const auto &R : Rs)
        {
            const auto &rmat = actual_gf.at(tau).at(R);
            for (int jloc = 0; jloc != desc_dm.n_loc(); ++jloc)
            {
                const int jglob = desc_dm.indx_l2g_c(jloc);
                for (int iloc = 0; iloc != desc_dm.m_loc(); ++iloc)
                {
                    const int iglob = desc_dm.indx_l2g_r(iloc);
                    assert(fequal(rmat(iloc, jloc), expected_gf.at(tau).at(R)(iglob, jglob),
                                  cplxdb{1e-12, 0.0}));
                }
            }
        }
    }
}

static void test_dmat_kblacs_reduced_kstar_matches_symmetry_restore()
{
    if (size_global < 2) return;

    const auto ctx = build_two_member_identity_kstar_context();
    const std::vector<SpeciesBasisLayout> wfc_layouts{{"X", {0}}};
    const std::vector<Vector3_Order<double>> kfrac_list{{0.0, 0.0, 0.0}};
    const std::vector<Vector3_Order<int>> Rs{{1, 0, 0}};
    const std::map<atom_t, size_t> atom_nw{{0, 1}};
    AtomicBasis atbasis_wfc({1});
    atbasis_wfc.set_l_shells({{0}});
    PeriodicBoundaryData pbc;

    MeanField mf_ref(1, 1, 1, 1);
    set_one_ao_reduced_kstar_meanfield(mf_ref);
    const auto expected = get_symmetry_restored_dmat_cplx_R(
        ctx, wfc_layouts, mf_ref, 0, 0, 0, kfrac_list, Rs.front(), atom_nw);
    const std::vector<double> taus{-0.25, 0.25};
    const auto expected_gf = get_symmetry_restored_gf_cplx_imagtimes_Rs(
        ctx, wfc_layouts, mf_ref, 0, 0, 0, kfrac_list, taus, Rs, atom_nw);

    KPointBlacsProcessShape shape(1, size_global, false);
    KPointBlacsParallelContext context(shape, mpi_comm_global_h.comm, 1);
    const auto desc_wfc_full = context.create_array_desc(1, 1, 1, 1);
    const auto desc_dm = context.create_array_desc(1, 1);
    const int has_empty_local_wfc =
        desc_wfc_full.m_loc() * desc_wfc_full.n_loc() == 0 ? 1 : 0;
    int empty_local_wfc_ranks = 0;
    MPI_Allreduce(&has_empty_local_wfc, &empty_local_wfc_ranks, 1, MPI_INT, MPI_SUM,
                  mpi_comm_global_h.comm);
    if (empty_local_wfc_ranks == 0)
        throw std::runtime_error("kBLACS fixture did not create an empty local WFC block");

    MeanField mf(1, 1, 1, 1);
    mf.get_efermi() = 0.0;
    mf.get_eigenvals()[0](0, 0) = -1.0;
    mf.get_weight()[0](0, 0) = 2.0;
    // Keep a replicated source WFC on every rank. Ranks with no local
    // descriptor entries must ignore this nonlocal storage.
    mf.get_eigenvectors()[0][0][0].create(1, 1);
    mf.get_eigenvectors()[0][0][0](0, 0) = {1.0, 0.0};

    const auto actual = get_symmetry_restored_dmat_cplx_Rs_kblacs_para(
        0, 0, 0, mf, kfrac_list, Rs, context, desc_wfc_full, desc_dm, ctx, pbc,
        atbasis_wfc);
    const auto actual_gf = get_symmetry_restored_gf_cplx_imagtimes_Rs_kblacs_para(
        0, 0, 0, mf, kfrac_list, taus, Rs, context, desc_wfc_full, desc_dm, ctx, pbc,
        atbasis_wfc);
    if (desc_dm.m_loc() > 0 && desc_dm.n_loc() > 0)
    {
        const auto diff = std::abs(actual.at(Rs.front())(0, 0) - expected(0, 0));
        if (diff > 1e-12)
        {
            throw std::runtime_error(
                "kBLACS density matrix for reduced k-star did not match symmetry restore");
        }
        for (const auto tau : taus)
        {
            const auto gf_diff = std::abs(
                actual_gf.at(tau).at(Rs.front())(0, 0)
                - expected_gf.at(tau).at(Rs.front())(0, 0));
            if (gf_diff > 1e-12)
            {
                throw std::runtime_error(
                    "kBLACS Green's function for reduced k-star did not match symmetry restore");
            }
        }
    }
}

// a=3A, k222, light, minimal + 2p in tier1, FHI-aims
static void test_dmat_cplx_Rs_kpara()
{
    using namespace librpa_int;
    std::vector<Vector3_Order<double>> kfrac_list;
    Matd eigvals, eigvecs;
    double efermi;
    int nk, nb;

    get_BCC_He_k222_light_min_2p_aims_ref(nk, nb, kfrac_list, efermi, eigvals, eigvecs);

    MeanField mf(1, nk, nb, nb);
    mf.get_efermi() = efermi;
    mf.get_weight()[0].zero_out();
    for (int ik = 0; ik < nk; ik++)
    {
        mf.get_weight()[0](ik, 0) = mf.get_weight()[0](ik, 1) = 2.0 / nk;
    }
    assert(eigvals.nc() == nb && eigvals.nr() == nk);
    for (int ik = 0; ik < nk; ik++)
    {
        for (int ib = 0; ib < nb; ib++)
            mf.get_eigenvals()[0](ik, ib) = eigvals(ik, ib);
    }
    assert(fequal_array(nb * nk, eigvals.ptr(), mf.get_eigenvals()[0].c));
    assert(eigvecs.nc() == nb * nb && eigvecs.nr() == nk);
    for (int ik = 0; ik < nk; ik++)
    {
        if (ik % size_global != myid_global) continue;
        auto &eig = mf.get_eigenvectors()[0][0][ik];
        eig.create(nb, nb);
        for (int iao = 0; iao < nb; iao++)
        {
            for (int ib = 0; ib < nb; ib++)
            {
                eig.c[ib * nb + iao] = eigvecs(ik, iao * nb + ib);
            }
        }
    }
    mpi_comm_global_h.barrier();
    Vector3_Order<int> R{-1, 0, 0};
    Matd dmat_ref_m100({
        {-1.656665871802132e-02, 2.885814663513059e-17, 7.551473425114088e-18,-2.720649452993437e-19,-1.936499264960836e-02,-1.082003306516625e-03,-1.082003306516585e-03,-1.082003318794373e-03,},
        { 2.885814663513059e-17, 9.549903637296769e-06,-1.034814855959821e-19, 1.631860483245147e-19, 1.082003306516695e-03, 9.371034341063192e-19, 9.244270218276978e-20, 6.921477700570555e-19,},
        { 7.551473425114088e-18,-1.034814855959821e-19, 9.549903637296966e-06, 5.186474099402387e-20, 1.082003306516712e-03, 3.677406994863247e-19,-1.261211977533549e-19, 4.578765111328676e-19,},
        {-2.720649452993437e-19, 1.631860483245147e-19, 5.186474099402387e-20,-9.549903854382776e-06, 1.082003318794432e-03, 1.507969010062873e-19,-2.681143407333009e-19, 2.673805149556281e-19,},
        {-1.936499264960836e-02, 1.082003306516695e-03, 1.082003306516712e-03, 1.082003318794432e-03,-1.656665871802127e-02,-2.866825624231874e-18,-2.728744541224779e-17,-8.266331956538578e-18,},
        {-1.082003306516625e-03, 9.371034341063192e-19, 3.677406994863247e-19, 1.507969010062873e-19,-2.866825624231874e-18, 9.549903637295102e-06,-4.566099855643121e-20,-3.930112136175861e-20,},
        {-1.082003306516585e-03, 9.244270218276978e-20,-1.261211977533549e-19,-2.681143407333009e-19,-2.728744541224779e-17,-4.566099855643121e-20, 9.549903637295123e-06,-9.670600743374751e-21,},
        {-1.082003318794373e-03, 6.921477700570555e-19, 4.578765111328676e-19, 2.673805149556281e-19,-8.266331956538578e-18,-3.930112136175861e-20,-9.670600743374751e-21,-9.549903854381701e-06,},
    }, MAJOR::ROW);
    assert(dmat_ref_m100.nc() == nb && dmat_ref_m100.nr() == nb);

    if (size_global == 1)
    {
        auto dmat_serial = mf.get_dmat_cplx_R(0, 0, 0, kfrac_list, R).real();
        // print_matrix("dmat_serial[-1,0,0]", dmat_serial, ofs_myid, true);
        assert(fequal_array(nb * nb, dmat_ref_m100.ptr(), dmat_serial.c, false));
    }
    auto dmat_para = get_dmat_cplx_Rs_kpara(0, mf, kfrac_list, {R}, mpi_comm_global_h).at(R).real();
    // assert(fequal_array(nb * nb, dmat_ref_m100.ptr(), dmat_para.c, myid_global == 0));
    assert(fequal_array(nb * nb, dmat_ref_m100.ptr(), dmat_para.c, false));
}

static SymmetryContext build_gamma_tr_spinor_kstar_context()
{
    SymmetryContext ctx;
    ctx.set_available();
    ctx.atom_to_type[0] = 0;
    ctx.input_coord_frac[0] = {0.0, 0.0, 0.0};

    SymmetryOperation identity_operation;
    identity_operation.rotation.Identity();
    identity_operation.translation = {0.0, 0.0, 0.0};
    ctx.rspace_operations.push_back(identity_operation);
    ctx.rsh_rotations.emplace_back();
    ctx.rsh_rotations.back()[0] = ComplexMatrix(1, 1);
    ctx.rsh_rotations.back()[0](0, 0) = {1.0, 0.0};
    ctx.spin_operations.push_back(SymmetrySpinOperation{
        0, {1.0, 0.0, 0.0, 1.0}, false, SymmetrySpinActionSource::Identity});
    ctx.spin_operations.push_back(SymmetrySpinOperation{
        0, {1.0, 0.0, 0.0, 1.0}, true, SymmetrySpinActionSource::Identity});
    ctx.kspace_actions.push_back(SymmetryGeometricAction{0, {0}});
    ctx.kspace_actions.push_back(SymmetryGeometricAction{1, {1}});

    SymmetryKAtomRotation atom_rotation;
    atom_rotation.atom_from = 0;
    atom_rotation.atom_to = 0;
    atom_rotation.atom_type = 0;
    atom_rotation.lmax = 0;
    atom_rotation.bloch_rsh_rotations[0] = ComplexMatrix(1, 1);
    atom_rotation.bloch_rsh_rotations[0](0, 0) = {1.0, 0.0};

    SymmetryKStar star;
    star.star_index = 0;
    star.k_ibz = {0.0, 0.0, 0.0};
    star.members.resize(2);
    for (int im = 0; im != 2; ++im)
    {
        star.members[im].spatial_isym = 0;
        star.members[im].action_id = static_cast<std::size_t>(im);
        star.members[im].k_bz = {0.0, 0.0, 0.0};
        star.members[im].atom_rotations.push_back(atom_rotation);
    }
    star.members[1].time_reversal = true;
    ctx.kstars.push_back(star);
    return ctx;
}

static void test_gf_kblacs_reduced_kstar_spinor_matches_serial_restore()
{
    if (size_global < 2) return;

    const auto ctx = build_gamma_tr_spinor_kstar_context();
    const std::vector<SpeciesBasisLayout> wfc_layouts{{"X", {0}}};
    const std::vector<Vector3_Order<double>> kfrac_list{{0.0, 0.0, 0.0}};
    const std::vector<Vector3_Order<int>> Rs{{0, 0, 0}, {1, 0, 0}};
    const std::vector<double> taus{-0.25, 0.25};
    const std::map<atom_t, size_t> atom_nw{{0, 1}};
    AtomicBasis atbasis_wfc({1});
    atbasis_wfc.set_l_shells({{0}});
    PeriodicBoundaryData pbc;

    MeanField mf_ref(1, 1, 2, 1, 2);
    mf_ref.get_efermi() = 0.0;
    mf_ref.get_eigenvals()[0](0, 0) = -1.0;
    mf_ref.get_eigenvals()[0](0, 1) = 1.0;
    mf_ref.get_weight()[0](0, 0) = 1.0;
    mf_ref.get_weight()[0](0, 1) = 0.0;
    for (int ispinor = 0; ispinor != 2; ++ispinor)
        mf_ref.get_eigenvectors()[0][ispinor][0].create(2, 1);
    mf_ref.get_eigenvectors()[0][0][0](0, 0) = {0.6, 0.1};
    mf_ref.get_eigenvectors()[0][0][0](1, 0) = {-0.2, 0.7};
    mf_ref.get_eigenvectors()[0][1][0](0, 0) = {0.5, -0.3};
    mf_ref.get_eigenvectors()[0][1][0](1, 0) = {0.1, 0.4};

    const auto expected = get_symmetry_restored_gf_cplx_imagtimes_Rs_spinor(
        ctx, wfc_layouts, mf_ref, 0, kfrac_list, taus, Rs, atom_nw, std::vector<bool>{});
    KPointBlacsProcessShape shape(1, size_global, false);
    KPointBlacsParallelContext context(shape, mpi_comm_global_h.comm, 1);
    const auto desc_wfc = context.create_array_desc(1, 2, 1, 2);
    const auto desc_dm = context.create_array_desc(1, 1);

    MeanField mf(1, 1, 2, 1, 2);
    mf.get_efermi() = 0.0;
    mf.get_eigenvals()[0](0, 0) = -1.0;
    mf.get_eigenvals()[0](0, 1) = 1.0;
    mf.get_weight()[0](0, 0) = 1.0;
    mf.get_weight()[0](0, 1) = 0.0;
    if (context.kpoint_blacs_root_global_rank(0) == myid_global)
    {
        mf.get_eigenvectors()[0][0][0] = mf_ref.get_eigenvectors()[0][0][0];
        mf.get_eigenvectors()[0][1][0] = mf_ref.get_eigenvectors()[0][1][0];
    }

    const auto actual = get_symmetry_restored_gf_cplx_imagtimes_Rs_kblacs_para_spinor(
        0, mf, kfrac_list, taus, Rs, context, desc_wfc, desc_dm, ctx, pbc, atbasis_wfc);
    for (const auto tau : taus)
        for (const auto &R : Rs)
        {
            const auto &exp_blocks = expected.at(tau).at(R);
            const auto &act_blocks = actual.at(tau).at(R);
            const ComplexMatrix *exp_channel[4] = {
                &exp_blocks.b00, &exp_blocks.b01, &exp_blocks.b10, &exp_blocks.b11};
            const Matz *act_channel[4] = {
                &act_blocks.b00, &act_blocks.b01, &act_blocks.b10, &act_blocks.b11};
            for (int s = 0; s != 4; ++s)
                for (int jloc = 0; jloc != desc_dm.n_loc(); ++jloc)
                    for (int iloc = 0; iloc != desc_dm.m_loc(); ++iloc)
                    {
                        const int jglob = desc_dm.indx_l2g_c(jloc);
                        const int iglob = desc_dm.indx_l2g_r(iloc);
                        if (!fequal((*act_channel[s])(iloc, jloc),
                                    (*exp_channel[s])(iglob, jglob), cplxdb{1e-12, 0.0}))
                            throw std::runtime_error(
                                "kBLACS spinor Green's function did not match serial restore");
                    }
        }
}

static void test_dmat_kblacs_reduced_kstar_spinor_matches_serial_restore()
{
    if (size_global < 2) return;

    const auto ctx = build_gamma_tr_spinor_kstar_context();
    const std::vector<SpeciesBasisLayout> wfc_layouts{{"X", {0}}};
    const std::vector<Vector3_Order<double>> kfrac_list{{0.0, 0.0, 0.0}};
    const std::vector<Vector3_Order<int>> Rs{{0, 0, 0}, {1, 0, 0}};
    const std::map<atom_t, size_t> atom_nw{{0, 1}};
    AtomicBasis atbasis_wfc({1});
    atbasis_wfc.set_l_shells({{0}});
    PeriodicBoundaryData pbc;

    MeanField mf_ref(1, 1, 2, 1, 2);
    mf_ref.get_efermi() = 0.0;
    mf_ref.get_eigenvals()[0](0, 0) = -1.0;
    mf_ref.get_eigenvals()[0](0, 1) = 1.0;
    mf_ref.get_weight()[0](0, 0) = 1.0;
    mf_ref.get_weight()[0](0, 1) = 0.0;
    for (int ispinor = 0; ispinor != 2; ++ispinor)
    {
        mf_ref.get_eigenvectors()[0][ispinor][0].create(2, 1);
    }
    mf_ref.get_eigenvectors()[0][0][0](0, 0) = {0.6, 0.1};
    mf_ref.get_eigenvectors()[0][0][0](1, 0) = {-0.2, 0.7};
    mf_ref.get_eigenvectors()[0][1][0](0, 0) = {0.5, -0.3};
    mf_ref.get_eigenvectors()[0][1][0](1, 0) = {0.1, 0.4};

    std::map<Vector3_Order<int>, SpinorBlocks4<ComplexMatrix>> expected;
    for (const auto &R : Rs)
    {
        expected.emplace(R, get_symmetry_restored_dmat_cplx_R_spinor(ctx, wfc_layouts, mf_ref, 0,
                                                                     kfrac_list, R, atom_nw));
    }

    KPointBlacsProcessShape shape(1, size_global, false);
    KPointBlacsParallelContext context(shape, mpi_comm_global_h.comm, 1);
    const auto desc_wfc = context.create_array_desc(1, 2, 1, 2);
    const auto desc_dm = context.create_array_desc(1, 1);

    MeanField mf(1, 1, 2, 1, 2);
    mf.get_efermi() = 0.0;
    mf.get_eigenvals()[0](0, 0) = -1.0;
    mf.get_eigenvals()[0](0, 1) = 1.0;
    mf.get_weight()[0](0, 0) = 1.0;
    mf.get_weight()[0](0, 1) = 0.0;
    if (context.kpoint_blacs_root_global_rank(0) == myid_global)
    {
        mf.get_eigenvectors()[0][0][0] = mf_ref.get_eigenvectors()[0][0][0];
        mf.get_eigenvectors()[0][1][0] = mf_ref.get_eigenvectors()[0][1][0];
    }

    const auto actual = get_symmetry_restored_dmat_cplx_Rs_kblacs_para_spinor(
        0, mf, kfrac_list, Rs, context, desc_wfc, desc_dm, ctx, pbc, atbasis_wfc);

    for (const auto &R : Rs)
    {
        const auto &exp_blocks = expected.at(R);
        const auto &act_blocks = actual.at(R);
        const ComplexMatrix *exp_channel[4] = {&exp_blocks.b00, &exp_blocks.b01, &exp_blocks.b10,
                                               &exp_blocks.b11};
        const Matz *act_channel[4] = {&act_blocks.b00, &act_blocks.b01, &act_blocks.b10,
                                      &act_blocks.b11};
        for (int s = 0; s != 4; ++s)
        {
            for (int jloc = 0; jloc != desc_dm.n_loc(); ++jloc)
            {
                const int jglob = desc_dm.indx_l2g_c(jloc);
                for (int iloc = 0; iloc != desc_dm.m_loc(); ++iloc)
                {
                    const int iglob = desc_dm.indx_l2g_r(iloc);
                    if (!fequal((*act_channel[s])(iloc, jloc), (*exp_channel[s])(iglob, jglob),
                                cplxdb{1e-12, 0.0}))
                    {
                        throw std::runtime_error(
                            "kBLACS spinor density matrix did not match serial restore");
                    }
                }
            }
        }
    }
}

int main (int argc, char *argv[])
{
    int provided;
    MPI_Init_thread(&argc, &argv, LIBRPA_MPI_THREAD_LEVEL, &provided);
    init_global_mpi(MPI_COMM_WORLD);
    init_global_io();

#if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
    BlacsCtxtHandler world_blacs_h(MPI_COMM_WORLD);
    world_blacs_h.init();
    world_blacs_h.set_square_grid();
    world_blacs_h.init_ddla_handle();
#endif

    // Fictious examples
    test_dmat_cplx_Rs_kpara(3, 8, 2);
    test_dmat_cplx_Rs_kpara(15, 4, 2);
    test_gf_cplx_Rs_kpara(3, 7, 2, 1.0, 1.0);
    test_gf_cplx_Rs_kpara(15, 3, 2, 0.5, -1.0);
    test_gf_cplx_Rs_kpara(3, 7, 2, 1.0, -1.0, 1);
    test_gf_cplx_Rs_kpara(3, 7, 2, 1.0, 1.0, 3);
    test_gf_cplx_Rs_kpara(3, 7, 2, 1.0, -1.0, 3, 1);
    test_gf_cplx_Rs_kpara(3, 7, 2, 1.0, 1.0, 3, 1);

    // Actual examples
    test_dmat_cplx_Rs_kpara();
    test_dmat_cplx_Rs_kblacs_para_full_wfc();
    test_gf_cplx_imagtimes_Rs_kblacs_para_full_wfc();
    test_gf_cplx_imagtimes_Rs_kblacs_para_full_wfc(true);
    test_dm_gf_kblacs_para_redistributed_full_wfc();
    test_dmat_kblacs_reduced_kstar_matches_symmetry_restore();
    test_dmat_gf_kblacs_reduced_kstar_matches_full_bz_fourier();
    test_dmat_gf_kblacs_reduced_kstar_matches_full_bz_fourier(true);
    test_dmat_gf_kblacs_reduced_kstar_matches_full_bz_fourier(false, true);
    test_gf_kblacs_reduced_kstar_spinor_matches_serial_restore();
    test_dmat_kblacs_reduced_kstar_spinor_matches_serial_restore();

#if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
    world_blacs_h.exit();
#endif
    finalize_global_io();
    finalize_global_mpi();
    MPI_Finalize();

    return 0;
}
