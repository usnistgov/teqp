#define CATCH_CONFIG_MAIN
#include "catch/catch.hpp"

#include "teqp/core.hpp"
#include "teqp/models/pcsaft.hpp"
#include "teqp/models/cubicsuperancillary.hpp"
#include "teqp/models/CPA.hpp"
#include "teqp/models/eos.hpp"

#include "teqp/algorithms/VLE.hpp"
#include "teqp/algorithms/critical_tracing.hpp"

// Imports from boost
#include <boost/multiprecision/cpp_bin_float.hpp>
using namespace boost::multiprecision;
#include "teqp/finite_derivs.hpp"

auto build_vdW_argon() {
    double Omega_b = 1.0 / 8, Omega_a = 27.0 / 64;
    double Tcrit = 150.687, pcrit = 4863000.0; // Argon
    double R = get_R_gas<double>(); // Universal gas constant
    double b = Omega_b * R * Tcrit / pcrit;
    double ba = Omega_b / Omega_a / Tcrit / R;
    double a = b / ba;

    auto vdW = vdWEOS1(a, b);
    return vdW;
}

auto build_simple() {
    // Argon + Xenon
    std::valarray<double> Tc_K = { 150.687, 289.733 };
    std::valarray<double> pc_Pa = { 4863000.0, 5842000.0 };
    auto R = 1.380649e-23 * 6.02214076e23; ///< Exact value, given by k_B*N_A
    int i = 0;
    double ai = 27.0 / 64.0 * pow(R * Tc_K[i], 2) / pc_Pa[i];
    double bi = 1.0 / 8.0 * R * Tc_K[i] / pc_Pa[i];
    return vdWEOS1(ai, bi);
}
auto build_vdW() {
    // Argon + Xenon
    std::valarray<double> Tc_K = { 150.687, 289.733 };
    std::valarray<double> pc_Pa = { 4863000.0, 5842000.0 };
    return vdWEOS(Tc_K, pc_Pa);
}

TEST_CASE("Check virial coefficients for vdW", "[virial]")
{
    auto vdW = build_vdW_argon();

    double Omega_b = 1.0 / 8, Omega_a = 27.0 / 64;
    double Tcrit = 150.687, pcrit = 4863000.0; // Argon
    double R = get_R_gas<double>(); // Universal gas constant
    double b = Omega_b * R * Tcrit / pcrit;
    double ba = Omega_b / Omega_a / Tcrit / R;
    double a = b / ba;

    double T = 300.0;
    Eigen::ArrayXd molefrac(1); molefrac = 1.0;

    constexpr int Nvir = 8;

    // Numerical solutions from alphar
    using vd = VirialDerivatives<decltype(vdW)>;
    auto Bn = vd::get_Bnvir<Nvir, ADBackends::autodiff>(vdW, T, molefrac);
    auto Bnmcx = vd::get_Bnvir<Nvir, ADBackends::multicomplex>(vdW, T, molefrac);

    // Exact solutions for virial coefficients for van der Waals 
    auto get_vdW_exacts = [a, b, R, T](int Nmax) {
        std::map<int, double> o = { {2, b - a / (R * T)} };
        for (auto i = 3; i <= Nmax; ++i) {
            o[i] = pow(b, i - 1);
        }
        return o;
    };
    auto Bnexact = get_vdW_exacts(Nvir);

    // This one with complex step derivatives as another check
    double B2 = vd::get_B2vir(vdW, T, molefrac);
    double B2exact = b - a / (R * T);
    CHECK(std::abs(B2exact-Bnexact[2]) < 1e-15);
    CHECK(std::abs(B2-Bnexact[2]) < 1e-15);

    // And all the remaining derivatives
    for (auto i = 2; i <= Nvir; ++i) {
        auto numeric = Bn[i];
        auto exact = Bnexact[i];
        auto err = std::abs(numeric-exact);
        auto relerr = err / std::abs(Bn[i]);
        CAPTURE(numeric);
        CAPTURE(exact);
        CAPTURE(i);
        CAPTURE(relerr);
        CHECK(relerr < 1e-15);
    }
}

TEST_CASE("Check neff", "[virial]")
{
    double T = 298.15;
    double rho = 3.0;
    const Eigen::Array2d molefrac = { 0.5, 0.5 };
    auto f = [&T, &rho, &molefrac](const auto& model) {
        auto neff = TDXDerivatives<decltype(model)>::get_neff(model, T, rho, molefrac);
        CAPTURE(neff);
        CHECK(neff > 0);
        CHECK(neff < 100);
    };
    // This quantity is undefined for the van der Waals EOS because Ar20 is always 0
    //SECTION("vdW") {
    //    f(build_simple());
    //}
    SECTION("PCSAFT") {
        std::vector<std::string> names = { "Methane", "Ethane" };
        f(PCSAFTMixture(names));
    }
}

TEST_CASE("Check Hessian of Psir", "[virial]")
{
    
    double T = 298.15;
    double rho = 3.0;
    const Eigen::Array2d rhovec = { rho / 2, rho / 2 };
    auto get_worst_error = [&T, &rhovec](const auto &model){ 
        using id = IsochoricDerivatives <decltype(model)>;
        auto H1 = id::build_Psir_Hessian_autodiff(model, T, rhovec);
        auto H2 = id::build_Psir_Hessian_mcx(model, T, rhovec);
        auto err = (H1.array() - H2).abs().maxCoeff();
        CAPTURE(err);
        CHECK(err < 1e-15);
        return;
    };
    SECTION("simple") {
        get_worst_error(build_simple());
    }
    SECTION("less_simple") {
        get_worst_error(build_vdW());
    }
}

TEST_CASE("Check p four ways for vdW", "[virial][p]")
{
    auto model = build_simple();
    const double T = 298.15;
    const double rho = 3000.0;
    const auto rhovec = (Eigen::ArrayXd(2) << rho / 2, rho / 2).finished();
    const auto molefrac = rhovec/rhovec.sum();

    // Exact solution from EOS
    auto pexact = model.p(T, 1/rho);
    
    // Numerical solution from alphar
    using id = IsochoricDerivatives<decltype(model)>;
    auto pfromderiv = rho*model.R(molefrac)*T + id::get_pr(model, T, rhovec);
    using tdx = TDXDerivatives<decltype(model)>;
    auto p_ar0n = rho*model.R(molefrac) *T*(1 + tdx::get_Ar0n<3>(model, T, rho, molefrac)[1]);

    // Numerical solution from virial expansion
    constexpr int Nvir = 8;
    using vd = VirialDerivatives<decltype(model)>;
    auto Bn = vd::get_Bnvir<Nvir>(model, T, molefrac);
    auto Z = 1.0;
    for (auto i = 2; i <= Nvir; ++i){
        Z += Bn[i]*pow(rho, i-1);
        auto pvir = Z * rho * model.R(molefrac) * T;
    }
    auto pvir = Z*rho*model.R(molefrac) *T;

    CHECK(std::abs(pfromderiv - pexact)/pexact < 1e-15);
    CHECK(std::abs(pvir - pexact)/pexact < 1e-8);
    CHECK(std::abs(p_ar0n - pexact) / pexact < 1e-8);
}

TEST_CASE("Check 0n derivatives", "[virial][p]")
{
    std::vector<std::string> names = { "Methane", "Ethane" };
    auto model = PCSAFTMixture(names);

    const double T = 100.0;
    const double rho = 126.1856883066021; 
    const auto rhovec = (Eigen::ArrayXd(2) << rho, 0).finished();
    const auto molefrac = rhovec / rhovec.sum();

    using tdx = TDXDerivatives<decltype(model)>;
    auto Ar02 = tdx::get_Ar02(model, T, rho, molefrac);
    auto Ar02n = tdx::get_Ar0n<2>(model, T, rho, molefrac)[2];
    CHECK(std::abs(Ar02 - Ar02n) < 1e-13);

    auto Ar01 = tdx::get_Ar01(model, T, rho, molefrac);
    auto Ar01n = tdx::get_Ar0n<4>(model, T, rho, molefrac)[1];
    CHECK(std::abs(Ar01 - Ar01n) < 1e-13);

}

TEST_CASE("Test all vdW derivatives good to numerical precision", "[vdW]") 
{
    using my_float_type = boost::multiprecision::number<boost::multiprecision::cpp_bin_float<100U>>;

    double a = 1, b = 2;
    auto model = vdWEOS1(a, b);

    double T_ = 300;
    double rho = 2.3e-5;
    Eigen::ArrayXd z(1); z.fill(1.0);
    my_float_type T = T_, D = rho, h = pow(my_float_type(10.0), -10);

    auto fD = [&](const auto& x) { return model.alphar(T_, x, z); };

    // Exact values from differentiation of alphar = -log(1-b*rho)-a*rho/(R*T)
    my_float_type bb = b, brho = D * bb;
    auto der02exact = static_cast<double>(pow(brho / (brho - 1.0), 2));
    auto der03exact = static_cast<double>(-2 * pow((brho) / (brho - 1.0), 3));
    auto der04exact = static_cast<double>(6 * pow((brho) / (brho - 1.0), 4));

    SECTION("2nd derivs") {
        auto der02_2 = static_cast<double>(pow(D, 2) * centered_diff<2, 2>(fD, D, h));
        auto der02_4 = static_cast<double>(pow(D, 2) * centered_diff<2, 4>(fD, D, h));
        auto der02_6 = static_cast<double>(pow(D, 2) * centered_diff<2, 6>(fD, D, h));
        auto der02_ad = TDXDerivatives<decltype(model)>::get_Ar02(model, T_, rho, z);
        auto target = Approx(der02exact).margin(1e-15*der02exact);
        CHECK(der02_ad == target);
        CHECK(der02_2 == target);
        CHECK(der02_4 == target);
        CHECK(der02_6 == target);
    }
    SECTION("3rd derivs") {
        auto der03_2 = static_cast<double>(pow(D, 3) * centered_diff<3, 2>(fD, D, h));
        auto der03_4 = static_cast<double>(pow(D, 3) * centered_diff<3, 4>(fD, D, h));
        auto der03_6 = static_cast<double>(pow(D, 3) * centered_diff<3, 6>(fD, D, h));
        auto der03_ad = TDXDerivatives<decltype(model)>::get_Ar0n<3>(model, T_, rho, z)[3];
        auto target = Approx(der03exact).margin(1e-15*der03exact);
        CHECK(der03_ad == target);
        CHECK(der03_2 == target);
        CHECK(der03_4 == target);
        CHECK(der03_6 == target);
    }
    SECTION("4th derivs") {
        auto der04_2 = static_cast<double>(pow(D, 4) * centered_diff<4, 2>(fD, D, h));
        auto der04_4 = static_cast<double>(pow(D, 4) * centered_diff<4, 4>(fD, D, h));
        auto der04_6 = static_cast<double>(pow(D, 4) * centered_diff<4, 6>(fD, D, h));
        auto der04_ad = TDXDerivatives<decltype(model)>::get_Ar0n<4>(model, T_, rho, z)[4];
        auto target = Approx(der04exact).margin(1e-15*der04exact);
        CHECK(der04_ad == target);
        CHECK(der04_2 == target);
        CHECK(der04_4 == target);
        CHECK(der04_6 == target);
    }
}

TEST_CASE("Test infinite dilution critical locus derivatives", "[vdWcrit]")
{
    // Argon + Xenon
    std::valarray<double> Tc_K = { 150.687, 289.733 };
    std::valarray<double> pc_Pa = { 4863000.0, 5842000.0 };
    std::valarray<double> molefrac = { 1.0 };
    vdWEOS<double> vdW(Tc_K, pc_Pa);
    auto Zc = 3.0 / 8.0;
    using ct = CriticalTracing<decltype(vdW), double, Eigen::ArrayXd>;
    
    for (int i = 0; i < 2; ++i) {
        auto rhoc0 = pc_Pa[i] / (vdW.R(molefrac) * Tc_K[i]) / Zc;
        double T0 = Tc_K[i];
        Eigen::ArrayXd rhovec0(2); rhovec0.setZero(); rhovec0[i] = rhoc0;

        // Values for infinite dilution
        auto infdil = ct::get_drhovec_dT_crit(vdW, T0, rhovec0);
        auto epinfdil = ct::eigen_problem(vdW, T0, rhovec0);

        // Just slightly not infinite dilution, values should be very similar
        Eigen::ArrayXd rhovec0almost = rhovec0; rhovec0almost[1 - i] = 1e-6;
        auto dil = ct::get_drhovec_dT_crit(vdW, T0, rhovec0almost);
        auto epdil = ct::eigen_problem(vdW, T0, rhovec0almost);
        int rr = 0;
        
    }
}

TEST_CASE("Trace critical locus for vdW", "[vdW][crit]")
{
    // Argon + Xenon
    std::valarray<double> Tc_K = { 150.687, 289.733 };
    std::valarray<double> pc_Pa = { 4863000.0, 5842000.0 };
    const std::valarray<double> molefrac = { 1.0 };
    vdWEOS<double> vdW(Tc_K, pc_Pa);
    auto Zc = 3.0/8.0;
    std::valarray<double> max_spluses(Tc_K.size());
    for (auto ifluid = 0; ifluid < Tc_K.size(); ++ifluid) {
        auto rhoc0 = pc_Pa[ifluid] / (vdW.R(molefrac) * Tc_K[ifluid]) / Zc;
        double T0 = Tc_K[ifluid];
        Eigen::ArrayXd rhovec0(2); rhovec0 = 0.0; rhovec0[ifluid] = rhoc0;
        REQUIRE(rhovec0[ifluid] / rhovec0.sum() == 1.0);

        using id = IsochoricDerivatives<decltype(vdW)>;
        double splus = id::get_splus(vdW, T0, rhovec0);
        REQUIRE(splus == Approx(-log(1 - 1.0 / 3.0)));

        auto tic0 = std::chrono::steady_clock::now();
        std::string filename = "";
        using ct = CriticalTracing<decltype(vdW), double, Eigen::ArrayXd>;
        auto trace = ct::trace_critical_arclength_binary(vdW, T0, rhovec0, filename);
        auto tic1 = std::chrono::steady_clock::now();
        double max_splus = 0;
        for (auto &point : trace) {
            double splus = point.at("s^+");
            max_splus = std::max(max_splus, splus);
        }
        max_spluses[ifluid] = max_splus;
    }
    CHECK(max_spluses.min() == Approx(max_spluses.max()).epsilon(0.01));
    CHECK(max_spluses.min() > -log(1 - 1.0 / 3.0));
}

TEST_CASE("TEST B12", "") {
    const auto model = build_vdW();
    const double T = 298.15;
    const auto molefrac = (Eigen::ArrayXd(2) <<  1/3, 2/3).finished();
    using vd = VirialDerivatives<decltype(model)>;
    auto B12 = vd::get_B12vir(model, T, molefrac);
}

TEST_CASE("Test psir gradient", "") {
    const auto model = build_vdW();
    const double T = 298.15;

    using id = IsochoricDerivatives<decltype(model)>;
    const Eigen::Array2d rhovec = { 1, 2 };
    const Eigen::Array2d molefrac = { 1 / 3, 2 / 3 };
    auto psirfunc2 = [&model](const auto& T, const auto& rho_) {
        auto rhotot_ = rho_.sum();
        auto molefrac = (rho_ / rhotot_);
        return model.alphar(T, rhotot_, molefrac) * model.R(molefrac) * T * rhotot_;
    };
    auto chk0 = derivrhoi(psirfunc2, T, rhovec, 0);
    auto chk1 = derivrhoi(psirfunc2, T, rhovec, 1);
    auto grad = id::build_Psir_gradient_autodiff(model, T, rhovec);
    auto err0 = std::abs((chk0 - grad[0])/chk0);
    auto err1 = std::abs((chk1 - grad[1])/chk1);
    CAPTURE(err0);
    CAPTURE(err1);
    CHECK(err0 < 1e-12);
    CHECK(err1 < 1e-12);
}

TEST_CASE("Test extrapolate from critical point", "") {
    std::valarray<double> Tc_K = { 150.687};
    std::valarray<double> pc_Pa = { 4863000.0};
    std::valarray<double> molefrac = { 1.0 };
    vdWEOS<double> model(Tc_K, pc_Pa);
    const auto Zc = 3.0 / 8.0;
    auto stepper = [&](double step) {
        auto rhoc_molm3 = pc_Pa[0] / (model.R(molefrac) * Tc_K[0]) / Zc;
        auto T = Tc_K[0] - step; 
        auto rhovec = extrapolate_from_critical(model, Tc_K[0], rhoc_molm3, T);
        auto z = (Eigen::ArrayXd(1) << 1.0).finished();

        auto b = model.b(z), a = model.a(T, z), R = model.R(molefrac);
        auto Ttilde = R*T*b/a;
        using namespace CubicSuperAncillary;
        auto SArhoL = supercubic(VDW_CODE, RHOL_CODE, Ttilde) / b;
        auto SArhoV = supercubic(VDW_CODE, RHOV_CODE, Ttilde) / b;

        auto resid = IsothermPureVLEResiduals(model, T);
        auto rhosoln = do_pure_VLE_T(resid, rhovec[0], rhovec[1], 20);
        auto r0 = resid.call(rhosoln);

        auto errrhoL = SArhoL - rhosoln[0], errrhoV = SArhoV - rhosoln[1];
        if (std::abs(errrhoL)/SArhoL > 1e-10) { 
            throw std::range_error("rhoL error > 1e-10"); }
        if (std::abs(errrhoV)/SArhoV > 1e-10) { 
            throw std::range_error("rhoV error > 1e-10"); }
    };
    CHECK_NOTHROW(stepper(0.01));
    CHECK_NOTHROW(stepper(0.1));
    CHECK_NOTHROW(stepper(1.0));
    CHECK_NOTHROW(stepper(10.0));
}

TEST_CASE("Test pure VLE", "") {
    const auto model = build_vdW_argon();
    double T = 100.0;
    auto resid = IsothermPureVLEResiduals(model, T);
    auto rhovec = (Eigen::ArrayXd(2) << 22834.056386882046, 1025.106554560764).finished();
    auto r0 = resid.call(rhovec);
    auto J = resid.Jacobian(rhovec);
    auto v = J.matrix().colPivHouseholderQr().solve(-r0.matrix()).array().eval();
    auto rhovec1 = rhovec + v;
    auto r1 = resid.call(rhovec1);
    CHECK((r0.cwiseAbs() > r1.cwiseAbs()).eval().all());

    auto rhosoln = do_pure_VLE_T(resid, 22834.056386882046, 1025.106554560764, 20);
    auto rsoln = resid.call(rhosoln); 
    CHECK((rsoln.cwiseAbs() <  1e-8).all());
}

TEST_CASE("Test pure VLE with non-unity R0/Rr", "") {
    const auto model = build_vdW_argon();
    double T = 100.0;

    auto residnormal = IsothermPureVLEResiduals(model, T);
    auto soln0 = do_pure_VLE_T(residnormal, 22834.056386882046, 1025.106554560764, 20);
    auto r0 = residnormal.call(soln0);

    double Omega_b = 1.0 / 8, Omega_a = 27.0 / 64;
    double Tcrit = 150.687, pcrit = 4863000.0; // Argon
    double Rr = 7.9144;
    double b = Omega_b * Rr * Tcrit / pcrit;
    double ba = Omega_b / Omega_a / Tcrit / Rr; // b/a
    double a = b / ba;
    auto vdW = vdWEOS1(a, b);

    auto residspecial = IsothermPureVLEResiduals(vdW, T);
    residspecial.Rr = Rr;
    auto solnspecial = do_pure_VLE_T(residspecial, 22834.056386882046, 1025.106554560764, 20);
    auto r1 = residspecial.call(solnspecial);

    auto rr = 0;
}

TEST_CASE("Test water", "") {
    std::valarray<double> a0 = {0.12277}, bi = {0.000014515}, c1 = {0.67359}, Tc = {647.096}, 
                          molefrac = {1.0};
    auto R = 8.3144598; 
    CPA::CPACubic cub(CPA::cubic_flag::SRK, a0, bi, c1, Tc, R);
    double T = 400, rhomolar = 100;
    
    auto z = (Eigen::ArrayXd(1) << 1).finished();

    using tdx = TDXDerivatives<decltype(cub)>;
    auto alphar = cub.alphar(T, rhomolar, molefrac);
    double p_noassoc = T*rhomolar*R*(1+tdx::get_Ar01(cub, T, rhomolar, z));
    CAPTURE(p_noassoc);

    std::vector<CPA::association_classes> schemes = { CPA::association_classes::a4C };
    std::valarray<double> epsAB = { 16655 }, betaAB = { 0.0692 };
    CPA::CPAAssociation cpaa(std::move(cub), schemes, epsAB, betaAB, R);

    CPA::CPAEOS cpa(std::move(cub), std::move(cpaa));
    using tdc = TDXDerivatives<decltype(cpa)>;
    auto Ar01 = tdc::get_Ar01(cpa, T, rhomolar, z);
    double p_withassoc = T*rhomolar*R*(1 + Ar01);
    CAPTURE(p_withassoc);

    //REQUIRE(p_withassoc == 3.14);
}

TEST_CASE("Test water w/ factory", "") {
   using namespace CPA;
   nlohmann::json water = {
        {"a0i / Pa m^6/mol^2",0.12277 }, {"bi / m^3/mol", 0.000014515}, {"c1", 0.67359}, {"Tc / K", 647.096},
        {"epsABi / J/mol", 16655.0}, {"betaABi", 0.0692}, {"class","4C"}
   };
   nlohmann::json j = {
       {"cubic","SRK"},
       {"pures", {water}},
       {"R_gas / J/mol/K", 8.3144598}
   };
   auto cpa = CPAfactory(j);

   double T = 400, rhomolar = 100, R = 8.3144598;
   auto z = (Eigen::ArrayXd(1) << 1).finished();
   using tdc = TDXDerivatives<decltype(cpa)>;
   auto Ar01 = tdc::get_Ar01(cpa, T, rhomolar, z);
   double p_withassoc = T * rhomolar * R * (1 + Ar01);
   CAPTURE(p_withassoc);

   //REQUIRE(p_withassoc == 3.14);
}

TEST_CASE("Check zero(ish)","") {
    double zero = 0.0;
    REQUIRE(zero == 0.0);
    autodiff::dual2nd zeroad(0.0);
    REQUIRE(zeroad == 0.0);
    //mcx::MultiComplex zeromcx = 0.0;
    //REQUIRE(zeromcx == 0.0);
}