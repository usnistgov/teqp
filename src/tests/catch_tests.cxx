#define CATCH_CONFIG_MAIN
#include "catch/catch.hpp"

#include "teqp/core.hpp"

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

    auto T = 300.0;
    std::valarray<double> molefrac = { 1.0 };

    auto Nvir = 8;

    // Numerical solutions from alphar
    auto Bn = get_Bnvir(vdW, Nvir, T, molefrac);

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
    double B2 = get_B2vir(vdW, T, molefrac);
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

TEST_CASE("Check Hessian of Psir", "[virial]")
{
    double T = 298.15;
    double rho = 3.0;
    const std::valarray<double> rhovec = { rho / 2, rho / 2 };
    auto get_worst_error = [&T, &rhovec](const auto &model){ 
        auto H1 = build_Psir_Hessian_autodiff(model, T, rhovec);
        auto H2 = build_Psir_Hessian_mcx(model, T, rhovec);
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

TEST_CASE("Check p three ways for vdW", "[virial][p]")
{
    auto model = build_simple();
    const double T = 298.15;
    const double rho = 3000.0;
    const std::valarray<double> rhovec = { rho / 2, rho / 2 }, molefrac = {0.5, 0.5};

    // Exact solution from EOS
    auto pexact = model.p(T, 1/rho);
    
    // Numerical solution from alphar
    auto pfromderiv = rho*model.R*T + get_pr(model, T, rhovec);

    // Numerical solution from virial expansion
    auto Nvir = 8;
    auto Bn = get_Bnvir(model, 8, T, molefrac);
    auto Z = 1.0;
    for (auto i = 2; i <= Nvir; ++i){
        Z += Bn[i]*pow(rho, i-1);
        auto pvir = Z * rho * model.R * T;
    }
    auto pvir = Z*rho*model.R*T;

    CHECK(std::abs(pfromderiv - pexact)/pexact < 1e-15);
    CHECK(std::abs(pvir - pexact)/pexact < 1e-8);
}