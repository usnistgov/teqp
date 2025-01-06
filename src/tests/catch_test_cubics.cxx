
#include <fstream>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

using Catch::Approx;

#include "teqp/models/cubics/simple_cubics.hpp"
#include "teqp/models/cubics/advancedmixing_cubics.hpp"
#include "teqp/derivs.hpp"
#include "teqp/algorithms/VLE.hpp"
#include "teqp/cpp/teqpcpp.hpp"

#include <boost/numeric/odeint/stepper/euler.hpp>
#include <boost/numeric/odeint/stepper/runge_kutta_cash_karp54.hpp>

// Define the EOS types by interrogating the types returned by the respective
// factory function or by alias of the class name
using vad = std::valarray<double>;
using canonical_cubic_t = decltype(teqp::canonical_PR(vad{}, vad{}, vad{}));

using namespace teqp;

TEST_CASE("Test construction of cubic", "[cubic]")
{
    // Values taken from http://dx.doi.org/10.6028/jres.121.011
    std::valarray<double> Tc_K = { 190.564, 154.581, 150.687 },
                pc_Pa = { 4599200, 5042800, 4863000 }, 
               acentric = { 0.011, 0.022, -0.002};
    auto modelSRK = canonical_SRK(Tc_K, pc_Pa, acentric);
    auto modelPR = canonical_PR(Tc_K, pc_Pa, acentric);

    double T = 800, rho = 5000;
    auto molefrac = (Eigen::ArrayXd(3) << 0.5, 0.3, 0.2).finished();
    
    auto Ar02SRK = TDXDerivatives<decltype(modelSRK)>::get_Ar02(modelSRK, T, rho, molefrac);
    auto Ar01PR = TDXDerivatives<decltype(modelPR)>::get_Ar01(modelPR, T, rho, molefrac);
    auto Ar02PR = TDXDerivatives<decltype(modelPR)>::get_Ar02(modelPR, T, rho, molefrac);
    auto Ar03PR = TDXDerivatives<decltype(modelPR)>::get_Ar0n<3>(modelPR, T, rho, molefrac)[3];
    auto Ar04PR = TDXDerivatives<decltype(modelPR)>::get_Ar0n<4>(modelPR, T, rho, molefrac)[4];
    int rr = 0;
}

TEST_CASE("Check SRK with kij setting", "[cubic]")
{
    // Values taken from http://dx.doi.org/10.6028/jres.121.011
    std::valarray<double> Tc_K = { 190.564, 154.581, 150.687 },
        pc_Pa = { 4599200, 5042800, 4863000 },
        acentric = { 0.011, 0.022, -0.002 };
    Eigen::ArrayXXd kij_right(3, 3); kij_right.setZero();
    Eigen::ArrayXXd kij_bad(2, 20); kij_bad.setZero();

    SECTION("No kij") {
        CHECK_NOTHROW(canonical_SRK(Tc_K, pc_Pa, acentric));
    }
    SECTION("Correctly shaped kij matrix") {
        CHECK_NOTHROW(canonical_SRK(Tc_K, pc_Pa, acentric, kij_right));
    }
    SECTION("Incorrectly shaped kij matrix") {
        CHECK_THROWS(canonical_SRK(Tc_K, pc_Pa, acentric, kij_bad));
    }
}

TEST_CASE("Check calling superancillary curves", "[cubic][superanc]") 
{
    std::valarray<double> Tc_K = { 150.687 };
    std::valarray<double> pc_Pa = { 4863000.0 };
    std::valarray<double> acentric = { 0.0 }; 
    std::valarray<double> z = {1.0};
    SECTION("PR") {
        auto model = canonical_PR(Tc_K, pc_Pa, acentric);
        auto [rhoL, rhoV] = model.superanc_rhoLV(130.0);
        CHECK(rhoL > rhoV);
    }
    SECTION("PR super large temp") {
        auto model = canonical_PR(Tc_K, pc_Pa, acentric);
        CHECK_THROWS(model.superanc_rhoLV(1.3e6));
    }
    SECTION("PR super small temp") {
        auto model = canonical_PR(Tc_K, pc_Pa, acentric);
        CHECK_THROWS(model.superanc_rhoLV(1.3e-10));
    }
    SECTION("SRK") {
        auto model = canonical_SRK(Tc_K, pc_Pa, acentric);
        auto [rhoL, rhoV] = model.superanc_rhoLV(130.0);
        CHECK(rhoL > rhoV);
    }
    SECTION("SRK with custom R") {
        auto model = canonical_SRK(Tc_K, pc_Pa, acentric, {}, 8.4);
        CHECK(model.R(z) == 8.4);
    }
    SECTION("PR with custom R") {
        auto model = canonical_PR(Tc_K, pc_Pa, acentric, {}, 8.4);
        CHECK(model.R(z) == 8.4);
    }
}

TEST_CASE("Check orthobaric density derivatives for pure fluid", "[cubic][superanc]")
{
    std::valarray<double> Tc_K = { 150.687 };
    std::valarray<double> pc_Pa = { 4863000.0};
    std::valarray<double> acentric = { 0.0};
    
    double T = 130.0, dT = 0.001;
    auto molefrac = (Eigen::ArrayXd(1) << 1.0).finished();
    
    auto model = canonical_PR(Tc_K, pc_Pa, acentric);
    using tdx = TDXDerivatives<decltype(model)>;
    using iso = IsochoricDerivatives<decltype(model)>;
    
    auto R = model.R(molefrac);
    auto [rhoL, rhoV] = model.superanc_rhoLV(T);
    CHECK(rhoL > rhoV);
    
    // Finite difference test
    auto [rhoLp, rhoVp] = model.superanc_rhoLV(T+dT);
    auto [rhoLm, rhoVm] = model.superanc_rhoLV(T-dT);
    auto pLp = rhoLp*R*(T+dT) + iso::get_pr(model, T+dT, rhoLp*molefrac);
    auto pLm = rhoLm*R*(T-dT) + iso::get_pr(model, T-dT, rhoLm*molefrac);
    
    // Exact solution for density derivative
    // Change in enthalpy (Deltah) is equal to change in residual enthalpy (Deltahr) because ideal parts cancel
    auto hrVLERTV = tdx::get_Ar01(model, T, rhoV, molefrac) + tdx::get_Ar10(model, T, rhoV, molefrac);
    auto hrVLERTL = tdx::get_Ar01(model, T, rhoL, molefrac) + tdx::get_Ar10(model, T, rhoL, molefrac);
    auto dpdrhoL = R*T*(1 + 2*tdx::get_Ar01(model, T, rhoL, molefrac) + tdx::get_Ar02(model, T, rhoL, molefrac));
    auto dpdTL = R*rhoL*(1 + tdx::get_Ar01(model, T, rhoL, molefrac) - tdx::get_Ar11(model, T, rhoL, molefrac));
    auto deltahr_over_T = R*(hrVLERTV-hrVLERTL);
    auto dpsatdT = deltahr_over_T/(1/rhoV-1/rhoL); // From Clausius-Clapeyron; dp/dT = Deltas/Deltav = Deltah/(T*Deltav); Delta=V-L
    
    auto dpsatdT_routine = dpsatdT_pure(model, T, rhoL, rhoV);
    
    CHECK(dpsatdT == Approx((pLp - pLm)/(2*dT)));
    CHECK(dpsatdT_routine == Approx((pLp - pLm)/(2*dT)));
    
    auto drhosatdTL = -dpdTL/dpdrhoL + dpsatdT/dpdrhoL;
    CHECK(drhosatdTL == Approx((rhoLp-rhoLm)/(2*dT)));
}

TEST_CASE("Check manual integration of subcritical VLE isotherm for binary mixture", "[cubic][isochoric][traceisotherm]")
{
    using namespace boost::numeric::odeint;
    // Values taken from http://dx.doi.org/10.6028/jres.121.011
    std::valarray<double> Tc_K = { 190.564, 154.581},
                         pc_Pa = { 4599200, 5042800},
                      acentric = { 0.011, 0.022};
    const auto modelptr = teqp::cppinterface::adapter::make_owned(canonical_PR(Tc_K, pc_Pa, acentric));
    const auto& model = teqp::cppinterface::adapter::get_model_cref<canonical_cubic_t>(modelptr.get());
    const auto N = Tc_K.size();
    using state_type = std::vector<double>; 
    REQUIRE(N == 2);
    auto get_start = [&](double T, auto i) {
        std::valarray<double> Tc_(Tc_K[i], 1), pc_(pc_Pa[i], 1), acentric_(acentric[i], 1);
        auto PR = canonical_PR(Tc_, pc_, acentric_);
        auto [rhoL, rhoV] = PR.superanc_rhoLV(T);
        state_type o(N*2);
        o[i] = rhoL;
        o[1 - i] = 0;
        o[i + N] = rhoV;
        o[1 - i + N] = 0;
        return o;
    };
    double T = 120; 
    // Derivative function with respect to pressure
    auto xprime = [&](const state_type& X, state_type& Xprime, double /*t*/) {
        REQUIRE(X.size() % 2 == 0);
        auto N = X.size() / 2;
        // Memory maps into the state vector for inputs and their derivatives
        auto rhovecL = Eigen::Map<const Eigen::ArrayXd>(&(X[0]), N);
        auto rhovecV = Eigen::Map<const Eigen::ArrayXd>(&(X[0])+N, N);
        auto drhovecdpL = Eigen::Map<Eigen::ArrayXd>(&(Xprime[0]), N);
        auto drhovecdpV = Eigen::Map<Eigen::ArrayXd>(&(Xprime[0]) + N, N); 
        std::tie(drhovecdpL, drhovecdpV) = get_drhovecdp_Tsat(model, T, rhovecL.eval(), rhovecV.eval());
    };
    auto get_p = [&](const state_type& X) {
        REQUIRE(X.size() % 2 == 0);
        auto N = X.size() / 2;
        // Memory maps into the state vector for rho vector
        auto rhovecL = Eigen::Map<const Eigen::ArrayXd>(&(X[0]), N);
        auto rho = rhovecL.sum();
        auto molefrac = rhovecL / rhovecL.sum();

        auto pfromderiv = rho * modelptr->R(molefrac) * T + modelptr->get_pr(T, rhovecL);
        return pfromderiv;
    };

    SECTION("Manual integration") {

        for (int i : { 0 }) {
            state_type X0 = get_start(T, i); // Starting point; liquid, then vapor
            double p0 = get_p(X0);
            state_type Xfinal = get_start(T, 1 - i); // Ending point; liquid, then vapor
            double pfinal = get_p(Xfinal);

            //euler<state_type> integrator;
            runge_kutta_cash_karp54< state_type > integrator;
            int Nstep = 10000;
            double p = p0, pmax = pfinal, dp = (pmax - p0) / (Nstep - 1);

            auto write = [&]() {
                //std::cout << p << " " << X0[0] << "," << X0[1] << std::endl;
            };
            for (auto i = 0; p < pmax; ++i) {
                if (p + dp > pmax) { break; }
                write();
                integrator.do_step(xprime, X0, p, dp);
                p += dp;

                // Try to polish the solution (but don't use the polished values)
                {
                    auto rhovecL = Eigen::Map<const Eigen::ArrayXd>(&(X0[0]), N).eval();
                    auto rhovecV = Eigen::Map<const Eigen::ArrayXd>(&(X0[0 + N]), N).eval();
                    auto x = (Eigen::ArrayXd(2) << rhovecL(0) / rhovecL.sum(), rhovecL(1) / rhovecL.sum()).finished();
                    auto [return_code, rhoL, rhoV] = mix_VLE_Tx(model, T, rhovecL, rhovecV, x, 1e-10, 1e-8, 1e-10, 1e-8, 10);

                    // And the other way around just to test the routine for TP solving
                    auto r = mix_VLE_Tp(model, T, p*1.1, rhovecL, rhovecV);
                    int rr = 0;

                }
                
            }
            double diffs = 0;
            for (auto i = 0; i < X0.size(); ++i) {
                diffs += std::abs(X0[i] - Xfinal[i]);
            }
            CHECK(diffs < 0.1);
            write();
        }
    }
    SECTION("Parametric integration of isotherm") {
        int i = 0;
        auto X = get_start(T, 0);
        state_type Xfinal = get_start(T, 1 - i); // Ending point; liquid, then vapor
        double pfinal_goal = get_p(Xfinal); 
        
        auto N = X.size() / 2;
        Eigen::ArrayXd rhovecL0 = Eigen::Map<const Eigen::ArrayXd>(&(X[0]), N);
        Eigen::ArrayXd rhovecV0 = Eigen::Map<const Eigen::ArrayXd>(&(X[0]) + N, N);
        TVLEOptions opt;
        opt.abs_err = 1e-10;
        opt.rel_err = 1e-10;
        opt.integration_order = 5;
        auto J = trace_VLE_isotherm_binary(model, T, rhovecL0, rhovecV0, opt);
        auto Nstep = J.size();

        std::ofstream file("isoT.json"); file << J;

        double pfinal = J.back().at("pL / Pa").back();
        CHECK(std::abs(pfinal / pfinal_goal-1) < 1e-5);
    }
}

TEST_CASE("Check infinite dilution of isoline VLE derivatives", "[cubic][isochoric][infdil]")
{
    // Methane + propane
    std::valarray<double> Tc_K = { 190.564, 369.89 },
        pc_Pa = { 4599200, 4251200.0 },
        acentric = { 0.011, 0.1521 };
    auto model = canonical_PR(Tc_K, pc_Pa, acentric);
    const auto N = Tc_K.size();
    
    using state_type = std::valarray<double>;
    REQUIRE(N == 2);
    auto get_start = [&](double T, auto i) {
        std::valarray<double> Tc_(Tc_K[i], 1), pc_(pc_Pa[i], 1), acentric_(acentric[i], 1);
        auto PR = canonical_PR(Tc_, pc_, acentric_);
        auto [rhoL, rhoV] = PR.superanc_rhoLV(T);
        auto z = (Eigen::ArrayXd(1) << 1.0).finished();
        using tdx = TDXDerivatives<decltype(model)>;
        auto p0 = rhoL * PR.R(z) * T * (1 + tdx::get_Ar01(PR, T, rhoL, z));
        //std::cout << p << std::endl;
        state_type o(N * 2);
        o[i] = rhoL;
        o[i + N] = rhoV;
        return std::make_tuple(o, p0);
    };
    int i = 1;
    double T = 250;
    auto [rhostart_dil, p0] = get_start(T, i);

    auto checker = [](auto & dernotdil, auto &derdil) {
        auto err0 = (std::get<0>(dernotdil).array()/std::get<0>(derdil).array() - 1).cwiseAbs().maxCoeff();
        auto err1 = (std::get<1>(dernotdil).array()/std::get<1>(derdil).array() - 1).cwiseAbs().maxCoeff();
        CAPTURE(err0);
        CAPTURE(err1);
        return err0 < 1e-5 && err1 < 1e-5; // These are absolute fractional deviations
    };

    SECTION("Along isotherm") {
        // Derivative function with respect to p
        std::valarray<double> rhostart_notdil = rhostart_dil;
        rhostart_notdil[1 - i] += 1e-3;
        rhostart_notdil[1 - i + N] += 1e-3;
        // Polish the pertubed solution
        Eigen::ArrayXd rhoL0 = Eigen::Map<const Eigen::ArrayXd>(&(rhostart_notdil[0]), 2);
        Eigen::ArrayXd rhoV0 = Eigen::Map<const Eigen::ArrayXd>(&(rhostart_notdil[0]) + N, N);
        Eigen::ArrayXd xL0 = rhoL0 / rhoL0.sum();
        auto [code, rhoL00, rhoV00] = mix_VLE_Tx(model, T, rhoL0, rhoV0, xL0, 1e-10, 1e-10, 1e-10, 1e-10, 10);
        Eigen::Map<Eigen::ArrayXd>(&(rhostart_notdil[0]), 2) = rhoL00;
        Eigen::Map<Eigen::ArrayXd>(&(rhostart_notdil[2]), 2) = rhoV00;

        auto xprime = [&](const state_type& X) {
            REQUIRE(X.size() % 2 == 0);
            auto N = X.size() / 2;
            // Memory maps into the state vector for inputs and their derivatives
            auto rhovecL = Eigen::Map<const Eigen::ArrayXd>(&(X[0]), N);
            auto rhovecV = Eigen::Map<const Eigen::ArrayXd>(&(X[0]) + N, N);
            return get_drhovecdp_Tsat(model, T, rhovecL.eval(), rhovecV.eval());
        };
        auto dernotdil = xprime(rhostart_notdil); 
        auto derdil = xprime(rhostart_dil);
        CHECK(checker(dernotdil, derdil));
    }
    SECTION("Along isobar") {
        std::valarray<double> rhostart_notdil = rhostart_dil;
        rhostart_notdil[1 - i] += 1e-3;
        rhostart_notdil[1 - i + N] += 1e-3;
        // Polish the pertubed solution
        Eigen::ArrayXd rhoL0 = Eigen::Map<const Eigen::ArrayXd>(&(rhostart_notdil[0]), 2);
        Eigen::ArrayXd rhoV0 = Eigen::Map<const Eigen::ArrayXd>(&(rhostart_notdil[0]) + N, N);
        Eigen::ArrayXd xL0 = rhoL0 / rhoL0.sum();
        
        auto [code, Tnew, rhoL00, rhoV00] = mixture_VLE_px(model, p0, xL0, T, rhoL0, rhoV0);
        Eigen::Map<Eigen::ArrayXd>(&(rhostart_notdil[0]), 2) = rhoL00;
        Eigen::Map<Eigen::ArrayXd>(&(rhostart_notdil[2]), 2) = rhoV00;

        // Derivative function with respect to T
        auto xprime = [&](double T, const state_type& X) {
            REQUIRE(X.size() % 2 == 0);
            auto N = X.size() / 2;
            // Memory maps into the state vector for inputs and their derivatives
            auto rhovecL = Eigen::Map<const Eigen::ArrayXd>(&(X[0]), N);
            auto rhovecV = Eigen::Map<const Eigen::ArrayXd>(&(X[0]) + N, N);
            return get_drhovecdT_psat(model, T, rhovecL.eval(), rhovecV.eval());
        };
        auto dernotdil = xprime(Tnew, rhostart_notdil); 
        auto derdil = xprime(T, rhostart_dil);
        CHECK(checker(dernotdil, derdil));
    }
}

TEST_CASE("Check manual integration of subcritical VLE isobar for binary mixture", "[cubic][isochoric][traceisobar]")
{
    using namespace boost::numeric::odeint;
    // Values taken from http://dx.doi.org/10.6028/jres.121.011
    std::valarray<double> Tc_K = { 190.564, 154.581 },
        pc_Pa = { 4599200, 5042800 },
        acentric = { 0.011, 0.022 };
    auto model = canonical_PR(Tc_K, pc_Pa, acentric);
    const auto N = Tc_K.size();
    using state_type = std::vector<double>;
    REQUIRE(N == 2);

    auto get_start = [&](double T, auto i) {
        std::valarray<double> Tc_(Tc_K[i], 1), pc_(pc_Pa[i], 1), acentric_(acentric[i], 1);
        auto PR = canonical_PR(Tc_, pc_, acentric_);
        auto [rhoL, rhoV] = PR.superanc_rhoLV(T);
        state_type o(N * 2);
        o[i] = rhoL;
        o[1 - i] = 0;
        o[i + N] = rhoV;
        o[(1 - i) + N] = 0;
        return o;
    };
    double T0 = 120; // Just to get a pressure, start at this point

    // Derivative function with respect to temperature at constant pressure
    auto xprime = [&](const state_type& X, state_type& Xprime, double T) {
        REQUIRE(X.size() % 2 == 0);
        auto N = X.size() / 2;
        // Memory maps into the state vector for inputs and their derivatives
        auto rhovecL = Eigen::Map<const Eigen::ArrayXd>(&(X[0]), N);
        auto rhovecV = Eigen::Map<const Eigen::ArrayXd>(&(X[0]) + N, N);
        auto drhovecdTL = Eigen::Map<Eigen::ArrayXd>(&(Xprime[0]), N);
        auto drhovecdTV = Eigen::Map<Eigen::ArrayXd>(&(Xprime[0]) + N, N);
        std::tie(drhovecdTL, drhovecdTV) = get_drhovecdT_psat(model, T, rhovecL.eval(), rhovecV.eval());
    };
    auto get_p = [&](const state_type& X, double T) {
        REQUIRE(X.size() % 2 == 0);
        auto N = X.size() / 2;
        // Memory maps into the state vector for rho vector
        auto rhovecL = Eigen::Map<const Eigen::ArrayXd>(&(X[0]), N);
        auto rho = rhovecL.sum();
        auto molefrac = rhovecL / rhovecL.sum();

        using id = IsochoricDerivatives<decltype(model)>;
        auto pfromderiv = rho * model.R(molefrac) * T + id::get_pr(model, T, rhovecL);
        return pfromderiv;
    };
    SECTION("Manual integration") {

        for (int i : { 0 }) {
            state_type X0 = get_start(T0, i); // Starting point; liquid, then vapor
            double Tfinal = T0 - 40;
            double pinit = get_p(X0, T0);

            //euler<state_type> integrator;
            runge_kutta_cash_karp54< state_type > integrator;
            int Nstep = 1000;
            double T = T0, Tmax = Tfinal, dT = (Tmax - T0) / (Nstep - 1);

            auto write = [&]() {
                //std::cout << T << " " << X0[0] / (X0[0] + X0[1]) << std::endl;
            };
            for (auto k = 0; k < Nstep; ++k) {
                write();
                integrator.do_step(xprime, X0, T, dT);
                T += dT;

                // Try to polish the solution (but don't use the polished values)
                {
                    Eigen::ArrayXd rhovecL = Eigen::Map<const Eigen::ArrayXd>(&(X0[0]), N).eval();
                    Eigen::ArrayXd rhovecV = Eigen::Map<const Eigen::ArrayXd>(&(X0[0 + N]), N).eval();
                    Eigen::ArrayXd x = (Eigen::ArrayXd(2) << rhovecL(0) / rhovecL.sum(), rhovecL(1) / rhovecL.sum()).finished();
                    double p = get_p(X0, T);
                    auto [return_code, Tnew, rhovecLnew, rhovecVnew] = mixture_VLE_px(model, p, x, T, rhovecL, rhovecV);
                    int rr = 0;
                }
                if (X0[0] / (X0[0] + X0[1]) < 0.01) {
                    break;
                }
            }
            double pfinal = get_p(X0, T);

            double diffs = 0;
            for (auto i = 0U; i < X0.size(); ++i) {
                diffs += std::abs(pinit-pfinal);
            }
            CHECK(diffs < 0.1);
        }
    }
    SECTION("Parametric integration of isobar") {
        auto X = get_start(T0, 0);
        double pinit = get_p(X, T0);

        auto N = X.size() / 2;
        Eigen::ArrayXd rhovecL0 = Eigen::Map<const Eigen::ArrayXd>(&(X[0]), N);
        Eigen::ArrayXd rhovecV0 = Eigen::Map<const Eigen::ArrayXd>(&(X[0]) + N, N);
        PVLEOptions opt;
        opt.abs_err = 1e-10;
        opt.rel_err = 1e-10;
        opt.integration_order = 5;
        auto J = trace_VLE_isobar_binary(model, pinit, T0, rhovecL0, rhovecV0, opt);
        auto Nstep = J.size();

        std::ofstream file("isoP.json"); file << J;
    }
}

TEST_CASE("Bad kmat options", "[PRkmat]"){
    SECTION("null; ok"){
        auto j = nlohmann::json::parse(R"({
            "kind": "PR",
            "model": {
                "Tcrit / K": [190],
                "pcrit / Pa": [3.5e6],
                "acentric": [0.11],
                "kmat": null
            }
        })");
        CHECK_NOTHROW(teqp::cppinterface::make_model(j));
    }
    SECTION("empty; ok"){
        auto j = nlohmann::json::parse(R"({
            "kind": "PR",
            "model": {
                "Tcrit / K": [190],
                "pcrit / Pa": [3.5e6],
                "acentric": [0.11],
                "kmat": []
            }
        })");
        CHECK_NOTHROW(teqp::cppinterface::make_model(j));
    }
    SECTION("empty for two components; ok"){
        auto j = nlohmann::json::parse(R"({
            "kind": "PR",
            "model": {
                "Tcrit / K": [190,200],
                "pcrit / Pa": [3.5e6,4e6],
                "acentric": [0.11,0.2],
                "kmat": []
            }
        })");
        CHECK_NOTHROW(teqp::cppinterface::make_model(j));
    }
    SECTION("wrong size for two components; fail"){
        auto j = nlohmann::json::parse(R"({
            "kind": "PR",
            "model": {
                "Tcrit / K": [190,200],
                "pcrit / Pa": [3.5e6,4e6],
                "acentric": [0.11,0.2],
                "kmat": [0.001]
            }
        })");
        CHECK_THROWS(teqp::cppinterface::make_model(j));
    }
}

TEST_CASE("Check generalized and alphas", "[PRalpha]"){
    auto j0 = nlohmann::json::parse(R"(
    {
        "kind": "PR",
        "model": {
            "Tcrit / K": [190],
            "pcrit / Pa": [3.5e6],
            "acentric": [0.11]
        }
    }
    )");
    auto j1 = nlohmann::json::parse(R"(
    {
        "kind": "cubic",
        "model": {
            "type": "PR",
            "Tcrit / K": [190],
            "pcrit / Pa": [3.5e6],
            "acentric": [0.11]
        }
    }
    )");
    auto j2 = nlohmann::json::parse(R"(
    {
        "kind": "cubic",
        "model": {
            "type": "PR",
            "Tcrit / K": [190],
            "pcrit / Pa": [3.5e6],
            "acentric": [0.11],
            "alpha": [
                {"type": "Twu", "c": [1, 2, 3]}
            ]
        }
    }
    )");
    
    // Parameters from Horstmann et al., doi:10.1016/j.fluid.2004.11.002
    auto j2MC = R"(
    {
        "kind": "cubic",
        "model": {
            "type": "SRK",
            "Tcrit / K": [647.30],
            "pcrit / Pa": [22048.321e3],
            "acentric": [0.11],
            "alpha": [
                {"type": "Mathias-Copeman", "c": [1.07830, -0.58321, 0.54619]}
            ]
        }
    }
    )"_json;
    
    auto j3 = nlohmann::json::parse(R"(
    {
        "kind": "PR",
        "model": {
            "Tcrit / K": [190],
            "pcrit / Pa": [3.5e6],
            "acentric": [0.11],
            "alpha": [
                {"type": "Twu", "c": [1, 2, 3]}
            ]
        }
    }
    )");
    
    auto j4 = nlohmann::json::parse(R"(
    {
        "kind": "cubic",
        "model": {
            "type": "PR",
            "Tcrit / K": [190],
            "pcrit / Pa": [3.5e6],
            "acentric": [0.11],
            "alpha": [
                {"type": "Twu", "c": [1, 2, 3]},
                {"type": "Twu", "c": [4, 5, 6]}
            ]
        }
    }
    )");
    
    SECTION("canonical PR"){
        const auto modelptr0 = teqp::cppinterface::make_model(j0);
        const auto& m0 = teqp::cppinterface::adapter::get_model_cref<canonical_cubic_t>(modelptr0.get());
        
        const auto modelptr1 = teqp::cppinterface::make_model(j1);
        const auto& m1 = teqp::cppinterface::adapter::get_model_cref<canonical_cubic_t>(modelptr1.get());
        
        const auto modelptr2 = teqp::cppinterface::make_model(j2);
        const auto& m2 = teqp::cppinterface::adapter::get_model_cref<canonical_cubic_t>(modelptr2.get());
        
        const auto modelptr2MC = teqp::cppinterface::make_model(j2MC);
        const auto& m2MC = teqp::cppinterface::adapter::get_model_cref<canonical_cubic_t>(modelptr2MC.get());
        
        CHECK(m1.get_meta() == m0.get_meta());
        CHECK(m2.get_meta() != m0.get_meta());
        
        CHECK_THROWS(teqp::cppinterface::make_model(j3));
        CHECK_THROWS(teqp::cppinterface::make_model(j4));
    }
    SECTION("water with Mathias-Copeman"){
        const auto m = teqp::cppinterface::make_model(j2MC);
        const auto& mptr = teqp::cppinterface::adapter::get_model_cref<canonical_cubic_t>(m.get());
        double T = 99.9 + 273.15;
        auto [rhoL, rhoV] = mptr.superanc_rhoLV(T);
        auto z = (Eigen::ArrayXd(1) << 1.0).finished();
        double R = m->get_R(z);
        double p = rhoL*R*T*(1+m->get_Ar01(T, rhoL, z));
        CHECK(p == Approx(101325).margin(1000));
    }
}

TEST_CASE("QCPR", "[QCPR]"){
    
    /// Naming convention of variables follows the paper, not teqp
    auto j = R"(
    {
        "kind": "QCPRAasen",
        "model": {
            "Ls": [156.21, 0.40453],
            "Ms": [-0.0062072, 0.95861],
            "Ns": [5.047, 0.8396],
            "As": [3.0696, 0.4673],
            "Bs": [12.682, 2.4634],
            "cs / m^3/mol": [-3.8139, -2.4665],
            "Tcrit / K": [33.19, 44.492],
            "pcrit / Pa": [12.964e5, 26.79],
            "kmat": [[0.0, 0.18], [0.18, 0.0]],
            "lmat": [[0.0, 0.0], [0.0, 0.0]]
        }
    }
    )"_json;
    auto model = make_model(j);
    double T = 50.0;
    auto z = (Eigen::ArrayXd(2) << 0.3, 0.7).finished();
    CHECK(std::isfinite(model->get_B12vir(T, z)));
    
    SECTION("funny R"){
        /// Naming convention of variables follows the paper, not teqp
        auto j = R"(
        {
            "kind": "QCPRAasen",
            "model": {
                "Ls": [156.21, 0.40453],
                "Ms": [-0.0062072, 0.95861],
                "Ns": [5.047, 0.8396],
                "As": [3.0696, 0.4673],
                "Bs": [12.682, 2.4634],
                "cs / m^3/mol": [-3.8139, -2.4665],
                "Tcrit / K": [33.19, 44.492],
                "pcrit / Pa": [12.964e5, 26.79],
                "kmat": [[0.0, 0.18], [0.18, 0.0]],
                "lmat": [[0.0, 0.0], [0.0, 0.0]],
                "R / J/mol/K": 8.4
            }
        }
        )"_json;
        auto model = make_model(j);
        auto z = (Eigen::ArrayXd(2) << 0.3, 0.7).finished();
        CHECK(model->get_R(z) == 8.4);
    }
}

TEST_CASE("Advanced cubic EOS", "[AdvancedPR]"){

    // Values for CO2 + N2 from Table 6.3 of Lasala dissertation which come from DIPPR
    std::valarray<double> Tc_K = {304.21, 126.19},
                pc_Pa = {7.383e6, 3395800.0},
               acentric = {0.22394, 0.0372};
    
    // Classic Peng-Robinson alpha function and covolume
    std::vector<AlphaFunctionOptions> alphas;
    std::vector<double> b;
    for (auto i = 0; i < acentric.size(); ++i){
        auto mi = 0.37464 + 1.54226*acentric[i] - 0.26992*acentric[i]*acentric[i];
        alphas.push_back(BasicAlphaFunction(Tc_K[i], mi));
        b.push_back(teqp::AdvancedPRaEres<double>::get_bi(Tc_K[i], pc_Pa[i]));
    }
    
    SECTION("Check pcrit"){
        
        // Matrices for putting the coefficients in directly
//        Eigen::ArrayXXd mWilson = (Eigen::ArrayXXd(2,2) << 0.0, -3.4768, 3.5332, 0.0).finished().transpose();
//        Eigen::ArrayXXd nWilson = (Eigen::ArrayXXd(2,2) << 0.0, 825, -585, 0.0).finished().transpose();
        
        Eigen::ArrayXXd mWilson = (Eigen::ArrayXXd(2,2) << 0.0, 0.0, 0.0, 0.0).finished();
        
        for (double T: {223.1, 253.05, 273.1}){
            std::size_t ipure = 0;
            
            double A12 = -3.4768*T + 825;
            double A21 = 3.5332*T - 585;
            Eigen::ArrayXXd nWilson = (Eigen::ArrayXXd(2,2) << 0.0, A12, A21, 0.0).finished();
            auto aresmodel = WilsonResidualHelmholtzOverRT<double>(b, mWilson, nWilson);
            AdvancedPRaEOptions options; options.CEoS = -0.52398;
            auto model = teqp::AdvancedPRaEres<double>(Tc_K, pc_Pa, alphas, aresmodel, Eigen::ArrayXXd::Zero(2, 2), options);
            
            // Solve for starting point with superancillary function
            auto [rhoL, rhoV] = model.superanc_rhoLV(T, ipure);
            
            Eigen::ArrayXd rhovecL0 = Eigen::ArrayXd::Zero(2); rhovecL0(ipure) = rhoL;
            Eigen::ArrayXd rhovecV0 = Eigen::ArrayXd::Zero(2); rhovecV0(ipure) = rhoV;
            
            TVLEOptions opt; opt.revision = 2;
            auto J = trace_VLE_isotherm_binary(model, T, rhovecL0, rhovecV0, opt);
            auto pcrit = J["data"].back()["pL / Pa"];
            std::cout << T << ", " << pcrit << std::endl;
            std::ofstream file("isoT_advcub"+std::to_string(T)+".json"); file << J;
        }
    }
}

TEST_CASE("Advanced cubic EOS w/ make_model", "[AdvancedPR]"){
    auto j = R"({
        "kind": "advancedPRaEres",
        "model": {
           "Tcrit / K": [304.21, 126.19],
           "pcrit / Pa": [7.383e6, 3395800.0],
           "alphas": [{"type": "PR78", "acentric": 0.22394}, {"type": "PR78", "acentric": 0.0372}],
           "aresmodel": {"type": "Wilson", "m": [[0.0, -3.4768], [3.5332, 0.0]], "n": [[0.0, 825], [-585, 0.0]]},
           "options": {"s": 2.0, "brule": "Quadratic", "CEoS": -0.52398}
        }
    })"_json;
    auto model = make_model(j);
    SECTION("funny R"){
        auto j = R"({
            "kind": "advancedPRaEres",
            "model": {
               "Tcrit / K": [304.21, 126.19],
               "pcrit / Pa": [7.383e6, 3395800.0],
               "alphas": [{"type": "PR78", "acentric": 0.22394}, {"type": "PR78", "acentric": 0.0372}],
               "aresmodel": {"type": "Wilson", "m": [[0.0, -3.4768], [3.5332, 0.0]], "n": [[0.0, 825], [-585, 0.0]]},
               "options": {"s": 2.0, "brule": "Quadratic", "CEoS": -0.52398, "R / J/mol/K": 8.4}
            }
        })"_json;
        auto z = (Eigen::ArrayXd(2) << 0.3, 0.7).finished();
        auto model = make_model(j);
        CHECK(model->get_R(z) == 8.4);
    }
}

TEST_CASE("RK-PR EOS w/ make_model", "[RKPR]"){
    auto j = R"({
        "kind": "RKPRCismondi2005",
        "model": {
           "delta_1": [1.6201],
           "Tcrit / K": [369.89],
           "pcrit / Pa": [4251200.0],
           "k": [1.97064],
           "kmat": [[0.0]],
           "lmat": [[0.0]]
        }
    })"_json;
    auto model = make_model(j);
    
    auto z = (Eigen::ArrayXd(1) << 1.0).finished();
    REQUIRE(model->get_R(z) == 8.31446261815324);
    CHECK(std::isfinite(model->get_Ar00(300, 1, z)));
    CHECK(std::isfinite(model->get_B2vir(300, z)));
    SECTION("funny R"){
        auto j = R"({
            "kind": "RKPRCismondi2005",
            "model": {
               "delta_1": [1.6201],
               "Tcrit / K": [369.89],
               "pcrit / Pa": [4251200.0],
               "k": [1.97064],
               "kmat": [[0.0]],
               "lmat": [[0.0]],
               "R / J/mol/K": 8.4
            }
        })"_json;
        auto model = make_model(j);
        CHECK(model->get_R(z) == 8.4);
    }
}
