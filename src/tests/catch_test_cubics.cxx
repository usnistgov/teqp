
#include <fstream>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

using Catch::Approx;

#include "teqp/models/cubics.hpp"
#include "teqp/derivs.hpp"
#include "teqp/algorithms/VLE.hpp"

#include <boost/numeric/odeint/stepper/euler.hpp>
#include <boost/numeric/odeint/stepper/runge_kutta_cash_karp54.hpp>

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
}

TEST_CASE("Check manual integration of subcritical VLE isotherm for binary mixture", "[cubic][isochoric]")
{
    using namespace boost::numeric::odeint;
    // Values taken from http://dx.doi.org/10.6028/jres.121.011
    std::valarray<double> Tc_K = { 190.564, 154.581},
                         pc_Pa = { 4599200, 5042800},
                      acentric = { 0.011, 0.022};
    auto model = canonical_PR(Tc_K, pc_Pa, acentric);
    const auto N = Tc_K.size();
    using state_type = std::vector<double>; 
    REQUIRE(N == 2);
    auto get_start = [&](double T, auto i) {
        std::valarray<double> Tc_(Tc_K[i], 1), pc_(pc_Pa[i], 1), acentric_(acentric[i], 1);
        auto PR = canonical_PR(Tc_, pc_, acentric_);
        auto [rhoL, rhoV] = PR.superanc_rhoLV(T);
        state_type o(N*2);
        o[i] = rhoL;
        o[i + N] = rhoV;
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
        std::tie(drhovecdpL, drhovecdpV) = get_drhovecdp_Tsat(model, T, rhovecL, rhovecV);
    };
    auto get_p = [&](const state_type& X) {
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
    // Values taken from http://dx.doi.org/10.6028/jres.121.011
    std::valarray<double> Tc_K = { 190.564, 154.581 },
        pc_Pa = { 4599200, 5042800 },
        acentric = { 0.011, 0.022 };
    auto model = canonical_PR(Tc_K, pc_Pa, acentric);
    const auto N = Tc_K.size();
    using state_type = std::valarray<double>;
    REQUIRE(N == 2);
    auto get_start = [&](double T, auto i) {
        std::valarray<double> Tc_(Tc_K[i], 1), pc_(pc_Pa[i], 1), acentric_(acentric[i], 1);
        auto PR = canonical_PR(Tc_, pc_, acentric_);
        auto [rhoL, rhoV] = PR.superanc_rhoLV(T);
        state_type o(N * 2);
        o[i] = rhoL;
        o[i + N] = rhoV;
        return o;
    };
    int i = 1;
    double T = 120;
    std::valarray<double> rhostart_dil = get_start(T, i);
    std::valarray<double> rhostart_notdil = rhostart_dil;
    rhostart_notdil[1-i] += 1e-6;
    rhostart_notdil[1-i+N] += 1e-6;
    auto checker = [](auto & dernotdil, auto &derdil) {
        auto err0 = (std::get<0>(dernotdil).array()/std::get<0>(derdil).array() - 1).cwiseAbs().maxCoeff();
        auto err1 = (std::get<1>(dernotdil).array()/std::get<1>(derdil).array() - 1).cwiseAbs().maxCoeff();
        CAPTURE(err0);
        CAPTURE(err1);
        return err0 < 1e-9 && err1 < 1e-9;
    };

    SECTION("Along isotherm") {
        // Derivative function with respect to p
        auto xprime = [&](const state_type& X) {
            REQUIRE(X.size() % 2 == 0);
            auto N = X.size() / 2;
            // Memory maps into the state vector for inputs and their derivatives
            auto rhovecL = Eigen::Map<const Eigen::ArrayXd>(&(X[0]), N);
            auto rhovecV = Eigen::Map<const Eigen::ArrayXd>(&(X[0]) + N, N);
            return get_drhovecdp_Tsat(model, T, rhovecL, rhovecV);
        };
        auto dernotdil = xprime(rhostart_notdil); 
        auto derdil = xprime(rhostart_dil);
        CHECK(checker(dernotdil, derdil));
    }
    SECTION("Along isobar") {
        // Derivative function with respect to T
        auto xprime = [&](const state_type& X) {
            REQUIRE(X.size() % 2 == 0);
            auto N = X.size() / 2;
            // Memory maps into the state vector for inputs and their derivatives
            auto rhovecL = Eigen::Map<const Eigen::ArrayXd>(&(X[0]), N);
            auto rhovecV = Eigen::Map<const Eigen::ArrayXd>(&(X[0]) + N, N);
            return get_drhovecdT_psat(model, T, rhovecL.eval(), rhovecV.eval());
        };
        auto dernotdil = xprime(rhostart_notdil); 
        auto derdil = xprime(rhostart_dil);
        CHECK(checker(dernotdil, derdil));
    }
}