#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/benchmark/catch_benchmark_all.hpp>
using Catch::Matchers::WithinRel;

#include <iostream>

#include "teqp/cpp/teqpcpp.hpp"
#include "teqp/models/association/association.hpp"
#include "teqp/models/CPA.hpp"
#include "teqp/cpp/deriv_adapter.hpp"

using namespace teqp;

TEST_CASE("Test making the indices and D like in Langenbach", "[association]"){
    auto b_m3mol = (Eigen::ArrayXd(2) << 0.0491/1e3, 0.0145/1e3).finished();
    auto beta = (Eigen::ArrayXd(2) << 8e-3, 69.2e-3).finished();
    auto epsilon_Jmol = (Eigen::ArrayXd(2) << 215.00*100, 166.55*100).finished();
    std::vector<std::vector<std::string>> molecule_sites = {{"B"},{"N","N","P"},{"P"}};
    association::AssociationOptions opt;
    opt.interaction_partners = {{"B", {"N", "P", "B"}}, {"N", {"P", "B"}}, {"P",  {"N", "B"}}};
    opt.site_order = {"B","P","N"};
    
    association::Association a(b_m3mol, beta, epsilon_Jmol, molecule_sites, opt);
}

TEST_CASE("Test ethanol + water association", "[association]"){
    auto b_m3mol = (Eigen::ArrayXd(2) << 0.0491/1e3, 0.0145/1e3).finished();
    auto beta = (Eigen::ArrayXd(2) << 8e-3, 69.2e-3).finished();
    auto epsilon_Jmol = (Eigen::ArrayXd(2) << 215.00*100, 166.55*100).finished();
    
    std::vector<std::vector<std::string>> molecule_sites = {{"e", "H"}, {"e", "e", "H", "H"}};
    association::AssociationOptions opt;
    opt.radial_dist = association::radial_dists::CS;
    opt.max_iters = 1000;
    opt.interaction_partners = {{"e", {"H",}}, {"H", {"e",}}};
    
    association::Association a(b_m3mol, beta, epsilon_Jmol, molecule_sites, opt);
    
    auto molefracs = (Eigen::ArrayXd(2) << 0.3, 0.7).finished();
    auto Ngroups = a.mapper.to_siteid.size();
    Eigen::ArrayXd X_init = Eigen::ArrayXd::Ones(Ngroups);
    
    double v = 3.0680691201961814e-5, T = 303.15;
    auto Delta = a.get_Delta(T, 1/v, molefracs);
    CAPTURE(Delta);
    CHECK_THAT(Delta(0,0), WithinRel(5.85623687e-27, 1e-8));
    CHECK_THAT(Delta(0,3), WithinRel(4.26510827e-27, 1e-8));
    CHECK_THAT(Delta(3,3), WithinRel(2.18581242e-27, 1e-8));
    Eigen::ArrayXd X_A = a.successive_substitution(T, 1/v, molefracs, X_init);
    CHECK_THAT(X_A[0], WithinRel(0.06258400385436955, 1e-8));
    CHECK_THAT(X_A[3], WithinRel(0.10938445109190545, 1e-8));
    
    BENCHMARK("SS"){
        return a.successive_substitution(T, 1/v, molefracs, X_init);
    };
    BENCHMARK("alphar"){
        return a.alphar(T, 1/v, molefracs);
    };
}

TEST_CASE("Ethanol with CPA and old class names", "[association]"){
    nlohmann::json ethanol = {
        {"a0i / Pa m^6/mol^2", 0.85164}, {"bi / m^3/mol", 0.0491e-3}, {"c1", 0.7502}, {"Tc / K", 513.92},
        {"epsABi / J/mol", 21500.0}, {"betaABi", 0.008}, {"class", "2B"}
    };
    nlohmann::json jCPA = {
        {"cubic", "SRK"},
        {"radial_dist", "CS"},
        {"pures", {ethanol}},
        {"R_gas / J/mol/K", 8.31446261815324}
    };
    nlohmann::json j = {
        {"kind", "CPA"},
        {"model", jCPA}
    };
    CHECK_NOTHROW(teqp::cppinterface::make_model(j, false));
}

TEST_CASE("Ethanol + water with CPA and old class names", "[association]"){
    nlohmann::json water = {
        {"a0i / Pa m^6/mol^2", 0.12277}, {"bi / m^3/mol", 0.0000145}, {"c1", 0.6736}, {"Tc / K", 647.13},
        {"epsABi / J/mol", 16655.0}, {"betaABi", 0.0692}, {"sites", {"e","e","H","H"}}
    };
    nlohmann::json ethanol = {
        {"a0i / Pa m^6/mol^2", 0.85164}, {"bi / m^3/mol", 0.0491e-3}, {"c1", 0.7502}, {"Tc / K", 513.92},
        {"epsABi / J/mol", 21500.0}, {"betaABi", 0.008}, {"sites", {"e","H"}}
    };
    nlohmann::json jCPA = {
        {"cubic", "SRK"},
        {"radial_dist", "CS"},
//        {"combining", "CR1"},
        {"pures", {ethanol, water}},
        {"R_gas / J/mol/K", 8.31446261815324}
    };
    nlohmann::json j = {
        {"kind", "CPA"},
        {"validate", false},
        {"model", jCPA}
    };
    CHECK_NOTHROW(teqp::cppinterface::make_model(j, false));
}

TEST_CASE("Ethanol + water with CPA against Clapeyron.jl", "[association]"){
    nlohmann::json water = {
        {"a0i / Pa m^6/mol^2", 0.12277}, {"bi / m^3/mol", 0.0000145}, {"c1", 0.6736}, {"Tc / K", 647.13},
        {"epsABi / J/mol", 16655.0}, {"betaABi", 0.0692}, {"sites", {"e","e","H","H"}}
    };
    nlohmann::json ethanol = {
        {"a0i / Pa m^6/mol^2", 0.85164}, {"bi / m^3/mol", 0.0491e-3}, {"c1", 0.7502}, {"Tc / K", 513.92},
        {"epsABi / J/mol", 21500.0}, {"betaABi", 0.008}, {"sites", {"e","H"}}
    };
    nlohmann::json jCPA = {
        {"cubic", "SRK"},
        {"radial_dist", "CS"},
//        {"combining", "CR1"},
        {"options", {}},
        {"pures", {ethanol, water}},
        {"R_gas / J/mol/K", 8.31446261815324}
    };
    nlohmann::json j = {
        {"kind", "CPA"},
        {"validate", false},
        {"model", jCPA}
    };
    auto model = teqp::cppinterface::make_model(j, false);

    auto molefracs = (Eigen::ArrayXd(2) << 0.3, 0.7).finished();
    double T = 303.15, rhomolar=1/3.0680691201961814e-5;
    double R = model->get_R(molefracs);
    
    auto alphar = model->get_Ar00(303.15, rhomolar, molefracs);
    CHECK_THAT(alphar, WithinRel(-8.333844120879878, 1e-8));
    
    double p = T*R*rhomolar*(1+model->get_Ar01(T, rhomolar, molefracs));
    CHECK_THAT(p, WithinRel(1e5, 1e-6));
    BENCHMARK("p(T,rho)"){
        return T*R*rhomolar*(1+model->get_Ar01(T, rhomolar, molefracs));
    };
}

TEST_CASE("Trace ethanol + water isobaric VLE with CPA", "[associationVLE]"){
    nlohmann::json water = {
        {"a0i / Pa m^6/mol^2", 0.12277}, {"bi / m^3/mol", 0.0000145}, {"c1", 0.6736}, {"Tc / K", 647.13},
        {"epsABi / J/mol", 16655.0}, {"betaABi", 0.0692}, {"sites", {"e","e","H","H"}}
    };
    nlohmann::json ethanol = {
        {"a0i / Pa m^6/mol^2", 0.85164}, {"bi / m^3/mol", 0.0491e-3}, {"c1", 0.7502}, {"Tc / K", 513.92},
        {"epsABi / J/mol", 21500.0}, {"betaABi", 0.008}, {"sites", {"e","H"}}
    };
    nlohmann::json options = {
        {"alpha", 0.4999}
    };
    nlohmann::json jCPA = {
        {"cubic", "SRK"},
        {"radial_dist", "CS"},
        {"options", options},
//        {"combining", "CR1"},
        {"pures", {ethanol, water}},
        {"R_gas / J/mol/K", 8.31446261815324}
    };
    
    nlohmann::json j = {
        {"kind", "CPA"},
        {"validate", false},
        {"model", jCPA}
    };
    auto model = teqp::cppinterface::make_model(j, false);
    
    nlohmann::json jeth = {
        {"kind", "CPA"},
        {"validate", false},
        {"model", {
            {"cubic", "SRK"},
            {"radial_dist", "CS"},
            {"pures", {ethanol}},
            {"R_gas / J/mol/K", 8.31446261815324}
        }}
    };
    auto modeleth = teqp::cppinterface::make_model(jeth, false);
    
    double T = 351;
    auto rhoLrhoV = modeleth->pure_VLE_T(T, 15985.159939940693, 35.82761304572266, 30);
    double rhoL = rhoLrhoV[0];
    auto z = (Eigen::ArrayXd(1) << 1.0).finished();
    auto p = rhoL*modeleth->R(z)*T*(1+modeleth->get_Ar01(T, rhoL, z));
    
    auto rhovecL = (Eigen::ArrayXd(2) << rhoLrhoV[0], 0).finished();
    auto rhovecV = (Eigen::ArrayXd(2) << rhoLrhoV[1], 0).finished();
    
    PVLEOptions opt; opt.verbosity = 100;
    auto o = model->trace_VLE_isobar_binary(p, T, rhovecL, rhovecV, opt);
    CHECK(o.size() > 10);
}


TEST_CASE("Benchmark association evaluations", "[associationbench]"){
    
    std::vector<std::vector<std::string>> molecule_sites = {{"e","e","H","H"}};
    
    auto get_canon = [&](){
        auto b_m3mol = (Eigen::ArrayXd(1) << 0.0000145).finished();
        auto beta = (Eigen::ArrayXd(1) << 0.0692).finished();
        auto epsilon_Jmol = (Eigen::ArrayXd(1) <<  16655.0).finished();
        association::AssociationOptions options;
        options.Delta_rule = association::Delta_rules::CR1;
        options.radial_dist = association::radial_dists::CS;
        return association::Association(b_m3mol, beta, epsilon_Jmol, molecule_sites, options);
    };
    auto get_Dufal = [&](){
        association::AssociationOptions options;
        options.Delta_rule = association::Delta_rules::Dufal;
        
        teqp::association::DufalData data;
        auto oneeig = [](double x){ return (Eigen::ArrayXd(1) << x).finished(); };
        double R = constants::R_CODATA2017;
        data.sigma_m = oneeig(3.0555e-10);
        data.epsilon_Jmol = oneeig(418.0*R);
        data.lambda_r = oneeig(35.823);
        data.kmat = build_square_matrix(R"([[0.0]])"_json);
        // Parameters for the associating part
        data.epsilon_HB_Jmol = oneeig(1600.0*R);
        data.K_HB_m3 = oneeig(496.66e-30);
        data.apply_mixing_rules();
        return association::Association(data, molecule_sites, options);
    };
    auto canon = get_canon();
    auto Dufal = get_Dufal();
    
    auto z = (Eigen::ArrayXd(1) << 1.0).finished();
    BENCHMARK("time Delta with canonical"){
        return canon.get_Delta(300.0, 1/3e-5, z);
    };
    BENCHMARK("time Delta with Dufal"){
        return Dufal.get_Delta(300.0, 1/3e-5, z);
    };
    std::cout << canon.get_Delta(300.0, 1/3e-5, z) << std::endl;
    std::cout << Dufal.get_Delta(300.0, 1/3e-5, z) << std::endl;
}

TEST_CASE("Check explicit solutions for association fractions from old and new code","[XA]"){
    double T = 298.15, rhomolar = 1000/0.01813;
    double epsABi = 16655.0, betaABi = 0.0692, bcubic = 0.0000145, RT = 8.31446261815324*T;
    auto molefrac = (Eigen::ArrayXd(1) << 1.0).finished();
    
    // Explicit solution from Huang & Radosz (old-school method)
    auto X_Huang = teqp::CPA::XA_calc_pure(4, teqp::CPA::association_classes::a4C, teqp::CPA::radial_dist::CS, epsABi, betaABi, bcubic, RT, rhomolar, molefrac);
    
    auto b_m3mol = (Eigen::ArrayXd(1) << 0.0145/1e3).finished();
    auto beta = (Eigen::ArrayXd(1) << 69.2e-3).finished();
    auto epsilon_Jmol = (Eigen::ArrayXd(1) << 166.55*100).finished();
    
    std::vector<std::vector<std::string>> molecule_sites = {{"e", "e", "H", "H"}};
    association::AssociationOptions opt;
    opt.radial_dist = association::radial_dists::CS;
    opt.max_iters = 1000;
    opt.allow_explicit_fractions = true;
    opt.interaction_partners = {{"e", {"H",}}, {"H", {"e",}}};
    association::Association aexplicit(b_m3mol, beta, epsilon_Jmol, molecule_sites, opt);
    
    opt.allow_explicit_fractions = false;
    association::Association anotexplicit(b_m3mol, beta, epsilon_Jmol, molecule_sites, opt);
    
    Eigen::ArrayXd X_init = Eigen::ArrayXd::Ones(2);
    
    // Could be short-circuited solution from new derivation, and should be equal to Huang & Radosz
    auto X_newderiv = aexplicit.successive_substitution(T, rhomolar, molefrac, X_init);
    
    // Force the iterative routines to be used as a further sanity check
    auto X_newderiv_iterative = anotexplicit.successive_substitution(T, rhomolar, molefrac, X_init);
    
    CHECK_THAT(X_Huang(0), WithinRel(X_newderiv(0), 1e-10));
    CHECK_THAT(X_Huang(0), WithinRel(X_newderiv_iterative(0), 1e-10));
    
    BENCHMARK("Calling explicit solution"){
        return aexplicit.successive_substitution(T, rhomolar, molefrac, X_init);
    };
    BENCHMARK("Calling non-explicit solution"){
        return anotexplicit.successive_substitution(T, rhomolar, molefrac, X_init);
    };
}
