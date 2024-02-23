#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/benchmark/catch_benchmark_all.hpp>
using Catch::Approx;

#include <iostream>

#include "teqp/cpp/teqpcpp.hpp"
#include "teqp/models/association/association.hpp"

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
    opt.interaction_partners = {{"e", {"H",}}, {"H", {"e",}}};
    
    association::Association a(b_m3mol, beta, epsilon_Jmol, molecule_sites, opt);
    
    auto molefracs = (Eigen::ArrayXd(2) << 0.3, 0.7).finished();
    auto Ngroups = a.mapper.to_siteid.size();
    Eigen::ArrayXd X_init = Eigen::ArrayXd::Ones(Ngroups);
    
    double v = 3.0680691201961814e-5;
    Eigen::ArrayXd X_A = a.successive_substitution(303.15, 1/v, molefracs, X_init);
    CHECK(X_A[0] == Approx(0.06258400385436955));
    CHECK(X_A[3] == Approx(0.10938445109190545));
    
    BENCHMARK("SS"){
        return a.successive_substitution(303.15, 1/v, molefracs, X_init);
    };
    BENCHMARK("alphar"){
        return a.alphar(303.15, 1/v, molefracs);
    };
}

TEST_CASE("Ethanol + water with CPA", "[association]"){
    nlohmann::json water = {
        {"a0i / Pa m^6/mol^2", 0.12277}, {"bi / m^3/mol", 0.0000145}, {"c1", 0.6736}, {"Tc / K", 647.13},
        {"epsABi / J/mol", 16655.0}, {"betaABi", 0.0692}, {"class", "4C"}
    };
    nlohmann::json ethanol = {
        {"a0i / Pa m^6/mol^2", 0.85164}, {"bi / m^3/mol", 0.0491e-3}, {"c1", 0.7502}, {"Tc / K", 513.92},
        {"epsABi / J/mol", 21500.0}, {"betaABi", 0.008}, {"class", "2B"}
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
        {"model", jCPA}
    };
    auto model = teqp::cppinterface::make_model(j, false);

    auto molefracs = (Eigen::ArrayXd(2) << 0.3, 0.7).finished();
    double T = 303.15, rhomolar=1/3.0680691201961814e-5;
    double R = model->get_R(molefracs);
    
    auto alphar = model->get_Ar00(303.15, rhomolar, molefracs);
    CHECK(alphar == Approx(-8.333844120879878));
    
    double p = T*R*rhomolar*(1+model->get_Ar01(T, rhomolar, molefracs));
    CHECK(p == Approx(1e5));
    BENCHMARK("p(T,rho)"){
        return T*R*rhomolar*(1+model->get_Ar01(T, rhomolar, molefracs));
    };
}
