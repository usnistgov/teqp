#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/benchmark/catch_benchmark_all.hpp>

#include "teqp/derivs.hpp"
#include "teqp/constants.hpp"
#include "teqp/models/LKP.hpp"
#include "teqp/cpp/teqpcpp.hpp"

TEST_CASE("Check LKP", "[LKP]"){
    using namespace teqp::LKP;
    
    SECTION("pure"){
        // methane, check values from TREND
        std::vector<double> Tc_K = {190.564};
        std::vector<double> pc_Pa = {4.5992e6};
        std::vector<double> acentric = {0.011};
        std::vector<std::vector<double>> kmat{{1.0}};
        double R = 8.3144598, Rratio = R/8.31446261815324;
        auto model = LKPMix(Tc_K, pc_Pa, acentric, R, kmat);
        auto z = (Eigen::ArrayXd(1) << 1.0).finished();
        
//        teqp::TDXDerivatives<decltype(model)>::get_Ar00(model, 300.0, 8000.1, z);
        
        nlohmann::json modelspec{
            {"Tcrit / K", Tc_K},
            {"pcrit / Pa", pc_Pa},
            {"acentric", acentric},
            {"R / J/mol/K", R},
            {"kmat", kmat}
        };
//        std::cout << spec.dump(2) << std::endl;
        
        CHECK_NOTHROW(make_LKPMix(modelspec));
        nlohmann::json badspec = modelspec;
        badspec["kmat"] = 4.7;
        CHECK_THROWS(make_LKPMix(badspec));
        
        nlohmann::json spec{
            {"kind", "LKP"},
            {"model", modelspec}
        };
        CHECK_NOTHROW(teqp::cppinterface::make_model(spec));
        auto ptr = teqp::cppinterface::make_model(spec);
        
        struct Point{ double T, rhomolar, alphar_expected; };
        for (auto& pt : std::vector<Point>{
            {250.0, 0.10000, -6.439733446631892E-006*Rratio},
            {250.0, 2000.1, -0.123596981046796*Rratio},
            {250.0, 4000.1, -0.236514837066550*Rratio},
            {250.0, 6000.1, -0.338079606706135*Rratio},
            {250.0, 8000.1, -0.427536610043866*Rratio},
            {300.0, 0.10000, -4.117843314773359E-006*Rratio},
            {300.0, 2000.1, -7.786181697528263E-002*Rratio},
            {300.0, 4000.1, -0.146552001951749*Rratio},
            {300.0, 6000.1, -0.205593594856033*Rratio},
            {300.0, 8000.1, -0.254322813564039*Rratio}
        }){
            auto alphar_actual = teqp::TDXDerivatives<decltype(model)>::get_Ar00(model, pt.T, pt.rhomolar, z);
            auto diff = pt.alphar_expected- alphar_actual;
            CAPTURE(diff);
            CHECK_THAT(alphar_actual, Catch::Matchers::WithinRel(pt.alphar_expected, 1e-11));

            auto alphar_actual_ptr = ptr->get_Ar00(pt.T, pt.rhomolar, z);
            CHECK_THAT(alphar_actual_ptr, Catch::Matchers::WithinRel(pt.alphar_expected, 1e-11));
        }
    }
    
    SECTION("methane + nitrogen mix"){
        // methane + nitrogen, check values from TREND
        std::vector<double> Tc_K = {190.564, 126.192};
        std::vector<double> pc_Pa = {4.5992e6, 3.3958e6};
        std::vector<double> acentric = {0.011, 0.037};
        std::vector<std::vector<double>> kmat{{1.0, 0.977}, {0.977, 1.0}};
        double R = 8.3144598, Rratio = R/8.31446261815324;
        auto model = LKPMix(Tc_K, pc_Pa, acentric, R, kmat);
        
        auto z = (Eigen::ArrayXd(2) << 0.8, 0.2).finished();
        
        auto zbad = (Eigen::ArrayXd(3) << 0.3, 0.3, 0.4).finished();
        CHECK_THROWS(model.alphar(300.0, 8000.0, zbad));
        
//        teqp::TDXDerivatives<decltype(model)>::get_Ar00(model, 300.0, 8000.1, z);
        
        struct Point{ double T, rhomolar, alphar_expected; };
        for (auto& pt : std::vector<Point>{
            {250.0, 0.1, -5.188676044010660E-006*Rratio},
            {250.0, 2000.1, -9.916618378733857E-002*Rratio},
            {250.0, 4000.1, -0.188906817859445*Rratio},
            {250.0, 6000.1, -0.268697996443447*Rratio},
            {250.0, 8000.1, -0.337879558020660*Rratio},
            {300.0, 0.1, -3.183162908790740E-006*Rratio},
            {300.0, 2000.1, -5.951058757516289E-002*Rratio},
            {300.0, 4000.1, -0.110579415736205*Rratio},
            {300.0, 6000.1, -0.152828530523784*Rratio},
            {300.0, 8000.1, -0.185681032885818*Rratio}
        }){
            auto alphar_actual = teqp::TDXDerivatives<decltype(model)>::get_Ar00(model, pt.T, pt.rhomolar, z);
            auto diff = pt.alphar_expected- alphar_actual;
            CAPTURE(diff);
            CHECK_THAT(alphar_actual, Catch::Matchers::WithinRel(pt.alphar_expected,1e-11));
        }
    }
}
TEST_CASE("LKP benchmark"){
    
    // methane + nitrogen, check values from TREND
    std::vector<double> Tc_K = {190.564, 126.192};
    std::vector<double> pc_Pa = {4.5992e6, 3.3958e6};
    std::vector<double> acentric = {0.011, 0.037};
    std::vector<std::vector<double>> kmat{{1.0, 0.977},{0.977, 1.0}};
    double R = 8.3144598;
    auto model = teqp::LKP::LKPMix(Tc_K, pc_Pa, acentric, R, kmat);
    auto z = (Eigen::ArrayXd(2) << 0.8, 0.2).finished();
    nlohmann::json modelspec{
        {"Tcrit / K", Tc_K},
        {"pcrit / Pa", pc_Pa},
        {"acentric", acentric},
        {"R / J/mol/K", R},
        {"kmat", kmat}
    };
    nlohmann::json spec{
        {"kind", "LKP"},
        {"model", modelspec}
    };
    auto ptr = teqp::cppinterface::make_model(spec);
    BENCHMARK("evaluation"){
        return ptr->get_Ar00(300.0, 8000.0, z);
    };
}
