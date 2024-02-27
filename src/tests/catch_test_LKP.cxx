#include "teqp/models/LKP.hpp"

TEST_CASE("Check LKP", "[LKP]"){
    using namespace teqp::LKP;
    // THF + water
    std::vector<double> Tc_K = { 540.2, 647.096 };
    std::vector<double> pc_Pa = { 5304.44e3, 22064.0e3 };
    std::vector<double> acentric = { 0.234, 0.3443 };
    std::vector<std::vector<double>> kmat = {{1,1}, {1,1}};
    auto model = LKPMix(Tc_K, pc_Pa, acentric, kmat);
    auto T = 303.15, rhomolar = 15000.0;
    auto z = (Eigen::ArrayXd(2) << 0.3, 0.7).finished();
    auto Ar01 = TDXDerivatives<decltype(model)>::get_Ar01(model, T, rhomolar, z);
}
