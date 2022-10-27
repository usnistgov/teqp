#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark_all.hpp>

#include "teqp/models/multifluid.hpp"

#include "teqp/derivs.hpp"
#include "teqp/ideal_eosterms.hpp"

using namespace teqp;

template<int Nderivsmax, int k>
class DerivativeHolderSquare{
    
public:
    Eigen::Array<double, Nderivsmax+1, Nderivsmax+1> derivs;
    
    template<typename Model, typename Scalar, typename VecType>
    DerivativeHolderSquare(const Model& model, const Scalar& T, const Scalar& rho, const VecType& z) {
        using tdx = TDXDerivatives<decltype(model), Scalar, VecType>;
        static_assert(Nderivsmax == 2, "It's gotta be 2");
        AlphaCallWrapper<k, Model> wrapper(model);
        
        auto AX02 = tdx::template get_Agen0n<2>(wrapper, T, rho, z);
        derivs(0, 0) = AX02[0];
        derivs(0, 1) = AX02[1];
        derivs(0, 2) = AX02[2];
        
        auto AX20 = tdx::template get_Agenn0<2>(wrapper, T, rho, z);
        derivs(0, 0) = AX20[0];
        derivs(1, 0) = AX20[1];
        derivs(2, 0) = AX20[2];
        
        derivs(1, 1) = tdx::template get_Agenxy<1,1>(wrapper, T, rho, z);
    }
};

TEST_CASE("multifluid derivatives", "[mf]")
{
    std::vector<std::string> names = { "Propane" };
    auto model = build_multifluid_model(names, "../mycp");

    double T = 300, rho = 2;
    Eigen::ArrayX<double> z(1); z.fill(1.0);
    
    auto json = nlohmann::json::parse(model.get_meta());
    auto jig = convert_CoolProp_idealgas(json.at("pures")[0].dump(), 0);
    nlohmann::json jigs = nlohmann::json::array(); jigs.push_back(jig);
    auto aig = teqp::IdealHelmholtz(jigs);
    
    BENCHMARK("alphar") {
        return model.alphar(T, rho, z);
    };
    
    BENCHMARK("All residual derivatives needed for first derivatives of h,s,u,p w.r.t. T&rho") {
        return DerivativeHolderSquare<2,0>(model, T, rho, z).derivs;
    };
    
    BENCHMARK("All ideal-gas derivatives needed for first derivatives of h,s,u,p w.r.t. T&rho") {
        return DerivativeHolderSquare<2,1>(aig, T, rho, z).derivs;
    };
}
