#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/benchmark/catch_benchmark_all.hpp>
using Catch::Approx;

#include <iostream>
#include <concepts>

#include "teqp/models/multifluid.hpp"
#include "teqp/models/multifluid_mutant.hpp"
#include "teqp/derivs.hpp"
#include "teqp/models/vdW.hpp"

using namespace teqp;

#include "test_common.in"

template<typename Model, typename Scalar, typename VectorType>
inline auto build_alphar_fgradHessian_autodiff(const Model& model, const Scalar& T, const Scalar& rhomolar, const VectorType& molefrac) {
    // Double derivatives in each component's concentration
    // N^N matrix (symmetric)

    dual2nd u; // the output scalar u = f(x), evaluated together with Hessian below
    ArrayXdual g;
    ArrayXdual2nd molefracdual(molefrac.size()); for (auto i = 0; i < molefrac.size(); ++i) { molefracdual[i] = molefrac[i]; }
    auto hfunc = [&model, &T, &rhomolar](const ArrayXdual2nd& molefracdual_) {
        return forceeval(model.alphar(T, rhomolar, molefracdual_));
    };
    // Evaluate the function value u, its gradient, and its Hessian matrix H
    Eigen::MatrixXd H = autodiff::hessian(hfunc, wrt(molefracdual), at(molefracdual), u, g);
    // Remove autodiff stuff from the numerical values
    auto f = getbaseval(u);
    auto gg = g.cast<double>().eval();
    return std::make_tuple(f, gg, H);
}

TEST_CASE("Invalid derivatives", "[compderivs]"){
    auto vdW = vdWEOS1(3, 4);
    double T = 300, rhomolar = 3000;
    auto molefrac = (Eigen::ArrayXd(2) << 0.3, 0.7).finished();
    double tau = 1/T, delta=rhomolar/1;
    // See:  https://github.com/catchorg/Catch2/blob/devel/docs/assertions.md#expressions-with-commas
    auto bad_model = [&](){ return TDXDerivatives<decltype(vdW)>:: get_AtaudeltaXi<1, 1, 1>(vdW, tau, delta, molefrac, 0); };
    CHECK_THROWS_AS(bad_model(), teqp::NotImplementedError);
}

TEST_CASE("Test composition derivatives with get_ArTDXi", "[compderivs]"){
    nlohmann::json spec{
        {"components", {"METHANE", "NITROGEN"}},
        {"root", FLUIDDATAPATH},
        {"BIP", ""},
        {"departure", ""}
    };
    auto model = multifluidfactory(spec);
    const int iT = 0, iD = 0, iXi = 1;
    double T = 300, rhomolar = 3000;
    auto molefrac = (Eigen::ArrayXd(2) << 0.3, 0.7).finished();
    double dx = 1e-7;
    auto molefracp = (Eigen::ArrayXd(2) << 0.3+dx, 0.7-dx).finished();
    auto molefracm = (Eigen::ArrayXd(2) << 0.3-dx, 0.7+dx).finished();
    using TDX = TDXDerivatives<decltype(model)>;
    
    auto Tr = model.redfunc.get_Tr(molefrac), tau = Tr/T;
    auto rhor = model.redfunc.get_rhor(molefrac), delta = rhomolar/rhor;
    
    auto valtd0 = TDX::get_AtaudeltaXi<iT, iD, iXi>(model, tau, delta, molefrac, 0);
    auto valtd1 = TDX::get_AtaudeltaXi<iT, iD, iXi>(model, tau, delta, molefrac, 1);
    double xNdep = valtd0-valtd1;
    
    auto val0 = TDX::get_ATrhoXi<iT, iD, iXi>(model, T, rhomolar, molefrac, 0);
    auto val1 = TDX::get_ATrhoXi<iT, iD, iXi>(model, T, rhomolar, molefrac, 1);
    auto val02 = TDX::get_ATrhoXi<iT, iD, 2>(model, T, rhomolar, molefrac, 0);
    auto val11 = TDX::get_ATrhoXiXj<iT, iD, 1, 1>(model, T, rhomolar, molefrac, 0, 1);
    auto val12 = TDX::get_ATrhoXi<iT, iD, 2>(model, T, rhomolar, molefrac, 1);
    
    auto [f,grad,H] = build_alphar_fgradHessian_autodiff(model, T, rhomolar, molefrac);
    std::cout << "alphar: " << model.alphar(T, rhomolar, molefrac) << std::endl;
    std::cout << val0 << std::endl;
    std::cout << val1 << std::endl;
    std::cout << (model.alphar(T, rhomolar, molefracp)-model.alphar(T, rhomolar, molefracm))/(2*dx) << std::endl;
    std::cout << val0 - val1 << std::endl;
    std::cout << grad << std::endl;
    std::cout << H << std::endl;
}


TEST_CASE("get_AtaudeltaXi with multifluid mutant", "[mutant]") {
    std::string root = FLUIDDATAPATH;
    nlohmann::json flags = { {"estimate", "Lorentz-Berthelot"} };
    auto BIPcollection = root + "/dev/mixtures/mixture_binary_pairs.json";
    auto model = build_multifluid_model({ "R32", "R1234ZEE" }, FLUIDDATAPATH, BIPcollection, flags);
    std::string s0 = R"({"0": {"1": {"BIP": {"betaT": 1.0, "gammaT": 1.0, "betaV": 1.0, "gammaV": 1.0, "Fij": 1.0}, "departure": {"type": "none"}}}})";
    nlohmann::json j = nlohmann::json::parse(s0);
    auto mutant = build_multifluid_mutant(model, j);
    double tau = 1.3, delta = 0.9;
    auto molefrac = (Eigen::ArrayXd(2) << 0.3, 0.7).finished();
    TDXDerivatives<decltype(mutant)>::get_AtaudeltaXi<1, 1, 1>(mutant, tau, delta, molefrac, 0);
}
