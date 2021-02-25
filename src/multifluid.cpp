#include "teqp/core.hpp"
#include "teqp/models/multifluid.hpp"
#include "teqp/critical_tracing.hpp"

auto build_multifluid_model(const std::vector<std::string>& components) {
    using namespace nlohmann;
    std::string coolprop_root = "C:/Users/ihb/Code/CoolProp";
    auto BIPcollection = json::parse(std::ifstream(coolprop_root + "/dev/mixtures/mixture_binary_pairs.json"));

    auto [Tc, vc] = MultiFluidReducingFunction::get_Tcvc(coolprop_root, components);
    auto F = MultiFluidReducingFunction::get_F_matrix(BIPcollection, components);
    auto funcs = get_departure_function_matrix(coolprop_root, BIPcollection, components);
    auto EOSs = get_EOSs(coolprop_root,  components);
    auto [betaT, gammaT, betaV, gammaV] = MultiFluidReducingFunction::get_BIP_matrices(BIPcollection, components);

    auto redfunc = MultiFluidReducingFunction(betaT, gammaT, betaV, gammaV, Tc, vc);
    
    return MultiFluid(
            std::move(redfunc), 
            std::move(CorrespondingStatesContribution(std::move(EOSs))), 
            std::move(DepartureContribution(std::move(F), std::move(funcs)))
    );
}

void trace() {
    auto model = build_multifluid_model({ "methane", "ethane" });
    auto rhoc0 = 1.0/model.redfunc.vc[0];
    auto T = model.redfunc.Tc[0];
    const auto dT = 1;
    std::valarray<double> rhovec = { rhoc0, 0.0 };
    for (auto iter = 0; iter < 1000; ++iter) {
        auto drhovecdT = get_drhovec_dT_crit(model, T, rhovec);
        rhovec += drhovecdT * dT;
        T += dT;
        int rr = 0;
        auto z0 = rhovec[0] / rhovec.sum();
        std::cout << z0 << " ," << rhovec[0] << "," << T << std::endl;
        if (z0 < 0) {
            break;
        }
    }
}

int main(){
    //test_dummy();
    trace();
    auto model = build_multifluid_model({ "methane", "ethane" });
    std::valarray<double> rhovec = { 1.0, 2.0 };
    double T = 300;
    auto alphar = model.alphar(T, rhovec);
    double h = 1e-100;
    auto alpharcom = model.alphar(std::complex<double>(T, h), rhovec).imag()/h;
    MultiComplex<double> Th{{T, h}};
    auto alpharcom2 = model.alphar(Th, rhovec).complex().imag()/h;
    return EXIT_SUCCESS;
}
