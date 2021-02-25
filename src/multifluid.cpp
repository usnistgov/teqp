#include "teqp/core.hpp"
#include "teqp/models/multifluid.hpp"

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

int main(){
    //test_dummy();
    auto model = build_multifluid_model({ "methane", "ethane" });
    std::valarray<double> rhovec = { 1.0, 2.0 };
    auto alphar = model.alphar(300.0, rhovec);
    return EXIT_SUCCESS;
}
