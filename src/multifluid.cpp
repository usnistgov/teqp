#include "teqp/core.hpp"
#include "teqp/models/multifluid.hpp"

auto build_multifluid_model(const std::vector<std::string>& components) {
    using namespace nlohmann;
    std::string coolprop_root = "C:/Users/ihb/Code/CoolProp";
    auto BIPcollection = json::parse(std::ifstream(coolprop_root + "/dev/mixtures/mixture_binary_pairs.json"));
    
    std::vector<std::vector<DummyEOS>> funcs(2); for (auto i = 0; i < funcs.size(); ++i) { funcs[i].resize(funcs.size()); }
    std::vector<DummyEOS> EOSs(components.size());

    auto f = DummyEOS();
    auto fd = f.alphar(1.1, 1.1);
    auto fi = f.alphar(1,1);

    auto [Tc, vc] = MultiFluidReducingFunction::get_Tcvc(coolprop_root, components);
    auto F = MultiFluidReducingFunction::get_F_matrix(BIPcollection, components);
    auto [betaT, gammaT, betaV, gammaV] = MultiFluidReducingFunction::get_BIP_matrices(BIPcollection, components);

    auto redfunc = MultiFluidReducingFunction(betaT, gammaT, betaV, gammaV, Tc, vc);
    
    return MultiFluid(
            std::move(redfunc), 
            std::move(CorrespondingStatesContribution(std::move(EOSs))), 
            std::move(DepartureContribution(std::move(F), std::move(funcs)))
    );
}

int main(){
    test_dummy();
    auto model = build_multifluid_model({ "Methane", "Ethane" });
    std::valarray<double> rhovec = { 1.0, 2.0 };
    auto alphar = model.alphar(300.0, rhovec);
    return EXIT_SUCCESS;
}
