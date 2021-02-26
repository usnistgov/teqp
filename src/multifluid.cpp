#include "teqp/core.hpp"
#include "teqp/models/multifluid.hpp"
#include "teqp/critical_tracing.hpp"

//#include "autodiff/forward.hpp"
//#include "autodiff/reverse.hpp"

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
    auto model = build_multifluid_model({ "Methane", "Ethane" });
    auto rhoc0 = 1.0/model.redfunc.vc[0];
    auto T = model.redfunc.Tc[0];
    const auto dT = -1;
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

void trace_arclength() {
    auto model = build_multifluid_model({ "R32", "R1234yf" });
    auto rhoc0 = 1.0 / model.redfunc.vc[1];
    auto T = model.redfunc.Tc[1];
    double t = 0.0, dt = 200;
    std::valarray<double> last_drhodt;
    std::valarray<double> rhovec = { 0.0, rhoc0 };
    auto norm = [](const auto &v){ return sqrt((v*v).sum()); };
    auto dot = [](const auto& v1, const auto& v2) { return (v1*v2).sum(); };
    for (auto iter = 0; iter < 1000; ++iter) {

        auto drhodT = get_drhovec_dT_crit(model, T, rhovec);
        auto dTdt = 1.0 / norm(drhodT);
        auto drhodt = drhodT * dTdt;

        // Flip the sign if the tracing wants to go backwards, or if the first step would take you to negative concentrations
        if (iter > 0 && dot(drhodt, last_drhodt) < 0){
            drhodt *= -1;
            dTdt *= -1;
        }
        else if (iter == 0 && any(rhovec + drhodt * dt < 0)){
            drhodt *= -1;
            dTdt *= -1;
        }

        rhovec += drhodt*dt;
        T += dTdt*dt;
        
        auto rhotot = rhovec.sum(); 
        auto z0 = rhovec[0] / rhotot;
        
        std::cout << z0 << " ," << rhovec[0] << "," << T << "," << rhotot*model.R*T + get_pr(model, T, rhovec) << std::endl;
        if (z0 < 0 || z0 > 1) {
            break;
        }
        last_drhodt = drhodt;
    }
}

int main(){
    //test_dummy();
    //trace();
    trace_arclength();
    auto model = build_multifluid_model({ "methane", "ethane" });
    std::valarray<double> rhovec = { 1.0, 2.0 };
    double T = 300;
    auto alphar = model.alphar(T, rhovec);
    double h = 1e-100;
    auto alpharcom = model.alphar(std::complex<double>(T, h), rhovec).imag()/h;
    MultiComplex<double> Th{{T, h}};
    auto alpharcom2 = model.alphar(Th, rhovec).complex().imag()/h;

    //autodiff::dual varT;
    //auto dalphardT = derivative([&model, &rhovec](auto &T){return model.alphar(T, rhovec); }, wrt(varT), at(varT));

    return EXIT_SUCCESS;
}
