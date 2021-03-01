#include "teqp/core.hpp"
#include "teqp/models/multifluid.hpp"
#include "teqp/critical_tracing.hpp"
#include <optional>

//#include "autodiff/forward.hpp"
//#include "autodiff/reverse.hpp"

auto build_multifluid_model(const std::vector<std::string>& components, const std::string &coolprop_root, const nlohmann::json &BIPcollection) {
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

template<typename ModelType>
void trace(std::vector<std::string> fluids, const ModelType& model, int i){
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

template<typename Callable, typename Inputs>
auto NewtonRaphson(Callable f, const Inputs &args, double tol) {
    // Jacobian matrix
    Eigen::ArrayXd x = args, r0;
    Eigen::MatrixXd J(args.size(), args.size());
    for (int iter = 0; iter < 30; ++iter){
        r0 = f(x);
        for (auto i = 0; i < args.size(); ++i) {
            auto dri = std::max(1e-3*x[i], 1e-8);
            auto argsplus = x;
            argsplus[i] += dri;
            J.col(i) = (f(argsplus) - r0)/dri; // Forward centered diff to avoid negative composition possibility
        }
        Eigen::ArrayXd v = J.colPivHouseholderQr().solve(-r0.matrix());
        x += v;
        auto err = r0.matrix().norm();
        if (err < tol) {
            break;
        }
    }
    return x;
}

template<typename ModelType>
void trace_arclength(std::vector<std::string> fluids, const ModelType &model, int i) {

    auto rhoc0 = 1.0 / model.redfunc.vc[i];
    auto T = model.redfunc.Tc[i];
    double t = 0.0, dt = 100;
    std::valarray<double> last_drhodt;
    std::valarray<double> rhovec(2); rhovec[i] = { rhoc0 }; rhovec[1-i] = 0.0;
    auto dot = [](const auto& v1, const auto& v2) { return (v1 * v2).sum(); }; 
    auto norm = [](const auto &v){ return sqrt((v*v).sum()); };
    std::string filename = fluids[0] + "_" + fluids[1] + ".csv";
    std::ofstream ofs(filename);
    double c = 1.0;
    ofs << "z0 / mole frac.,rho0 / mol/m^3,rho1 / mol/m^3,T / K,p / Pa,c" << std::endl;
    for (auto iter = 0; iter < 1000; ++iter) {
        auto rhotot = rhovec.sum();
        auto z0 = rhovec[0] / rhotot;

        auto write_line = [&rhovec, &rhotot, &z0, &model, &T, &c, &ofs](){
            std::stringstream out;
            out << z0 << "," << rhovec[0] << "," << rhovec[1] << "," << T << "," << rhotot*model.R*T + get_pr(model, T, rhovec) << "," << c << std::endl;
            std::string sout(out.str());
            ofs << sout;
            std::cout << sout;
        };
        if (iter == 0) {
            write_line();
        }

        auto drhodT = get_drhovec_dT_crit(model, T, rhovec);
        auto dTdt = 1.0 / norm(drhodT);
        auto drhodt = drhodT * dTdt;

        // Flip the sign if the tracing wants to go backwards, or if the first step would take you to negative concentrations
        if (iter == 0 && any(rhovec + c*drhodt*dt < 0)) {
            c *= -1;
        }
        else if (iter > 0 && dot(c*drhodt, last_drhodt) < 0){
            c *= -1;
        }

        rhovec += c*drhodt*dt;
        T += c*dTdt*dt;

        z0 = rhovec[0] / rhovec.sum();

        auto polish_x_resid = [&model, &z0](const auto& x) {
            auto T = x[0];
            std::valarray<double> rhovec = { x[1], x[2] };
            auto z0new = rhovec[0] / rhovec.sum();
            auto derivs = get_derivs(model, T, rhovec);
            // First two are residuals on critical point, third is residual on composition
            return (Eigen::ArrayXd(3) << derivs.tot[2], derivs.tot[3], z0new - z0).finished();
        };
        try {
            Eigen::ArrayXd x0(3); x0 << T, rhovec[0], rhovec[1];
            auto r0 = polish_x_resid(x0);
            auto x = NewtonRaphson(polish_x_resid, x0, 1e-10);
            auto r = polish_x_resid(x);
            Eigen::ArrayXd change = x0 - x;
            if (!std::isfinite(T) || !std::isfinite(x[1]) || !std::isfinite(x[2])) {
                throw std::invalid_argument("Something not finite; aborting polishing");
            }
            T = x[0]; rhovec[0] = x[1]; rhovec[1] = x[2];
        }
        catch (std::exception& e) {
            std::cout << e.what() << std::endl;
        }
        
        rhotot = rhovec.sum(); 
        z0 = rhovec[0] / rhotot;
        
        if (z0 < 0 || z0 > 1) {
            break;
        }
        last_drhodt = c*drhodt;
        write_line();
    }
}

int main(){
    //test_dummy();
    //trace(); 
    std::string coolprop_root = "C:/Users/ihb/Code/CoolProp";
    coolprop_root = "../mycp";
    auto BIPcollection = nlohmann::json::parse(
        std::ifstream(coolprop_root + "/dev/mixtures/mixture_binary_pairs.json")
    );
    std::vector<std::vector<std::string>> pairs = { 
        { "CarbonDioxide", "R1234YF" }, { "CarbonDioxide","R1234ZE(E)" }, { "ETHYLENE","R1243ZF" }, 
        { "R1234YF","R1234ZE(E)" }, { "R134A","R1234YF" }, { "R23","R1234YF" }, 
        { "R32","R1123" }, { "R32","R1234YF" }, { "R32","R1234ZE(E)" }
    };
    for (auto &pp : pairs) {
        using ModelType = decltype(build_multifluid_model(pp, coolprop_root, BIPcollection));
        std::optional<ModelType> model{std::nullopt};
        try {
             model.emplace(build_multifluid_model(pp, coolprop_root, BIPcollection));
        }
        catch (std::exception &e) {
            std::cout << e.what() << std::endl;
            std::cout << pp[0] << "&" << pp[1] << std::endl;
            continue;
        }
        for (int i : {0, 1}){
            trace_arclength(pp, model.value(), i);
        }
    }

    /*auto model = build_multifluid_model({ "methane", "ethane" });
    std::valarray<double> rhovec = { 1.0, 2.0 };
    double T = 300;
    auto alphar = model.alphar(T, rhovec);
    double h = 1e-100;
    auto alpharcom = model.alphar(std::complex<double>(T, h), rhovec).imag()/h;
    MultiComplex<double> Th{{T, h}};
    auto alpharcom2 = model.alphar(Th, rhovec).complex().imag()/h;*/

    //autodiff::dual varT;
    //auto dalphardT = derivative([&model, &rhovec](auto &T){return model.alphar(T, rhovec); }, wrt(varT), at(varT));

    return EXIT_SUCCESS;
}
