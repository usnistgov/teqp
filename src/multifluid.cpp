#define USE_AUTODIFF

#include "teqp/core.hpp"
#include "teqp/models/multifluid.hpp"

#include <optional>

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
            J.col(i) = (f(argsplus) - r0)/dri; // Forward diff to avoid negative concentration possibility
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
void trace_arclength(std::vector<std::string> fluids, const ModelType &model, std::size_t i) {

    auto rhoc0 = 1.0 / model.redfunc.vc[i];
    auto T = model.redfunc.Tc[i];
    double t = 0.0, dt = 100;
    std::valarray<double> last_drhodt;
    std::valarray<double> rhovec(2); rhovec[i] = { rhoc0 }; rhovec[1L-i] = 0.0;

    // Non-analytic terms make it impossible to initialize AT the pure components
    if (fluids[0] == "CarbonDioxide" || fluids[1] == "CarbonDioxide"){
        if (i == 0) {
            rhovec[i] *= 0.9999;
            rhovec[1L - i] = 0.9999;
        }
        else {
            rhovec[i] *= 1.0001;
            rhovec[1L-i] = 1.0001;
        }
        double zi = rhovec[i]/rhovec.sum();
        T = zi* model.redfunc.Tc[i] + (1-zi)* model.redfunc.Tc[1L-i];
    }
    auto dot = [](const auto& v1, const auto& v2) { return (v1 * v2).sum(); }; 
    auto norm = [](const auto &v){ return sqrt((v*v).sum()); };
    std::string filename = fluids[0] + "_" + fluids[1] + ".csv";
    std::ofstream ofs(filename);
    double c = 1.0;
    ofs << "z0 / mole frac.,rho0 / mol/m^3,rho1 / mol/m^3,T / K,p / Pa,c" << std::endl;
    for (auto iter = 0; iter < 1000; ++iter) {
        auto rhotot = rhovec.sum();
        auto z0 = rhovec[0] / rhotot;

        if (fluids[0] == "CarbonDioxide" || fluids[1] == "CarbonDioxide") {
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
                throw;
                std::cout << e.what() << std::endl;
            }
        }

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

        auto eval = [](const auto& ex){ return std::valarray<bool>(ex); };

        // Flip the sign if the tracing wants to go backwards, or if the first step would take you to negative concentrations
        if (iter == 0 && any(eval((rhovec + c*drhodt*dt) < 0))) {
            c *= -1;
        }
        else if (iter > 0 && dot(std::valarray<double>(c*drhodt), last_drhodt) < 0){
            c *= -1;
        }

        rhovec += c*drhodt*dt;
        T += c*dTdt*dt;

        z0 = rhovec[0] / rhovec.sum();
        if (z0 < 0 || z0 > 1) {
            break;
        }

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
        //write_line();
    }
}

class Timer {
private:
    int N;
    decltype(std::chrono::steady_clock::now()) tic;
public:
    Timer(int N) : N(N), tic(std::chrono::steady_clock::now()){}
    ~Timer() {
        auto elap = std::chrono::duration<double>(std::chrono::steady_clock::now()-tic).count();
        std::cout << elap/N*1e6 << " us/call" << std::endl;
    }
};

void trace_critical_loci(const std::string &coolprop_root, const nlohmann::json &BIPcollection) {
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
}

int main(){
   
    std::string coolprop_root = "C:/Users/ihb/Code/CoolProp";
    coolprop_root = "../mycp";
    auto BIPcollection = nlohmann::json::parse(
        std::ifstream(coolprop_root + "/dev/mixtures/mixture_binary_pairs.json")
    );

    // Critical curves
    {
        Timer t(1);
        trace_critical_loci(coolprop_root, BIPcollection);
    }

{
    auto model = build_multifluid_model({ "methane", "ethane" }, coolprop_root, BIPcollection);
    std::valarray<double> rhovec = { 1.0, 2.0 };
    double T = 300;
    {
        const std::valarray<double> molefrac = { rhovec[0]/rhovec.sum(), rhovec[1]/rhovec.sum() };
        const double rho = rhovec.sum();
        volatile double T = 300.0;
        constexpr int N = 10000;
        volatile double alphar;
        double rrrr = get_Ar01(model, T, rho, molefrac);
        double rrrr2 = get_Ar02(model, T, rho, molefrac);
        {
            Timer t(N);
            for (auto i = 0; i < N; ++i){
                alphar = model.alphar(T, rho, molefrac);
            }
            std::cout << alphar << std::endl;
        }
        {
            Timer t(N);
            for (auto i = 0; i < N; ++i) {
                alphar = get_Ar01<ADBackends::complex_step>(model, T, rho, molefrac);
            }
            std::cout << alphar << "; 1st CSD" << std::endl;
        }
        {
            Timer t(N);
            for (auto i = 0; i < N; ++i) {
                alphar = get_Ar01<ADBackends::autodiff>(model, T, rho, molefrac);
            }
            std::cout << alphar << "; 1st autodiff::autodiff" << std::endl;
        }
        {
            Timer t(N);
            for (auto i = 0; i < N; ++i) {
                alphar = get_Ar01<ADBackends::multicomplex>(model, T, rho, molefrac);
            }
            std::cout << alphar << "; 1st MCX" << std::endl;
        }
        {
            Timer t(N);
            for (auto i = 0; i < N; ++i) {
                alphar = get_Ar02(model, T, rho, molefrac);
            }
            std::cout << alphar << std::endl;
        }
    }

    auto alphar = model.alphar(T, rhovec);
    auto Ar01 = get_Ar01(model, T, rhovec);
    auto Ar10 = get_Ar10(model, T, rhovec);
    auto splus = get_splus(model, T, rhovec);

    std::valarray<double> molefrac = { 1.0/3.0, 2.0/3.0 };
    auto B2 = get_B2vir(model, T, molefrac);

    std::valarray<double> dilrho = 0.00000000001*molefrac;
    auto B2other = get_Ar01(model, T, dilrho)/dilrho.sum();

    std::valarray<double> zerorho = 0.0*rhovec;
    auto Ar01dil = get_Ar01(model, T, zerorho);
    
    int ttt =0 ;
}
    return EXIT_SUCCESS;
}
