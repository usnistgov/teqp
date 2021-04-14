#define USE_AUTODIFF

#include "teqp/core.hpp"
#include "teqp/models/multifluid.hpp"

#include <optional>

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
        std::optional<ModelType> optmodel{std::nullopt};
        try {
            optmodel.emplace(build_multifluid_model(pp, coolprop_root, BIPcollection));
        }
        catch (std::exception &e) {
            std::cout << e.what() << std::endl;
            std::cout << pp[0] << "&" << pp[1] << std::endl;
            continue;
        }
        for (int i : {0, 1}){
            const auto &model = optmodel.value();
            auto rhoc0 = 1.0 / model.redfunc.vc[i];
            auto T0 = model.redfunc.Tc[i];
            Eigen::ArrayXd rhovec(2); rhovec[i] = { rhoc0 }; rhovec[1L - i] = 0.0;

            using ct = CriticalTracing<ModelType>;

            // Non-analytic terms make it impossible to initialize AT the pure components
            if (pp[0] == "CarbonDioxide" || pp[1] == "CarbonDioxide") {
                if (i == 0) {
                    rhovec[i] *= 0.9999;
                    rhovec[1L - i] = 0.9999;
                }
                else {
                    rhovec[i] *= 1.0001;
                    rhovec[1L - i] = 1.0001;
                }
                double zi = rhovec[i] / rhovec.sum();
                double T = zi * model.redfunc.Tc[i] + (1 - zi) * model.redfunc.Tc[1L - i];
                double z0 = (i == 0) ? zi : 1-zi;
                auto [Tnew, rhonew] = ct::critical_polish_molefrac(model, T, rhovec, z0);
                T0 = Tnew;
                rhoc0 = rhovec.sum();
            }
            std::string filename = pp[0] + "_" + pp[1] + ".csv";
            ct::trace_critical_arclength_binary(model, T0, rhovec, filename);
        }
    }
}

template<typename J>
void time_calls(const std::string &coolprop_root, const J &BIPcollection) {
    auto model = build_multifluid_model({ "methane", "ethane" }, coolprop_root, BIPcollection);
    Eigen::ArrayXd rhovec(2); rhovec << 1.0, 2.0;
    double T = 300;
    {
        const auto molefrac = (Eigen::ArrayXd(2) << rhovec[0] / rhovec.sum(), rhovec[1] / rhovec.sum()).finished();

        using vd = VirialDerivatives<decltype(model)>;
        auto B12 = vd::get_B12vir(model, T, molefrac);

        using id = IsochoricDerivatives<decltype(model)>;
        auto mu = id::get_chempot_autodiff(model, T, rhovec);

        const double rho = rhovec.sum();
        double T = 300.0;
        constexpr int N = 10000;
        volatile double alphar;
        using tdx = TDXDerivatives<decltype(model)>;
        double rrrr = tdx::get_Ar01(model, T, rho, molefrac);
        double rrrr2 = tdx::get_Ar02(model, T, rho, molefrac);
        {
            Timer t(N);
            for (auto i = 0; i < N; ++i) {
                alphar = model.alphar(T, rho, molefrac);
            }
            std::cout << alphar << " function call" << std::endl;
        }
        {
            Timer t(N);
            for (auto i = 0; i < N; ++i) {
                alphar = tdx::get_Ar01<ADBackends::complex_step>(model, T, rho, molefrac);
            }
            std::cout << alphar << "; 1st CSD" << std::endl;
        }
        {
            Timer t(N);
            for (auto i = 0; i < N; ++i) {
                alphar = tdx::get_Ar01<ADBackends::autodiff>(model, T, rho, molefrac);
            }
            std::cout << alphar << "; 1st autodiff::autodiff" << std::endl;
        }
        {
            Timer t(N);
            for (auto i = 0; i < N; ++i) {
                alphar = tdx::get_Ar01<ADBackends::multicomplex>(model, T, rho, molefrac);
            }
            std::cout << alphar << "; 1st MCX" << std::endl;
        }
        {
            Timer t(N);
            for (auto i = 0; i < N; ++i) {
                alphar = tdx::get_Ar02(model, T, rho, molefrac);
            }
            std::cout << alphar << "; 2nd autodiff" << std::endl;
        }
        {
            Timer t(N);
            for (auto i = 0; i < N; ++i) {
                auto o = vd::get_Bnvir<3, ADBackends::autodiff>(model, T, molefrac)[3];
            }
            std::cout << alphar << "; 3 derivs" << std::endl;
        }
        {
            Timer t(N);
            for (auto i = 0; i < N; ++i) {
                auto o = vd::get_Bnvir<4, ADBackends::autodiff>(model, T, molefrac)[4];
            }
            std::cout << alphar << "; 4 derivs" << std::endl;
        }
        {
            Timer t(N);
            for (auto i = 0; i < N; ++i) {
                auto o = vd::get_Bnvir<5, ADBackends::autodiff>(model, T, molefrac)[5];
            }
            std::cout << alphar << "; 5 derivs" << std::endl;
        }
    }
}

int main(){
   
    std::string coolprop_root = "C:/Users/ihb/Code/CoolProp";
    coolprop_root = "../mycp";
    auto BIPcollection = coolprop_root + "/dev/mixtures/mixture_binary_pairs.json";

    // Critical curves
    {
        Timer t(1);
        trace_critical_loci(coolprop_root, BIPcollection);
    }

    //time_calls(coolprop_root, BIPcollection);

{
    auto model = build_multifluid_model({ "methane", "ethane" }, coolprop_root, BIPcollection);
    Eigen::ArrayXd rhovec(2); rhovec << 1.0, 2.0;
    double T = 300;
    const auto molefrac = rhovec/rhovec.sum();

    using tdx = TDXDerivatives<decltype(model)>;
    const auto b = ADBackends::autodiff;
    auto rho = rhovec.sum();
    auto alphar = model.alphar(T, rho, rhovec);
    auto Ar01 = tdx::get_Ar01<b>(model, T, rho, molefrac);
    auto Ar10 = tdx::get_Ar10(model, T, rho, molefrac);
    auto Ar02 = tdx::get_Ar02(model, T, rho, molefrac);
    auto Ar11 = tdx::get_Ar11<b>(model, T, rho, molefrac);
    auto Ar11mcx = tdx::get_Ar11<ADBackends::multicomplex>(model, T, rho, molefrac);
    auto Ar20 = tdx::get_Ar20(model, T, rho, molefrac);
    using id = IsochoricDerivatives<decltype(model)>;
    auto splus = id::get_splus(model, T, rhovec);
    
    int ttt = 0;
}
    return EXIT_SUCCESS;
}
