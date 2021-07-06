#include "teqp/core.hpp"
#include "teqp/models/multifluid.hpp"

#include <optional>

template<typename J>
void time_calls(const std::string &coolprop_root, const J &BIPcollection) {
    auto model = build_multifluid_model({ "methane", "ethane" }, coolprop_root, BIPcollection);
    Eigen::ArrayXd rhovec(2); rhovec << 1.0, 2.0;
    double T = 300;
    {
        const auto molefrac = (Eigen::ArrayXd(2) << rhovec[0] / rhovec.sum(), rhovec[1] / rhovec.sum()).finished();

        using vd = VirialDerivatives<decltype(model)>;
        auto B12 = vd::get_B12vir(model, T, molefrac);

        using id = IsochoricDerivatives<decltype(model), double, Eigen::ArrayXd>;
        auto mu = id::get_chempotVLE_autodiff(model, T, rhovec);

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
                auto o = vd::template get_Bnvir<3, ADBackends::autodiff>(model, T, molefrac)[3];
            }
            std::cout << alphar << "; 3 derivs" << std::endl;
        }
        {
            Timer t(N);
            for (auto i = 0; i < N; ++i) {
                auto o = vd::template get_Bnvir<4, ADBackends::autodiff>(model, T, molefrac)[4];
            }
            std::cout << alphar << "; 4 derivs" << std::endl;
        }
        {
            Timer t(N);
            for (auto i = 0; i < N; ++i) {
                auto o = vd::template get_Bnvir<5, ADBackends::autodiff>(model, T, molefrac)[5];
            }
            std::cout << alphar << "; 5 derivs" << std::endl;
        }
    }
}

int main(){
   
    std::string coolprop_root = "C:/Users/ihb/Code/CoolProp";
    coolprop_root = "../mycp";
    auto BIPcollection = coolprop_root + "/dev/mixtures/mixture_binary_pairs.json";

    {
        nlohmann::json flags = { {"estimate", true},{"another","key"} };
        auto model = build_multifluid_model({ "CarbonDioxide", "Water" }, coolprop_root, BIPcollection, flags); 
    }

   // // Critical curves
   //{
   //     Timer t(1);
   //     trace_critical_loci(coolprop_root, BIPcollection);
   // }*/

    time_calls(coolprop_root, BIPcollection);
    /*{
        nlohmann::json flags = { {"estimate", true},{"another","key"} };
        auto model = build_multifluid_model({ "Ethane", "R1234ze(E)" }, coolprop_root, BIPcollection, flags);

        nlohmann::json j = { {"betaT", 1.0},{"gammaT", 1.0},{"betaV", 1.0},{"gammaV", 1.0},{"Fij", 0.0} };
        auto mutant = build_mutant(model, j);
    }*/
{
    auto model = build_multifluid_model({ "methane", "ethane" }, coolprop_root, BIPcollection);
    Eigen::ArrayXd rhovec(2); rhovec << 1.0, 2.0;
    double T = 300;
    const auto molefrac = rhovec/rhovec.sum();

    using tdx = TDXDerivatives<decltype(model), double, Eigen::ArrayXd>;
    const auto b = ADBackends::autodiff;
    auto rho = rhovec.sum(); 
    auto alphar = model.alphar(T, rho, rhovec);
    auto Ar01 = tdx::get_Ar01<b>(model, T, rho, molefrac);
    auto Ar10 = tdx::get_Ar10<b>(model, T, rho, molefrac);
    auto Ar02 = tdx::get_Ar02<b>(model, T, rho, molefrac);
    auto Ar11 = tdx::get_Ar11<b>(model, T, rho, molefrac);
    
    //auto Ar11mcx = tdx::get_Ar11<ADBackends::multicomplex>(model, T, rho, molefrac);
    //auto Ar20 = tdx::get_Ar20(model, T, rho, molefrac);
    //using id = IsochoricDerivatives<decltype(model), double, Eigen::ArrayXd>;
    //auto splus = id::get_splus(model, T, rhovec);*/
    
    int ttt = 0;
}
    return EXIT_SUCCESS;
}
