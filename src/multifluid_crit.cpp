
#include "teqp/models/multifluid.hpp"
#include "teqp/algorithms/critical_tracing.hpp"

#include <optional>

using namespace teqp;

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
               auto [Tnew, rhonew] = ct::critical_polish_fixedmolefrac(model, T, rhovec, z0);
               T0 = Tnew;
               rhoc0 = rhovec.sum();
           }
           std::string filename = pp[0] + "_" + pp[1] + ".csv";
           ct::trace_critical_arclength_binary(model, T0, rhovec, filename);
       }
   }
}

int main(){
   
    std::string coolprop_root = "../mycp";
    auto BIPcollection = coolprop_root + "/dev/mixtures/mixture_binary_pairs.json";
    // Critical curves
    Timer t(1);
    trace_critical_loci(coolprop_root, BIPcollection);
    return EXIT_SUCCESS;
}
