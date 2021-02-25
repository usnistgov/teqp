#include "teqp/core.hpp"
#include <fstream>
#include <Eigen/Dense>

#include "json.hpp"
#include "teqp/models/multifluid.hpp"

template <typename Num>
auto cube(Num x) {
    return x*x*x;
}

class MultiFluidReducingFunction {
private:
    Eigen::MatrixXd betaT, gammaT, betaV, gammaV, YT, Yv;

public:

    template<typename ArrayLike>
    MultiFluidReducingFunction(
        const Eigen::MatrixXd &betaT, const Eigen::MatrixXd& gammaT, 
        const Eigen::MatrixXd& betaV, const Eigen::MatrixXd& gammaV, 
        const ArrayLike&Tc, const ArrayLike& vc)
      : betaT(betaT), gammaT(gammaT), betaV(betaV), gammaV(gammaV) {

        auto N = Tc.size();

        YT.resize(N,N); YT.setZero();
        Yv.resize(N, N); Yv.setZero();
        for (auto i = 0; i < N; ++i) {
            for (auto j = i+1; j < N; ++j) {
                YT(i, j) = betaT(i, j)*gammaT(i, j)*sqrt(Tc[i]*Tc[j]);
                YT(j, i) = betaT(j, i)*gammaT(j, i)*sqrt(Tc[i]*Tc[j]);
                Yv(i, j) = 1.0/8.0*betaV(i, j)*gammaV(i, j)*cube(cbrt(vc[i]) + cbrt(vc[j]));
                Yv(j, i) = 1.0/8.0*betaV(j, i)*gammaV(j, i)*cube(cbrt(vc[i]) + cbrt(vc[j]));
            }
        }
    }

    template <typename MoleFractions>
    auto Y(const MoleFractions &z, const Eigen::MatrixXd &Yc, const Eigen::MatrixXd &beta, const Eigen::MatrixXd &Yij){
        auto sum2 = 0.0;
        auto N = z.size();
        for i in range(0, N - 1){
            for j in range(i + 1, N){
                sum2 += 2*z[i]*z[j]*(z[i] + z[j])/(beta[i, j]**2*z[i] + z[j])*Yij[i, j];
            }
        }
        return (z*z*Yc).sum() + sum2;
    }

    static auto get_BIPdep(const nlohmann::json& collection, const std::vector<std::string>& components) {
        for (auto& el : collection) {
            if (components[0] == el["Name1"] && components[1] == el["Name2"]) {
                return el;
            }
            if (components[0] == el["Name2"] && components[1] == el["Name1"]) {
                return el;
            }
        }
    }
    static auto get_binary_interaction_double(const nlohmann::json& collection, const std::vector<std::string>& components) {
        auto el = get_BIPdep(collection, components);
        
        double betaT = el["betaT"], gammaT = el["gammaT"], betaV = el["betaV"], gammaV = el["gammaV"];
        // Backwards order of components, flip beta values
        if (components[0] == el["Name2"] && components[1] == el["Name1"]) {
            betaT = 1.0/betaT;
            betaV = 1.0/betaV;
        }
        return std::make_tuple(betaT, gammaT, betaV, gammaV);
    }
    static auto get_BIP_matrices(const nlohmann::json& collection, const std::vector<std::string>& components) {
        Eigen::MatrixXd betaT, gammaT, betaV, gammaV, YT, Yv;
        auto N = components.size();
        betaT.resize(N, N); betaT.setZero();
        gammaT.resize(N, N); gammaT.setZero();
        betaV.resize(N, N); betaV.setZero();
        gammaV.resize(N, N); gammaV.setZero();
        for (auto i = 0; i < N; ++i) {
            for (auto j = i + 1; j < N; ++j) {
                auto [betaT_, gammaT_, betaV_, gammaV_] = get_binary_interaction_double(collection, {components[i], components[j]});
                betaT(i, j) = betaT_;         betaT(j, i) = 1.0 / betaT(i, j);
                gammaT(i, j) = gammaT_;       gammaT(j, i) = gammaT(i, j);
                betaV(i, j) = betaV_;         betaV(j, i) = 1.0 / betaV(i, j);
                gammaV(i, j) = gammaV_;       gammaV(j, i) = gammaV(i, j);
            }
        }
        return std::make_tuple(betaT, gammaT, betaV, gammaV);
    }
    static auto get_Tcvc(const std::string& coolprop_root, const std::vector<std::string>& components) {
        std::vector<double> Tc, vc;
        using namespace nlohmann;
        for (auto& c : components) {
            auto j = json::parse(std::ifstream(coolprop_root + "/dev/fluids/" + c + ".json"));
            auto red = j["EOS"][0]["STATES"]["reducing"];
            double Tc_ = red["T"];
            double rhoc_ = red["rhomolar"];
            Tc.push_back(Tc_);
            vc.push_back(1.0 / rhoc_);
        }
        return std::make_tuple(Tc, vc);
    }
    static auto get_F_matrix(const nlohmann::json& collection, const std::vector<std::string>& components) {
        Eigen::MatrixXd F(components.size(), components.size());
        auto N = components.size();
        for (auto i = 0; i < N; ++i) { 
            F(i,i) = 0.0;
            for (auto j = i+1; j < N; ++j) {
                auto el = get_BIPdep(collection, {components[i], components[j]});
                F(i,j) = el["F"];
                F(j,i) = el["F"];
            }
        }
        return F;
    }
    template<typename MoleFractions> auto get_Tr(const MoleFractions& molefracs) const { return Y(z, Tc, betaT, YT); }
    template<typename MoleFractions> auto get_rhor(const MoleFractions& molefracs) const { return 1.0 / Y(z, vc, betaV, Yv);  }
};


auto build_multifluid_model(const std::vector<std::string>& components) {
    using namespace nlohmann;
    std::string coolprop_root = "C:/Users/ihb/Code/CoolProp";
    auto BIPcollection = json::parse(std::ifstream(coolprop_root + "/dev/mixtures/mixture_binary_pairs.json"));
    
    std::vector<std::vector<DummyEOS>> funcs(2); for (auto i = 0; i < funcs.size(); ++i) { funcs[i].resize(funcs.size()); }
    
    std::vector<DummyEOS> EOSs(components.size());

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
    
    return EXIT_SUCCESS;
}
