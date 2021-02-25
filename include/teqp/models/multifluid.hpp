#pragma once

#include "json.hpp"
#include <Eigen/Dense>
#include <fstream>

template<typename EOSCollection>
class CorrespondingStatesContribution {

private:
    const EOSCollection EOSs;
public:
    CorrespondingStatesContribution(EOSCollection&& EOSs) : EOSs(EOSs) {};

    template<typename TauType, typename DeltaType, typename MoleFractions>
    auto alphar(const TauType& tau, const DeltaType& delta, const MoleFractions& molefracs) const {
        using resulttype = decltype(tau* delta* molefracs[0]);
        resulttype alphar = 0.0;
        auto N = molefracs.size();
        for (auto i = 0; i < N; ++i) {
            alphar = alphar + molefracs[i] * EOSs[i].alphar(tau, delta);
        }
        return alphar;
    }
};

template<typename FCollection, typename DepartureFunctionCollection>
class DepartureContribution {

private:
    const FCollection F;
    const DepartureFunctionCollection funcs;
public:
    DepartureContribution(FCollection&& F, DepartureFunctionCollection&& funcs) : F(F), funcs(funcs) {};

    template<typename TauType, typename DeltaType, typename MoleFractions>
    auto alphar(const TauType& tau, const DeltaType& delta, const MoleFractions& molefracs) const {
        using resulttype = decltype(tau* delta* molefracs[0]);
        resulttype alphar = 0.0;
        auto N = molefracs.size();
        for (auto i = 0; i < N; ++i) {
            for (auto j = 0; j < N; ++j) {
                alphar = alphar + molefracs[i] * molefracs[j] * F(i,j) * funcs[i][j].alphar(tau, delta);
            }
        }
        return alphar;
    }
};

template<typename ReducingFunction, typename CorrespondingTerm, typename DepartureTerm>
class MultiFluid {
private:
    const ReducingFunction redfunc;
    const CorrespondingTerm corr;
    const DepartureTerm dep;

public:

    MultiFluid(ReducingFunction&& redfunc, CorrespondingTerm&& corr, DepartureTerm&& dep) : redfunc(redfunc), corr(corr), dep(dep) {};

    template<typename TType, typename RhoType>
    auto alphar(TType T,
        const RhoType& rhovec,
        const std::optional<typename RhoType::value_type> rhotot = std::nullopt) const
    {
        RhoType::value_type rhotot_ = (rhotot.has_value()) ? rhotot.value() : std::accumulate(std::begin(rhovec), std::end(rhovec), (decltype(rhovec[0]))0.0);
        auto molefrac = rhovec / rhotot_;
        auto Tred = redfunc.Tr(molefrac);
        auto rhored = redfunc.rhor(molefrac);
        auto delta = rhotot_ / rhored;
        auto tau = Tred / T;
        using resulttype = decltype(T* rhovec[0]);
        return corr.alphar(tau, delta, molefrac) + dep.alphar(tau, delta, molefrac);
    }
};

class MultiFluidReducingFunction {
private:
    Eigen::MatrixXd betaT, gammaT, betaV, gammaV, YT, Yv;

    template <typename Num>
    auto cube(Num x) {
        return x * x * x;
    }

public:

    template<typename ArrayLike>
    MultiFluidReducingFunction(
        const Eigen::MatrixXd& betaT, const Eigen::MatrixXd& gammaT,
        const Eigen::MatrixXd& betaV, const Eigen::MatrixXd& gammaV,
        const ArrayLike& Tc, const ArrayLike& vc)
        : betaT(betaT), gammaT(gammaT), betaV(betaV), gammaV(gammaV) {

        auto N = Tc.size();

        YT.resize(N, N); YT.setZero();
        Yv.resize(N, N); Yv.setZero();
        for (auto i = 0; i < N; ++i) {
            for (auto j = i + 1; j < N; ++j) {
                YT(i, j) = betaT(i, j) * gammaT(i, j) * sqrt(Tc[i] * Tc[j]);
                YT(j, i) = betaT(j, i) * gammaT(j, i) * sqrt(Tc[i] * Tc[j]);
                Yv(i, j) = 1.0 / 8.0 * betaV(i, j) * gammaV(i, j) * cube(cbrt(vc[i]) + cbrt(vc[j]));
                Yv(j, i) = 1.0 / 8.0 * betaV(j, i) * gammaV(j, i) * cube(cbrt(vc[i]) + cbrt(vc[j]));
            }
        }
    }

    template <typename MoleFractions>
    auto Y(const MoleFractions& z, const Eigen::MatrixXd& Yc, const Eigen::MatrixXd& beta, const Eigen::MatrixXd& Yij) {
        auto sum2 = 0.0;
        auto N = z.size();
        for i in range(0, N - 1) {
            for j in range(i + 1, N) {
                sum2 += 2 * z[i] * z[j] * (z[i] + z[j]) / (beta[i, j] * *2 * z[i] + z[j]) * Yij[i, j];
            }
        }
        return (z * z * Yc).sum() + sum2;
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
            betaT = 1.0 / betaT;
            betaV = 1.0 / betaV;
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
                auto [betaT_, gammaT_, betaV_, gammaV_] = get_binary_interaction_double(collection, { components[i], components[j] });
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
            F(i, i) = 0.0;
            for (auto j = i + 1; j < N; ++j) {
                auto el = get_BIPdep(collection, { components[i], components[j] });
                F(i, j) = el["F"];
                F(j, i) = el["F"];
            }
        }
        return F;
    }
    template<typename MoleFractions> auto get_Tr(const MoleFractions& molefracs) const { return Y(z, Tc, betaT, YT); }
    template<typename MoleFractions> auto get_rhor(const MoleFractions& molefracs) const { return 1.0 / Y(z, vc, betaV, Yv); }
};

class DummyEOS {
public:
    template<typename TType, typename RhoType> auto alphar(TType tau, const RhoType& delta) const { return tau * delta; }
};
class DummyReducingFunction {
public:
    template<typename MoleFractions> auto Tr(const MoleFractions& molefracs) const { return molefracs[0]; }
    template<typename MoleFractions> auto rhor(const MoleFractions& molefracs) const { return molefracs[0]; }
};
auto build_dummy_multifluid_model(const std::vector<std::string>& components) {
    std::vector<DummyEOS> EOSs(2);
    std::vector<std::vector<DummyEOS>> funcs(2); for (auto i = 0; i < funcs.size(); ++i) { funcs[i].resize(funcs.size()); }
    std::vector<std::vector<double>> F(2); for (auto i = 0; i < F.size(); ++i) { F[i].resize(F.size()); }

    struct Fwrapper {
    private: 
        const std::vector<std::vector<double>> F_;
    public:
        Fwrapper(const std::vector<std::vector<double>> &F) : F_(F){};
        auto operator ()(std::size_t i, std::size_t j) const{ return F_[i][j]; }
    };
    auto ff = Fwrapper(F);
    auto redfunc = DummyReducingFunction();
    return MultiFluid(std::move(redfunc), std::move(CorrespondingStatesContribution(std::move(EOSs))), std::move(DepartureContribution(std::move(ff), std::move(funcs))));
}
void test_dummy() {
    auto model = build_dummy_multifluid_model({ "A", "B" });
    std::valarray<double> rhovec = { 1.0, 2.0 };
    auto alphar = model.alphar(300.0, rhovec);
}