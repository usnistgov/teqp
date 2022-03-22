#pragma once

#include "teqp/types.hpp"

namespace teqp {

    namespace reducing {

        inline auto get_BIPdep(const nlohmann::json& collection, const std::vector<std::string>& identifiers, const nlohmann::json& flags) {

            // If force-estimate is provided in flags, the estimation will over-ride the provided model(s)
            if (flags.contains("force-estimate")) {
                std::string scheme = flags["estimate"];
                if (scheme == "Lorentz-Berthelot") {
                    return std::make_tuple(nlohmann::json({
                        {"betaT", 1.0}, {"gammaT", 1.0}, {"betaV", 1.0}, {"gammaV", 1.0}, {"F", 0.0}
                        }), false);
                }
                else {
                    throw std::invalid_argument("estimation scheme is not understood:" + scheme);
                }
            }

            // convert string to upper case
            auto toupper = [](const std::string s) { auto data = s; std::for_each(data.begin(), data.end(), [](char& c) { c = ::toupper(c); }); return data; };

            // First pass, check names
            std::string comp0 = toupper(identifiers[0]);
            std::string comp1 = toupper(identifiers[1]);
            for (auto& el : collection) {
                std::string name1 = toupper(el["Name1"]);
                std::string name2 = toupper(el["Name2"]);
                if (comp0 == name1 && comp1 == name2) {
                    return std::make_tuple(el, false);
                }
                if (comp0 == name2 && comp1 == name1) {
                    return std::make_tuple(el, true);
                }
            }
            // Second pass, check CAS#
            for (auto& el : collection) {
                std::string CAS1 = el["CAS1"];
                std::string CAS2 = el["CAS2"];
                if (identifiers[0] == CAS1 && identifiers[1] == CAS2) {
                    return std::make_tuple(el, false);
                }
                if (identifiers[0] == CAS2 && identifiers[1] == CAS1) {
                    return std::make_tuple(el, true);
                }
            }

            // If estimate is provided in flags, it will be the fallback solution for filling in interaction parameters
            if (flags.contains("estimate")) {
                std::string scheme = flags["estimate"];
                if (scheme == "Lorentz-Berthelot") {
                    return std::make_tuple(nlohmann::json({
                        {"betaT", 1.0}, {"gammaT", 1.0}, {"betaV", 1.0}, {"gammaV", 1.0}, {"F", 0.0}
                        }), false);
                }
                else {
                    throw std::invalid_argument("estimation scheme is not understood:" + scheme);
                }
            }
            else {
                throw std::invalid_argument("Can't match the binary pair for: " + identifiers[0] + "/" + identifiers[1]);
            }
        }

        /// Get the binary interaction parameters for a given binary pair
        inline auto get_binary_interaction_double(const nlohmann::json& collection, const std::vector<std::string>& identifiers, const nlohmann::json& flags, const std::vector<double>& Tc, const std::vector<double>& vc) {
            auto [el, swap_needed] = get_BIPdep(collection, identifiers, flags);

            double betaT, gammaT, betaV, gammaV;
            if (el.contains("betaT") && el.contains("gammaT") && el.contains("betaV") & el.contains("gammaV")) {
                betaT = el["betaT"]; gammaT = el["gammaT"]; betaV = el["betaV"]; gammaV = el["gammaV"];
                // Backwards order of components, flip beta values
                if (swap_needed) {
                    betaT = 1.0 / betaT;
                    betaV = 1.0 / betaV;
                }
            }
            else if (el.contains("xi") && el.contains("zeta")) {
                double xi = el["xi"], zeta = el["zeta"];
                gammaT = 0.5 * (Tc[0] + Tc[1] + xi) / (2 * sqrt(Tc[0] * Tc[1]));
                gammaV = 4.0 * (vc[0] + vc[1] + zeta) / (0.25 * pow(1 / pow(1 / vc[0], 1.0 / 3.0) + 1 / pow(1 / vc[1], 1.0 / 3.0), 3));
                betaT = 1.0;
                betaV = 1.0;
            }
            else {
                throw std::invalid_argument("Could not understand what to do with this binary model specification: " + el.dump());
            }
            return std::make_tuple(betaT, gammaT, betaV, gammaV);
        }

        /// Build the matrices of betaT, gammaT, betaV, gammaV for the multi-fluid model
        template <typename Tcvec, typename vcvec>
        inline auto get_BIP_matrices(const nlohmann::json& collection, const std::vector<std::string>& components, const nlohmann::json& flags, const Tcvec& Tc, const vcvec& vc) {
            Eigen::MatrixXd betaT, gammaT, betaV, gammaV, YT, Yv;
            auto N = components.size();
            betaT.resize(N, N); betaT.setZero();
            gammaT.resize(N, N); gammaT.setZero();
            betaV.resize(N, N); betaV.setZero();
            gammaV.resize(N, N); gammaV.setZero();
            for (auto i = 0; i < N; ++i) {
                for (auto j = i + 1; j < N; ++j) {
                    auto [betaT_, gammaT_, betaV_, gammaV_] = get_binary_interaction_double(collection, { components[i], components[j] }, flags, { Tc[i], Tc[j] }, { vc[i], vc[j] });
                    betaT(i, j) = betaT_;         betaT(j, i) = 1.0 / betaT(i, j);
                    gammaT(i, j) = gammaT_;       gammaT(j, i) = gammaT(i, j);
                    betaV(i, j) = betaV_;         betaV(j, i) = 1.0 / betaV(i, j);
                    gammaV(i, j) = gammaV_;       gammaV(j, i) = gammaV(i, j);
                }
            }
            return std::make_tuple(betaT, gammaT, betaV, gammaV);
        }

        /// Get a tuple of the arrays of Tc (in K) and vc (m^3/mol) for the pure components to be used as reducing parameters
        /// for the reducing function
        /// 
        /// Note: The reducing state of the EOS is used to obtain the values.  Usually but not always the reducing state is the critical point
        inline auto get_Tcvc(const std::vector<nlohmann::json>& pureJSON) {
            Eigen::ArrayXd Tc(pureJSON.size()), vc(pureJSON.size());
            auto i = 0;
            for (auto& j : pureJSON) {
                auto red = j["EOS"][0]["STATES"]["reducing"];
                double Tc_ = red.at("T");
                double rhoc_ = red.at("rhomolar");
                Tc[i] = Tc_;
                vc[i] = 1.0 / rhoc_;
                i++;
            }
            return std::make_tuple(Tc, vc);
        }

        /// Get the matrix F of Fij factors multiplying the departure functions
        inline auto get_F_matrix(const nlohmann::json& collection, const std::vector<std::string>& identifiers, const nlohmann::json& flags) {
            auto N = identifiers.size();
            Eigen::MatrixXd F(N, N);
            for (auto i = 0; i < N; ++i) {
                F(i, i) = 0.0;
                for (auto j = i + 1; j < N; ++j) {
                    auto [el, swap_needed] = get_BIPdep(collection, { identifiers[i], identifiers[j] }, flags);
                    if (el.empty()) {
                        F(i, j) = 0.0;
                        F(j, i) = 0.0;
                    }
                    else {
                        F(i, j) = el["F"];
                        F(j, i) = el["F"];
                    }
                }
            }
            return F;
        }
    }

    class MultiFluidReducingFunction {
    private:
        Eigen::MatrixXd YT, Yv;

        template <typename Num>
        auto cube(Num x) const {
            return forceeval(x * x * x);
        }
        template <typename Num>
        auto square(Num x) const {
            return forceeval(x * x);
        }

    public:
        const Eigen::MatrixXd betaT, gammaT, betaV, gammaV;
        const Eigen::ArrayXd Tc, vc;

        template<typename ArrayLike>
        MultiFluidReducingFunction(
            const Eigen::MatrixXd& betaT, const Eigen::MatrixXd& gammaT,
            const Eigen::MatrixXd& betaV, const Eigen::MatrixXd& gammaV,
            const ArrayLike& Tc, const ArrayLike& vc)
            : betaT(betaT), gammaT(gammaT), betaV(betaV), gammaV(gammaV), Tc(Tc), vc(vc) {

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
        auto Y(const MoleFractions& z, const Eigen::ArrayXd& Yc, const Eigen::MatrixXd& beta, const Eigen::MatrixXd& Yij) const {

            auto N = z.size();
            typename MoleFractions::value_type sum1 = 0.0;
            for (auto i = 0; i < N; ++i) {
                sum1 = sum1 + square(z[i]) * Yc[i];
            }

            typename MoleFractions::value_type sum2 = 0.0;
            for (auto i = 0; i < N - 1; ++i) {
                for (auto j = i + 1; j < N; ++j) {
                    sum2 = sum2 + 2.0 * z[i] * z[j] * (z[i] + z[j]) / (square(beta(i, j)) * z[i] + z[j]) * Yij(i, j);
                }
            }

            return forceeval(sum1 + sum2);
        }

        template<typename MoleFractions> auto get_Tr(const MoleFractions& molefracs) const { return Y(molefracs, Tc, betaT, YT); }
        template<typename MoleFractions> auto get_rhor(const MoleFractions& molefracs) const { return 1.0 / Y(molefracs, vc, betaV, Yv); }
    };

    class MultiFluidInvariantReducingFunction {
    private:
        Eigen::MatrixXd YT, Yv;
        template <typename Num> auto cube(Num x) const { return x * x * x; }
        template <typename Num> auto square(Num x) const { return x * x; }
    public:
        const Eigen::MatrixXd phiT, lambdaT, phiV, lambdaV;
        const Eigen::ArrayXd Tc, vc;

        template<typename ArrayLike>
        MultiFluidInvariantReducingFunction(
            const Eigen::MatrixXd& phiT, const Eigen::MatrixXd& lambdaT,
            const Eigen::MatrixXd& phiV, const Eigen::MatrixXd& lambdaV,
            const ArrayLike& Tc, const ArrayLike& vc)
            : phiT(phiT), lambdaT(lambdaT), phiV(phiV), lambdaV(lambdaV), Tc(Tc), vc(vc) {

            auto N = Tc.size();

            YT.resize(N, N); YT.setZero();
            Yv.resize(N, N); Yv.setZero();
            for (auto i = 0; i < N; ++i) {
                for (auto j = 0; j < N; ++j) {
                    YT(i, j) = sqrt(Tc[i] * Tc[j]);
                    YT(j, i) = sqrt(Tc[i] * Tc[j]);
                    Yv(i, j) = 1.0 / 8.0 * cube(cbrt(vc[i]) + cbrt(vc[j]));
                    Yv(j, i) = 1.0 / 8.0 * cube(cbrt(vc[i]) + cbrt(vc[j]));
                }
            }
        }
        template <typename MoleFractions>
        auto Y(const MoleFractions& z, const Eigen::MatrixXd& phi, const Eigen::MatrixXd& lambda, const Eigen::MatrixXd& Yij) const {
            auto N = z.size();
            typename MoleFractions::value_type sum = 0.0;
            for (auto i = 0; i < N; ++i) {
                for (auto j = 0; j < N; ++j) {
                    auto contrib = z[i] * z[j] * (phi(i, j) + z[j] * lambda(i, j)) * Yij(i, j);
                    sum += contrib;
                }
            }
            return sum;
        }
        template<typename MoleFractions> auto get_Tr(const MoleFractions& molefracs) const { return Y(molefracs, phiT, lambdaT, YT); }
        template<typename MoleFractions> auto get_rhor(const MoleFractions& molefracs) const { return 1.0 / Y(molefracs, phiV, lambdaV, Yv); }
    };


    template<typename... Args>
    class ReducingTermContainer {
    private:
        const std::variant<Args...> term;
        const auto& get_Tc_ref() const { return std::visit([](auto& t) { return std::ref(t.Tc); }, term); }
        const auto& get_vc_ref() const { return std::visit([](auto& t) { return std::ref(t.vc); }, term); }
    public:
        const Eigen::ArrayXd& Tc, & vc;

        template<typename Instance>
        ReducingTermContainer(const Instance& instance) : term(instance), Tc(get_Tc_ref()), vc(get_vc_ref()) {}

        template <typename MoleFractions>
        auto get_Tr(const MoleFractions& molefracs) const {
            return std::visit([&](auto& t) { return t.get_Tr(molefracs); }, term);
        }

        template <typename MoleFractions>
        auto get_rhor(const MoleFractions& molefracs) const {
            return std::visit([&](auto& t) { return t.get_rhor(molefracs); }, term);
        }
    };

    using ReducingFunctions = ReducingTermContainer<MultiFluidReducingFunction, MultiFluidInvariantReducingFunction>;

}; // namespace teqp