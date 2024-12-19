#pragma once

#include "teqp/types.hpp"

namespace teqp {

    namespace reducing {

        inline auto get_BIPdep(const nlohmann::json& collection, const std::vector<std::string>& identifiers, const nlohmann::json& flags) {
            if (!collection.is_array()){
                throw teqp::InvalidArgument("collection provided to get_BIPdep must be an array");
            }
            if (collection.size() > 0 && !collection[0].is_object()){
                throw teqp::InvalidArgument("entries in collection provided to get_BIPdep must be objects");
            }

            // If force-estimate is provided in flags, the estimation will over-ride the provided model(s)
            if (flags.contains("force-estimate")) {
                std::string scheme = flags.at("estimate");
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

            std::string comp0 = toupper(identifiers[0]);
            std::string comp1 = toupper(identifiers[1]);
            // O-th pass, check the hashes
            for (auto& el : collection) {
                if (!el.contains("hash1")){ continue; }
                std::string name1 = toupper(el.at("hash1"));
                std::string name2 = toupper(el.at("hash2"));
                if (comp0 == name1 && comp1 == name2) {
                    return std::make_tuple(el, false);
                }
                if (comp0 == name2 && comp1 == name1) {
                    return std::make_tuple(el, true);
                }
            }
            // First pass, check names
            for (auto& el : collection) {
                std::string name1 = toupper(el.at("Name1"));
                std::string name2 = toupper(el.at("Name2"));
                if (comp0 == name1 && comp1 == name2) {
                    return std::make_tuple(el, false);
                }
                if (comp0 == name2 && comp1 == name1) {
                    return std::make_tuple(el, true);
                }
            }
            // Second pass, check CAS#
            for (auto& el : collection) {
                std::string CAS1 = el.at("CAS1");
                std::string CAS2 = el.at("CAS2");
                if (identifiers[0] == CAS1 && identifiers[1] == CAS2) {
                    return std::make_tuple(el, false);
                }
                if (identifiers[0] == CAS2 && identifiers[1] == CAS1) {
                    return std::make_tuple(el, true);
                }
            }

            // If estimate is provided in flags, it will be the fallback solution for filling in interaction parameters
            if (flags.contains("estimate")) {
                std::string scheme = flags.at("estimate");
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
            if (el.contains("betaT") && el.contains("gammaT") && el.contains("betaV") && el.contains("gammaV")) {
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
            for (auto i = 0U; i < N; ++i) {
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
                auto red = j.at("EOS")[0].at("STATES").at("reducing");
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
            for (auto i = 0U; i < N; ++i) {
                F(i, i) = 0.0;
                for (auto j = i + 1; j < N; ++j) {
                    auto [el, swap_needed] = get_BIPdep(collection, { identifiers[i], identifiers[j] }, flags);
                    if (el.empty()) {
                        F(i, j) = 0.0;
                        F(j, i) = 0.0;
                    }
                    else {
                        F(i, j) = el.at("F");
                        F(j, i) = el.at("F");
                    }
                }
            }
            return F;
        }
    }

    class MultiFluidReducingFunction {
    private:
        Eigen::MatrixXd YT, Yv;

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
                    YT(i, j) = 2.0 * betaT(i, j) * gammaT(i, j) * sqrt(Tc[i] * Tc[j]);
                    YT(j, i) = 2.0 * betaT(j, i) * gammaT(j, i) * sqrt(Tc[i] * Tc[j]);
                    Yv(i, j) = 2.0 * 1.0 / 8.0 * betaV(i, j) * gammaV(i, j) * pow3(cbrt(vc[i]) + cbrt(vc[j]));
                    Yv(j, i) = 2.0 * 1.0 / 8.0 * betaV(j, i) * gammaV(j, i) * pow3(cbrt(vc[i]) + cbrt(vc[j]));
                }
            }
        }

        template <typename MoleFractions>
        auto Y(const MoleFractions& z, const Eigen::ArrayXd& Yc, const Eigen::MatrixXd& beta, const Eigen::MatrixXd& Yij) const {

            auto N = z.size();
            typename MoleFractions::value_type sum1 = 0.0;
            for (auto i = 0U; i < N; ++i) {
                sum1 = sum1 + pow2(z[i]) * Yc[i];
            }

            typename MoleFractions::value_type sum2 = 0.0;
            for (auto i = 0U; i < N - 1; ++i) {
                for (auto j = i + 1; j < N; ++j) {
                    auto den = beta(i, j)*beta(i, j) * z[i] + z[j];
                    if (getbaseval(den) != 0){
                        sum2 = sum2 + z[i] * z[j] * (z[i] + z[j]) / den * Yij(i, j);
                    }
                    else{
                        // constexpr check to abort if trying to do second derivatives
                        // and at least two compositions are zero. This should incur
                        // zero runtime overhead. First derivatives are ok.
                        if constexpr (is_eigen_impl<MoleFractions>::value){
                            using namespace autodiff::detail;
                            constexpr auto isDual_ = isDual<typename MoleFractions::Scalar>;
                            constexpr auto order = NumberTraits<typename MoleFractions::Scalar>::Order;
                            if constexpr (isDual_ && order > 1){
                                throw teqp::InvalidArgument("The multifluid reducing term of GERG does not permit more than one zero composition when taking second composition derivatives with autodiff");
                            }
                        }
                        double beta2 = beta(i,j)*beta(i,j);
                        sum2 = sum2 + Yij(i, j)*(
                             z[i]*z[j] + z[i]*z[i]*(1.0-beta2)
                        );
                    }
                }
            }

            return forceeval(sum1 + sum2);
        }

        template<typename MoleFractions> auto get_Tr(const MoleFractions& molefracs) const { return Y(molefracs, Tc, betaT, YT); }
        template<typename MoleFractions> auto get_rhor(const MoleFractions& molefracs) const { return 1.0 / Y(molefracs, vc, betaV, Yv); }
        
        const auto& get_mat(const std::string& key) const {
            if (key == "betaT"){ return betaT; }
            if (key == "gammaT"){ return gammaT; }
            if (key == "betaV"){ return betaV; }
            if (key == "gammaV"){ return gammaV; }
            throw std::invalid_argument("variable is not understood: " + key);
        }
        auto get_BIP(const std::size_t& i, const std::size_t& j, const std::string& key) const {
            const auto& mat = get_mat(key);
            if (i < static_cast<std::size_t>(mat.rows()) && j < static_cast<std::size_t>(mat.cols())){
                return mat(i,j);
            }
            else{
                throw std::invalid_argument("Indices are out of bounds");
            }
        }
        
    };

    class MultiFluidInvariantReducingFunction {
    private:
        Eigen::MatrixXd YT, Yv;

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
                    Yv(i, j) = 1.0 / 8.0 * pow3(cbrt(vc[i]) + cbrt(vc[j]));
                    Yv(j, i) = 1.0 / 8.0 * pow3(cbrt(vc[i]) + cbrt(vc[j]));
                }
            }
        }
        /// As implemented in Table 7.18 from GERG-2004
        template <typename MoleFractions>
        auto Y(const MoleFractions& z, const Eigen::MatrixXd& phi, const Eigen::MatrixXd& lambda, const Eigen::MatrixXd& Yij) const {
            auto N = z.size();
            typename MoleFractions::value_type sum = 0.0;
            for (auto i = 0U; i < N; ++i) {
                typename MoleFractions::value_type sumj1 = 0.0, sumj2 = 0.0;
                for (auto j = 0U; j < N; ++j) {
                    sumj1 += z[j] * phi(i, j) * Yij(i, j);
                    sumj2 += z[j] * cbrt(lambda(i,j)) * cbrt(Yij(i,j));
                }
                sum += z[i]*(sumj1 + sumj2*sumj2*sumj2);
            }
            return sum;
        }
        template<typename MoleFractions> auto get_Tr(const MoleFractions& molefracs) const { return Y(molefracs, phiT, lambdaT, YT); }
        template<typename MoleFractions> auto get_rhor(const MoleFractions& molefracs) const { return 1.0 / Y(molefracs, phiV, lambdaV, Yv); }
        
        const auto& get_mat(const std::string& key) const {
            if (key == "phiT"){ return phiT; }
            if (key == "lambdaT"){ return lambdaT; }
            if (key == "phiV"){ return phiV; }
            if (key == "lambdaV"){ return lambdaV; }
            throw std::invalid_argument("variable is not understood: " + key);
        }
        auto get_BIP(const std::size_t& i, const std::size_t& j, const std::string& key) const {
            const auto& mat = get_mat(key);
            if (i < static_cast<std::size_t>(mat.rows()) && j < static_cast<std::size_t>(mat.cols())){
                return mat(i,j);
            }
            else{
                throw std::invalid_argument("Indices are out of bounds");
            }
        }
    };


    template<typename... Args>
    class ReducingTermContainer {
    private:
        const std::variant<Args...> term;
        auto get_Tc() const { return std::visit([](const auto& t) { return std::cref(t.Tc); }, term); }
        auto get_vc() const { return std::visit([](const auto& t) { return std::cref(t.vc); }, term); }
    public:
        const Eigen::ArrayXd Tc, vc;

        template<typename Instance>
        ReducingTermContainer(const Instance& instance) : term(instance), Tc(get_Tc()), vc(get_vc()) {}

        template <typename MoleFractions>
        auto get_Tr(const MoleFractions& molefracs) const {
            return std::visit([&](auto& t) { return t.get_Tr(molefracs); }, term);
        }

        template <typename MoleFractions>
        auto get_rhor(const MoleFractions& molefracs) const {
            return std::visit([&](auto& t) { return t.get_rhor(molefracs); }, term);
        }
        
        auto get_BIP(const std::size_t& i, const std::size_t& j, const std::string& key) const {
            return std::visit([&](auto& t) { return t.get_BIP(i, j, key); }, term);
        }
    };

    using ReducingFunctions = ReducingTermContainer<MultiFluidReducingFunction, MultiFluidInvariantReducingFunction>;

}; // namespace teqp
