#pragma once


namespace teqp {

    /**
    This class holds a lightweight reference to the core parts of the model

    The reducing and departure functions are moved into this class, while the donor class is used for the corresponding states portion
    */
    template<typename DepartureFunction, typename BaseClass>
    class MultiFluidAdapter {

    private:
        std::string meta = "";

    public:
        const BaseClass& base;
        const ReducingFunctions redfunc;
        const DepartureFunction dep;

        template<class VecType>
        auto R(const VecType& molefrac) const { return base.R(molefrac); }

        MultiFluidAdapter(const BaseClass& base, ReducingFunctions&& redfunc, DepartureFunction&& depfunc) : base(base), redfunc(redfunc), dep(depfunc) {};

        /// Store some sort of metadata in string form (perhaps a JSON representation of the model?)
        void set_meta(const std::string& m) { meta = m; }
        /// Get the metadata stored in string form
        auto get_meta() const { return meta; }
        /// Return a binary interaction parameter
        const std::variant<double, std::string> get_BIP(const std::size_t &i, const std::size_t &j, const std::string& key) const{
            if (key == "F" || key == "Fij"){
                auto F = dep.get_F();
                if (0 <= i && i < F.rows() && 0 <= j && j < F.cols()){
                    return F(i,j);
                }
            }
            return redfunc.get_BIP(i, j, key);
        }
        
        template<typename TauType, typename DeltaType, typename MoleFracType>
        auto alphar_taudelta(const TauType& tau,
            const DeltaType& delta,
            const MoleFracType& molefrac) const
        {
            auto val = base.corr.alphar(tau, delta, molefrac) + dep.alphar(tau, delta, molefrac);
            return forceeval(val);
        }

        template<typename TType, typename RhoType, typename MoleFracType>
        auto alphar(const TType& T,
            const RhoType& rho,
            const MoleFracType& molefrac) const
        {
            auto Tred = forceeval(redfunc.get_Tr(molefrac));
            auto rhored = forceeval(redfunc.get_rhor(molefrac));
            auto delta = forceeval(rho / rhored);
            auto tau = forceeval(Tred / T);
            auto val = base.corr.alphar(tau, delta, molefrac) + dep.alphar(tau, delta, molefrac);
            return forceeval(val);
        }
    };

    template<class Model>
    auto build_multifluid_mutant(const Model& model, const nlohmann::json& jj) {

        auto N = model.redfunc.Tc.size();

        // Allocate the matrices of default models and F factors
        Eigen::MatrixXd F(N, N); F.setZero();
        std::vector<std::vector<DepartureTerms>> funcs(N);
        for (auto i = 0; i < N; ++i) { funcs[i].resize(N); }

        // Build the F and departure function matrix
        for (auto i = 0; i < N; ++i) {
            for (auto j = i; j < N; ++j) {
                if (i == j) {
                    funcs[i][i].add_term(NullEOSTerm());
                }
                else {
                    // Extract the given entry
                    auto entry = jj[std::to_string(i)][std::to_string(j)];

                    // Set the entry in the matrix of F and departure functions
                    auto dep = entry.at("departure");
                    auto BIP = entry.at("BIP");
                    F(i, j) = BIP.at("Fij");
                    F(j, i) = F(i, j);
                    funcs[i][j] = build_departure_function(dep);
                    funcs[j][i] = build_departure_function(dep);
                }
            }
        }

        // Determine what sort of reducing function is to be used
        auto get_reducing = [&](const auto& deptype) {
            auto red = model.redfunc;
            auto Tc = red.Tc, vc = red.vc;
            if (deptype == "invariant") {
                using mat = std::decay_t<decltype(MultiFluidInvariantReducingFunction::phiT)>;
                mat phiT = mat::Zero(N, N), lambdaT = mat::Zero(N, N), phiV = mat::Zero(N, N), lambdaV = mat::Zero(N, N);

                for (auto i = 0; i < N; ++i) {
                    for (auto j = i+1; j < N; ++j) {
                        // Extract the given entry
                        auto entry = jj.at(std::to_string(i)).at(std::to_string(j));

                        auto BIP = entry.at("BIP");
                        // Set the reducing function parameters in the copy
                        phiT(i, j) = BIP.at("phiT");
                        phiT(j, i) = phiT(i, j);
                        lambdaT(i, j) = BIP.at("lambdaT");
                        lambdaT(j, i) = -lambdaT(i, j);

                        phiV(i, j) = BIP.at("phiV");
                        phiV(j, i) = phiV(i, j);
                        lambdaV(i, j) = BIP.at("lambdaV");
                        lambdaV(j, i) = -lambdaV(i, j);
                    }
                }
                return ReducingFunctions(MultiFluidInvariantReducingFunction(phiT, lambdaT, phiV, lambdaV, Tc, vc));
            }
            else {
                using mat = std::decay_t<decltype(MultiFluidReducingFunction::betaT)>;
                mat betaT = mat::Zero(N, N), gammaT = mat::Zero(N, N), betaV = mat::Zero(N, N), gammaV = mat::Zero(N, N);

                for (auto i = 0; i < N; ++i) {
                    for (auto j = i+1; j < N; ++j) {
                        // Extract the given entry
                        auto entry = jj.at(std::to_string(i)).at(std::to_string(j));
                        auto BIP = entry.at("BIP");
                        // Set the reducing function parameters in the copy
                        betaT(i, j) = BIP.at("betaT");
                        betaT(j, i) = 1 / betaT(i, j);
                        betaV(i, j) = BIP.at("betaV");
                        betaV(j, i) = 1 / betaV(i, j);
                        gammaT(i, j) = BIP.at("gammaT"); gammaT(j, i) = gammaT(i, j);
                        gammaV(i, j) = BIP.at("gammaV"); gammaV(j, i) = gammaV(i, j);
                    }
                }
                return ReducingFunctions(MultiFluidReducingFunction(betaT, gammaT, betaV, gammaV, Tc, vc));
            }
        };
        std::string deptype = (jj.at("0").at("1").at("BIP").contains("type")) ? jj.at("0").at("1").at("BIP")["type"] : "";
        ReducingFunctions newred = get_reducing(deptype);

        auto newdep = DepartureContribution(std::move(F), std::move(funcs));
        auto mfa = MultiFluidAdapter(model, std::move(newred), std::move(newdep));
        /// Store the model spec in the adapted multifluid class
        mfa.set_meta(jj.dump());
        return mfa;
    }

}
