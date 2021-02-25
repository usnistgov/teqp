#pragma once


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