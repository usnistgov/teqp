#pragma once

/* 
Implemetations of the canonical cubic equations of state
*/

#include <vector>
#include <variant>
#include <valarray>

#include "teqp/types.hpp"
#include "teqp/constants.hpp"

/**
* \brief The standard alpha function used by Peng-Robinson and SRK
*/
template<typename NumType>
class BasicAlphaFunction {
private:
    NumType Tci, ///< The critical temperature
        mi;  ///< The "m" parameter
public:
    BasicAlphaFunction(NumType Tci, NumType mi) : Tci(Tci), mi(mi) {};

    template<typename TType>
    auto operator () (const TType& T) const {
        return forceeval(powi(forceeval(1.0 + mi * (1.0 - sqrt(T / Tci))), 2));
    }
};

// This could be extended with for instance Twu alpha functions, Mathias-Copeman alpha functions, etc.
using AlphaFunctionOptions = std::variant<BasicAlphaFunction<double>>;

template <typename NumType, typename AlphaFunctions>
class GenericCubic {
protected:
    std::valarray<NumType> ai, bi;
    std::valarray<std::valarray<NumType>> k;
    const NumType Delta1, Delta2, OmegaA, OmegaB;
    const AlphaFunctions alphas;

    template<typename TType, typename IndexType>
    auto get_ai(TType T, IndexType i) const { return ai[i]; }

    template<typename TType, typename IndexType>
    auto get_bi(TType T, IndexType i) const { return bi[i]; }

public:
    GenericCubic(NumType Delta1, NumType Delta2, NumType OmegaA, NumType OmegaB, const std::valarray<NumType>& Tc_K, const std::valarray<NumType>& pc_Pa, const AlphaFunctions& alphas)
        : Delta1(Delta1), Delta2(Delta2), OmegaA(OmegaA), OmegaB(OmegaB), alphas(alphas)
    {
        ai.resize(Tc_K.size());
        bi.resize(Tc_K.size());
        for (auto i = 0; i < Tc_K.size(); ++i) {
            ai[i] = OmegaA * pow(Ru * Tc_K[i], 2) / pc_Pa[i];
            bi[i] = OmegaB * Ru * Tc_K[i] / pc_Pa[i];
        }
        k = std::valarray<std::valarray<NumType>>(std::valarray<NumType>(0.0, Tc_K.size()), Tc_K.size());
    };

    const NumType Ru = get_R_gas<double>(); /// Universal gas constant, exact number

    template<class VecType>
    auto R(const VecType& molefrac) const {
        return Ru;
    }

    template<typename TType, typename CompType>
    auto get_a(TType T, const CompType& molefracs) const {
        std::common_type_t<TType, decltype(molefracs[0])> a_ = 0.0;
        auto ai = this->ai;
        for (auto i = 0; i < molefracs.size(); ++i) {
            auto alphai = forceeval(std::visit([&](auto& t) { return t(T); }, alphas[i]));
            auto ai_ = forceeval(ai[i] * alphai);
            for (auto j = 0; j < molefracs.size(); ++j) {
                auto alphaj = forceeval(std::visit([&](auto& t) { return t(T); }, alphas[j]));
                auto aj_ = ai[j] * alphaj;
                auto aij = forceeval((1 - k[i][j]) * sqrt(ai_ * aj_));
                a_ = a_ + molefracs[i] * molefracs[j] * aij;
            }
        }
        return forceeval(a_);
    }

    template<typename TType, typename CompType>
    auto get_b(TType T, const CompType& molefracs) const {
        std::common_type_t<TType, decltype(molefracs[0])> b_ = 0.0;
        for (auto i = 0; i < molefracs.size(); ++i) {
            b_ = b_ + molefracs[i] * bi[i];
        }
        return forceeval(b_);
    }

    template<typename TType, typename RhoType, typename MoleFracType>
    auto alphar(const TType& T,
        const RhoType& rho,
        const MoleFracType& molefrac) const
    {
        auto b = get_b(T, molefrac);
        auto Psiminus = -log(1.0 - b * rho);
        auto Psiplus = log((Delta1 * b * rho + 1) / (Delta2 * b * rho + 1)) / (b * (Delta1 - Delta2));
        auto val = Psiminus - get_a(T, molefrac) / (Ru * T) * Psiplus;
        return forceeval(val);
    }
};

template <typename TCType, typename PCType, typename AcentricType>
auto canonical_SRK(TCType Tc_K, PCType pc_K, AcentricType acentric) {
    double Delta1 = 1;
    double Delta2 = 0;
    auto m = 0.48 + 1.574 * acentric - 0.176 * acentric * acentric;

    std::vector<AlphaFunctionOptions> alphas;
    for (auto i = 0; i < Tc_K.size(); ++i) {
        alphas.emplace_back(BasicAlphaFunction(Tc_K[i], m[i]));
    }

    // See https://doi.org/10.1021/acs.iecr.1c00847
    double OmegaA = 1.0 / (9.0 * (cbrt(2) - 1));
    double OmegaB = (cbrt(2) - 1) / 3;

    return GenericCubic(Delta1, Delta2, OmegaA, OmegaB, Tc_K, pc_K, std::move(alphas));
}

template <typename TCType, typename PCType, typename AcentricType>
auto canonical_PR(TCType Tc_K, PCType pc_K, AcentricType acentric) {
    double Delta1 = 1+sqrt(2);
    double Delta2 = 1-sqrt(2);
    auto m = acentric*0.0;
    std::vector<AlphaFunctionOptions> alphas; 
    for (auto i = 0; i < Tc_K.size(); ++i) {
        if (acentric[i] < 0.491) {
            m[i] = 0.37464 + 1.54226*acentric[i] - 0.26992*pow(acentric[i], 2);
        }
        else {
            m[i] = 0.379642 + 1.48503*acentric[i] -0.164423*powi(acentric[i], 2) + 0.016666*powi(acentric[i], 3);
        }
        alphas.emplace_back(BasicAlphaFunction(Tc_K[i], m[i]));
    }

    // See https://doi.org/10.1021/acs.iecr.1c00847
    double OmegaA = 0.45723552892138218938;
    double OmegaB = 0.077796073903888455972;

    return GenericCubic(Delta1, Delta2, OmegaA, OmegaB, Tc_K, pc_K, std::move(alphas));
}