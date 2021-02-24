#pragma once

#include "core.hpp"

/* A (very) simple implementation of the van der Waals EOS*/
class vdWEOS1 {
private:
    double a, b;
    
public:
    vdWEOS1(double a, double b) : a(a), b(b) {};

    const double R = get_R_gas<double>();

    template<typename TType, typename RhoType>
    auto alphar(TType T, const RhoType& rho) const {
        auto rhotot = std::accumulate(std::begin(rho), std::end(rho), (RhoType::value_type)0.0);
        auto Psiminus = -log(1.0 - b * rhotot);
        auto Psiplus = rhotot;
        return Psiminus - a / (R * T) * Psiplus;
    }

    double p(double T, double v) {
        return R*T/(v - b) - a/(v*v);
    }
};

/* A slightly more involved implementation of van der Waals, 
this time with mixture properties */
template <typename NumType>
class vdWEOS {
private:
    std::valarray<NumType> ai, bi;
    std::valarray<std::valarray<NumType>> k;
public:
    vdWEOS(const std::valarray<NumType>& Tc_K, const std::valarray<NumType>& pc_Pa)
    {
        ai.resize(Tc_K.size());
        bi.resize(Tc_K.size());
        for (auto i = 0; i < Tc_K.size(); ++i) {
            ai[i] = 27.0 / 64.0 * pow(R * Tc_K[i], 2) / pc_Pa[i];
            bi[i] = 1.0 / 8.0 * R * Tc_K[i] / pc_Pa[i];
        }
        k = std::valarray<std::valarray<NumType>>(std::valarray<NumType>(0.0, Tc_K.size()), Tc_K.size());
    };

    const NumType R = get_R_gas<double>();

    template<typename TType, typename IndexType> 
    auto get_ai(TType T, IndexType i) const { return ai[i]; }

    template<typename TType, typename IndexType> 
    auto get_bi(TType T, IndexType i) const { return bi[i]; }

    template<typename TType, typename CompType>
    auto a(TType T, const CompType& molefracs) const {
        CompType::value_type a_ = 0.0;
        auto ai = this->ai;
        for (auto i = 0; i < molefracs.size(); ++i) {
            for (auto j = 0; j < molefracs.size(); ++j) {
                auto aij = (1 - k[i][j]) * sqrt(ai[i] * ai[j]);
                a_ += molefracs[i] * molefracs[j] * aij;
            }
        }
        return a_;
    }

    template<typename CompType>
    auto b(const CompType& molefracs) const {
        CompType::value_type b_ = 0.0;
        for (auto i = 0; i < molefracs.size(); ++i) {
            b_ += molefracs[i] * bi[i];
        }
        return b_;
    }

    template<typename TType, typename RhoType>
    auto alphar(TType T,
        const RhoType& rho,
        const std::optional<typename RhoType::value_type> rhotot = std::nullopt) const
    {
        RhoType::value_type rhotot_ = (rhotot.has_value()) ? rhotot.value() : std::accumulate(std::begin(rho), std::end(rho), (decltype(rho[0]))0.0);
        auto molefrac = rho / rhotot_;
        auto Psiminus = -log(1.0 - b(molefrac) * rhotot_);
        auto Psiplus = rhotot_;
        return Psiminus - a(T, molefrac) / (R * T) * Psiplus;
    }
};