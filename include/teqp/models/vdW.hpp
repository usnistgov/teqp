#pragma once

#include "teqp/constants.hpp"
#include "teqp/types.hpp"
#include "teqp/exceptions.hpp"

namespace teqp {

/// A (very) simple implementation of the van der Waals EOS
class vdWEOS1 {
private:
    double a, b;
public:
    /// Intializer, taking the a and b constants directly
    vdWEOS1(double a, double b) : a(a), b(b) {};
    
    /// Accessor functions
    double get_a() const{ return a; }
    double get_b() const{ return b; }

    const double Ru = 1.380649e-23 * 6.02214076e23; ///< Exact value, given by k_B*N_A

    /// \brief Get the universal gas constant 
    /// \note Here the real universal gas constant, with no composition dependence
    template<class VecType>
    auto R(const VecType& /*molefrac*/) const { return Ru; }

    /// The evaluation of \f$ \alpha^{\rm r}=a/(RT) \f$
    /// \param T The temperature
    /// \param rhotot The molar density
    /// \param molefrac The mole fractions of each component
    template<typename TType, typename RhoType, typename VecType>
    auto alphar(const TType &T, const RhoType& rhotot, const VecType &molefrac) const {
        return forceeval(-log(1.0 - b * rhotot) - (a / (R(molefrac) * T)) * rhotot);
    }

    /// \brief For testing, provide the pressure explicit form of the EOS
    double p(double T, double v) {
        return Ru*T/(v - b) - a/(v*v);
    }
};

/// A slightly more involved implementation of van der Waals, this time with mixture properties
template <typename NumType>
class vdWEOS {
protected:
    std::valarray<NumType> ai, bi;
    std::valarray<std::valarray<NumType>> k;

    template<typename TType, typename IndexType>
    auto get_ai(TType /*T*/, IndexType i) const { return ai[i]; }

    template<typename TType, typename IndexType>
    auto get_bi(TType /*T*/, IndexType i) const { return bi[i]; }

public:
    /// \brief Initializer, taking the arrays of critical temperatures and pressures
    /// \param Tc_K Array of critical temperatures in Kelvin
    /// \param pc_Pa Array of critical pressures in Pascal
    /// 
    /// \note All interaction parameters are set to default value of zero and cannot currently be tuned
    vdWEOS(const std::valarray<NumType>& Tc_K, const std::valarray<NumType>& pc_Pa)
    {
        if (Tc_K.size() != pc_Pa.size()){
            throw teqp::InvalidArgument("Sizes of Tc_K " + std::to_string(Tc_K.size()) + " and pc_Pa" + std::to_string(pc_Pa.size()) + " do not agree");
        }
        ai.resize(Tc_K.size());
        bi.resize(Tc_K.size());
        for (auto i = 0U; i < Tc_K.size(); ++i) {
            ai[i] = 27.0 / 64.0 * pow(Ru * Tc_K[i], 2) / pc_Pa[i];
            bi[i] = 1.0 / 8.0 * Ru * Tc_K[i] / pc_Pa[i];
        }
        k = std::valarray<std::valarray<NumType>>(std::valarray<NumType>(0.0, Tc_K.size()), Tc_K.size());
    }; 
    
    /// \brief Calculate the a parameter, based on quadratic mixing rules
    /// \param molefracs Array of mole fractions
    template<typename TType, typename CompType>
    auto a(TType /*T*/, const CompType& molefracs) const {
        typename CompType::value_type a_ = 0.0;
        for (auto i = 0; i < molefracs.size(); ++i) {
            for (auto j = 0; j < molefracs.size(); ++j) {
                auto aij = (1 - k[i][j]) * sqrt(this->ai[i] * this->ai[j]);
                a_ = a_ + molefracs[i] * molefracs[j] * aij;
            }
        }
        return forceeval(a_);
    }

    /// \brief Calculate the b parameter, based on linear mixing rules
    /// \param molefracs Array of mole fractions
    template<typename CompType>
    auto b(const CompType& molefracs) const {
        typename CompType::value_type b_ = 0.0;
        for (auto i = 0; i < molefracs.size(); ++i) {
            b_ = b_ + molefracs[i] * bi[i];
        }
        return forceeval(b_);
    }

    const NumType Ru = get_R_gas<double>(); ///< Universal gas constant, exact number

    /// \brief Get the universal gas constant 
    /// Here the real universal gas constant, with no composition dependence
    /// \note The array of mole fractions are ignored, but required to match other function calls
    template<class VecType>
    auto R(const VecType& /*molefrac*/) const {
        return Ru;
    }

    /// \brief The evaluation of \f$ \alpha^{\rm r}=a/(RT) \f$
    /// \param T The temperature
    /// \param rho The molar density
    /// \param molefrac The mole fractions of each component
    template<typename TType, typename RhoType, typename MoleFracType>
    auto alphar(const TType &T,
        const RhoType& rho,
        const MoleFracType &molefrac) const
    {
        if (static_cast<std::size_t>(molefrac.size()) != ai.size()) {
            throw teqp::InvalidArgument("mole fractions must be of size " + std::to_string(ai.size()) + " but are of size " + std::to_string(molefrac.size()));
        }
        auto Psiminus = -log(1.0 - b(molefrac) * rho);
        auto Psiplus = rho;
        auto val = Psiminus - a(T, molefrac) / (Ru * T) * Psiplus;
        return forceeval(val);
    }
};

}; // namespace teqp
