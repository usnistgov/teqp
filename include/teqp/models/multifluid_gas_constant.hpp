#pragma once

#include "teqp/constants.hpp"

namespace teqp{
namespace multifluid{
namespace gasconstant{

class MoleFractionWeighted {
    const std::vector<double> Rvals;
public:
    
    MoleFractionWeighted(const std::vector<double>& Rvals) : Rvals(Rvals) {};
    
    template<typename MoleFractions>
    auto get_R(const MoleFractions& molefracs) const {
        using resulttype = std::common_type_t<decltype(molefracs[0])>; // Type promotion, without the const-ness
        resulttype out = 0.0;
        auto N = molefracs.size();
        for (auto i = 0; i < N; ++i) {
            out += molefracs[i] * Rvals[i];
        }
        return forceeval(out);
    }
};

class CODATA{
public:
    template<typename MoleFractions>
    auto get_R(const MoleFractions& molefracs) const {
        return get_R_gas<decltype(molefracs[0])>();
    }
};

using GasConstantCalculator = std::variant<MoleFractionWeighted, CODATA>;

}
}
}
