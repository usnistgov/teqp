#pragma once

#include "teqp/types.hpp"

#include "teqp/models/association/association.hpp"
#include "teqp/models/multifluid.hpp"

namespace teqp{

class MultifluidPlusAssociation{
private:
    const decltype(multifluidfactory(nlohmann::json{})) m_multifluid;
    const association::Association m_association;
public:
    MultifluidPlusAssociation(const nlohmann::json &spec) :
        m_multifluid(multifluidfactory(spec.at("multifluid"))),
        m_association(association::Association::factory(spec.at("association").at("model"))){}
    
    const auto& get_association() const { return m_association; }
    
    template<class VecType>
    auto R(const VecType& molefrac) const {
        return get_R_gas<decltype(molefrac[0])>();
    }
    
    template <typename TType, typename RhoType, typename MoleFractions>
    auto alphar(const TType& T, const RhoType& rho, const MoleFractions& molefrac) const {
        return forceeval(
            m_multifluid.alphar(T, rho, molefrac)
            + m_association.alphar(T, rho, molefrac)
        );
    }
};

}; // namespace teqp
