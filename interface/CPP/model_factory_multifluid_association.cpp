#include "teqp/cpp/teqpcpp.hpp"
#include "teqp/cpp/deriv_adapter.hpp"

#include "model_flags.hpp" // Contains (optionally) macros to disable various models

#ifndef DISABLE_MULTIFLUIDASSOCIATION
#include "teqp/models/multifluid_association.hpp"
#endif

namespace teqp{
    namespace cppinterface{
#ifndef DISABLE_MULTIFLUIDASSOCIATION
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_multifluid_association(const nlohmann::json &j){
            return teqp::cppinterface::adapter::make_owned(MultifluidPlusAssociation(j));
        }
#else
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_multifluid_association(const nlohmann::json &){
            throw teqp::NotImplementedError("The multifluid+association model has been disabled");
        }
#endif
    }
}
