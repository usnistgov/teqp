#include "teqp/cpp/teqpcpp.hpp"
#include "teqp/cpp/deriv_adapter.hpp"

#include "model_flags.hpp" // Contains (optionally) macros to disable various models

#ifndef DISABLE_SAFTVRMIE
#include "teqp/models/saftvrmie.hpp"
#endif

namespace teqp{
    namespace cppinterface{
#if defined(DISABLE_SAFTVRMIE)
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_SAFTVRMie(const nlohmann::json &){
            throw teqp::NotImplementedError("The SAFTVRMie model has been disabled");
        }
#else
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_SAFTVRMie(const nlohmann::json &j){
            return teqp::cppinterface::adapter::make_owned(SAFTVRMie::SAFTVRMiefactory(j));
        }
#endif
    }
}
