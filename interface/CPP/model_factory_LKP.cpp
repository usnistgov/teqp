#include "teqp/cpp/teqpcpp.hpp"
#include "teqp/cpp/deriv_adapter.hpp"

#include "model_flags.hpp"

#ifndef DISABLE_LKP
#include "teqp/models/LKP.hpp"
#endif

namespace teqp{
    namespace cppinterface{
#ifndef DISABLE_LKP
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_LKP(const nlohmann::json &j){
            return teqp::cppinterface::adapter::make_owned(LKP::make_LKPMix(j));
        }
#else
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_LKP(const nlohmann::json &){
            throw teqp::NotImplementedError("The multifluid+association model has been disabled");
        }
#endif
    }
}
