#include "teqp/cpp/teqpcpp.hpp"
#include "teqp/cpp/deriv_adapter.hpp"

#include "model_flags.hpp"

#ifndef DISABLE_ADVANCEDMIXCUBIC
#include "teqp/models/cubics/advancedmixing_cubics.hpp"
#endif

namespace teqp{
    namespace cppinterface{
#ifndef DISABLE_ADVANCEDMIXCUBIC
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_advancedPRaEres(const nlohmann::json &j){
            return teqp::cppinterface::adapter::make_owned(make_AdvancedPRaEres(j));
        }
#else
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_advancedPRaEres(const nlohmann::json &){
            throw teqp::NotImplementedError("The advanced cubic mixing rules model has been disabled");
        }
#endif
    }
}
