#include "teqp/cpp/teqpcpp.hpp"
#include "teqp/cpp/deriv_adapter.hpp"

#include "model_flags.hpp"

#ifndef DISABLE_MULTIFLUIDACTIVITY
#include "teqp/models/multifluid/multifluid_activity.hpp"
#endif

namespace teqp{
    namespace cppinterface{
#ifndef DISABLE_MULTIFLUIDACTIVITY
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_multifluid_activity(const nlohmann::json &j){
            return teqp::cppinterface::adapter::make_owned(multifluid::multifluid_activity::MultifluidPlusActivity(j));
        }
#else
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_multifluid_activity(const nlohmann::json &){
            throw teqp::NotImplementedError("The multifluid+activity model has been disabled");
        }
#endif
    }
}
