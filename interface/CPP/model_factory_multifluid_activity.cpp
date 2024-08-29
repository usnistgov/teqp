#include "teqp/cpp/teqpcpp.hpp"
#include "teqp/cpp/deriv_adapter.hpp"
#include "teqp/models/multifluid/multifluid_activity.hpp"

namespace teqp{
    namespace cppinterface{
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_multifluid_activity(const nlohmann::json &j){
            return teqp::cppinterface::adapter::make_owned(multifluid::multifluid_activity::MultifluidPlusActivity(j));
        }
    }
}
