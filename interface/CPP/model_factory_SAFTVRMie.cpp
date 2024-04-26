#include "teqp/cpp/teqpcpp.hpp"
#include "teqp/cpp/deriv_adapter.hpp"
#include "teqp/models/saftvrmie.hpp"

namespace teqp{
    namespace cppinterface{
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_SAFTVRMie(const nlohmann::json &j){
            return teqp::cppinterface::adapter::make_owned(SAFTVRMie::SAFTVRMiefactory(j));
        }
    }
}
