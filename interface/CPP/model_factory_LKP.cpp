#include "teqp/cpp/teqpcpp.hpp"
#include "teqp/cpp/deriv_adapter.hpp"
#include "teqp/models/LKP.hpp"

namespace teqp{
    namespace cppinterface{
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_LKP(const nlohmann::json &j){
            return teqp::cppinterface::adapter::make_owned(LKP::make_LKPMix(j));
        }
    }
}
