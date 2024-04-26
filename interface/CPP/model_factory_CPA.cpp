#include "teqp/cpp/teqpcpp.hpp"
#include "teqp/cpp/deriv_adapter.hpp"
#include "teqp/models/CPA.hpp"

namespace teqp{
    namespace cppinterface{
        using teqp::cppinterface::adapter::make_owned;
    
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_CPA(const nlohmann::json &spec){
            return make_owned(CPA::CPAfactory(spec));
        }
    }
}
