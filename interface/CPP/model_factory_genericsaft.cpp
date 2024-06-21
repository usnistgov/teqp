#include "teqp/cpp/teqpcpp.hpp"
#include "teqp/cpp/deriv_adapter.hpp"
#include "teqp/models/saft/genericsaft.hpp"

namespace teqp{
    namespace cppinterface{
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_genericSAFT(const nlohmann::json &spec){
            return teqp::cppinterface::adapter::make_owned(saft::genericsaft::GenericSAFT(spec));
        }
    }
}
