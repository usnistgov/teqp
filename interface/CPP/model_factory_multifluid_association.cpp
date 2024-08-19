#include "teqp/cpp/teqpcpp.hpp"
#include "teqp/cpp/deriv_adapter.hpp"
#include "teqp/models/multifluid_association.hpp"

namespace teqp{
    namespace cppinterface{
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_multifluid_association(const nlohmann::json &j){
            return teqp::cppinterface::adapter::make_owned(MultifluidPlusAssociation(j));
        }
    }
}
