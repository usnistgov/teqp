#include "teqp/cpp/teqpcpp.hpp"
#include "teqp/cpp/deriv_adapter.hpp"
#include "teqp/models/multifluid.hpp"
#include "teqp/models/ammonia_water.hpp"
#include "teqp/models/mie/lennardjones.hpp"
#include "teqp/models/ECSHuberEly.hpp"

namespace teqp{
    namespace cppinterface{
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_multifluid(const nlohmann::json &j){
            return teqp::cppinterface::adapter::make_owned(multifluidfactory(j));
        }
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_multifluid_ECS_HuberEly1994(const nlohmann::json &j){
            return teqp::cppinterface::adapter::make_owned(ECSHuberEly::ECSHuberEly1994(j));
        }
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_AmmoniaWaterTillnerRoth(){
            return teqp::cppinterface::adapter::make_owned(AmmoniaWaterTillnerRoth());
        }
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_LJ126_TholJPCRD2016(){
            return teqp::cppinterface::adapter::make_owned(build_LJ126_TholJPCRD2016());
        }
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_LJ126_KolafaNezbeda1994(){
            return teqp::cppinterface::adapter::make_owned(LJ126KolafaNezbeda1994());
        }
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_LJ126_Johnson1993(){
            return teqp::cppinterface::adapter::make_owned(LJ126Johnson1993());
        }
    }
}
