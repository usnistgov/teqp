#include "teqp/cpp/teqpcpp.hpp"
#include "teqp/cpp/deriv_adapter.hpp"
#include "teqp/models/multifluid.hpp"


#include "model_flags.hpp" // Contains (optionally) macros to disable various models

#ifndef DISABLE_AMMONIAWATERTILLNERROTH
#include "teqp/models/ammonia_water.hpp"
#endif

#ifndef DISABLE_MIE
#include "teqp/models/mie/lennardjones.hpp"
#endif

#ifndef DISABLE_ECSHUBERELY1994
#include "teqp/models/ECSHuberEly.hpp"
#endif

namespace teqp{
    namespace cppinterface{
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_multifluid(const nlohmann::json &j){
            return teqp::cppinterface::adapter::make_owned(multifluidfactory(j));
        }
#ifndef DISABLE_ECSHUBERELY1994
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_multifluid_ECS_HuberEly1994(const nlohmann::json &j){
            return teqp::cppinterface::adapter::make_owned(ECSHuberEly::ECSHuberEly1994(j));
        }
#else
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_multifluid_ECS_HuberEly1994(const nlohmann::json &){
            throw teqp::NotImplementedError("The ECS model of Huber and Ely has been disabled");
        }
#endif

#ifndef DISABLE_AMMONIAWATERTILLNERROTH
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_AmmoniaWaterTillnerRoth(){
            return teqp::cppinterface::adapter::make_owned(AmmoniaWaterTillnerRoth());
        }
#else
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_AmmoniaWaterTillnerRoth(){
            throw teqp::NotImplementedError("The ammonia-water model of Tillner-Roth and Friend has been disabled");
        }
#endif

#ifndef DISABLE_MIE
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_LJ126_TholJPCRD2016(){
            return teqp::cppinterface::adapter::make_owned(build_LJ126_TholJPCRD2016());
        }
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_LJ126_KolafaNezbeda1994(){
            return teqp::cppinterface::adapter::make_owned(LJ126KolafaNezbeda1994());
        }
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_LJ126_Johnson1993(){
            return teqp::cppinterface::adapter::make_owned(LJ126Johnson1993());
        }
#else
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_LJ126_TholJPCRD2016(){
            throw teqp::NotImplementedError("The Lennard-Jones 12-6 model of Thol has been disabled");
        }
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_LJ126_KolafaNezbeda1994(){
            throw teqp::NotImplementedError("The Lennard-Jones 12-6 model of Kolafa-Nezbeda has been disabled");
        }
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_LJ126_Johnson1993(){
            throw teqp::NotImplementedError("The Lennard-Jones 12-6 model of Johnson has been disabled");
        }
#endif
    }
}
