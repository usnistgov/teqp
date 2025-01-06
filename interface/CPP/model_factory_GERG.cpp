#include "teqp/cpp/teqpcpp.hpp"
#include "teqp/cpp/deriv_adapter.hpp"

#include "model_flags.hpp" // Contains (optionally) macros to disable various models

#ifndef DISABLE_GERG200X
#include "teqp/models/GERG/GERG.hpp"
#endif

namespace teqp{
    namespace cppinterface{
#ifndef DISABLE_GERG200X
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_GERG2004resid(const nlohmann::json &spec){
            return teqp::cppinterface::adapter::make_owned(GERG2004::GERG2004ResidualModel(spec.at("names")));
        }
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_GERG2008resid(const nlohmann::json &spec){
            return teqp::cppinterface::adapter::make_owned(GERG2008::GERG2008ResidualModel(spec.at("names")));
        }
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_GERG2004idealgas(const nlohmann::json &spec){
            return teqp::cppinterface::adapter::make_owned(GERG2004::GERG2004IdealGasModel(spec.at("names")));
        }
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_GERG2008idealgas(const nlohmann::json &spec){
            return teqp::cppinterface::adapter::make_owned(GERG2008::GERG2008IdealGasModel(spec.at("names")));
        }
#else
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_GERG2004resid(const nlohmann::json &){
            throw teqp::NotImplementedError("The GERG-2004 residual model has been disabled");
        }
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_GERG2008resid(const nlohmann::json &){
            throw teqp::NotImplementedError("The GERG-2008 residual model has been disabled");
        }
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_GERG2004idealgas(const nlohmann::json &){
            throw teqp::NotImplementedError("The GERG-2004 ideal-gas model has been disabled");
        }
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_GERG2008idealgas(const nlohmann::json &){
            throw teqp::NotImplementedError("The GERG-2008 ideal-gas model has been disabled");
        }
#endif
    }
}
