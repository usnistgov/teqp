#include "teqp/cpp/teqpcpp.hpp"
#include "teqp/cpp/deriv_adapter.hpp"
#include "teqp/models/GERG/GERG.hpp"

namespace teqp{
    namespace cppinterface{
    
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
    }
}
