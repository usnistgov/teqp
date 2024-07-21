#include "teqp/cpp/teqpcpp.hpp"
#include "teqp/cpp/deriv_adapter.hpp"
#include "teqp/models/model_potentials/squarewell.hpp"
#include "teqp/models/model_potentials/exp6.hpp"
#include "teqp/models/model_potentials/2center_ljf.hpp"

#include "teqp/models/mie/mie.hpp"

namespace teqp{
    namespace cppinterface{
        using teqp::cppinterface::adapter::make_owned;
    
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_SW_EspindolaHeredia2009(const nlohmann::json &spec){
            return make_owned(squarewell::EspindolaHeredia2009(spec.at("lambda")));
        }
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_EXP6_Kataoka1992(const nlohmann::json &spec){
            return make_owned(exp6::Kataoka1992(spec.at("alpha")));
        }
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_Mie_Pohl2023(const nlohmann::json &spec){
            return make_owned(Mie::Mie6Pohl2023(spec.at("lambda_r")));
        }
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_Mie_Chaparro2023(const nlohmann::json &spec){
            return make_owned(FEANN::ChaparroJCP2023(spec.at("lambda_r"), spec.at("lambda_a")));
        }
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_2CLJF(const nlohmann::json &spec){
            return make_owned(twocenterljf::build_two_center_model(spec.at("author"), spec.at("L^*")));
        }
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_2CLJF_Dipole(const nlohmann::json &spec){
            return make_owned(twocenterljf::build_two_center_model_dipole(spec.at("author"), spec.at("L^*"), spec.at("(mu^*)^2")));
        }
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_2CLJF_Quadrupole(const nlohmann::json &spec){
            return make_owned(twocenterljf::build_two_center_model_quadrupole(spec.at("author"), spec.at("L^*"), spec.at("(Q^*)^2")));
        }
    }
}
