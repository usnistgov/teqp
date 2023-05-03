#include "teqp/cpp/teqpcpp.hpp"
#include "teqp/algorithms/critical_pure.hpp"
#include "teqp/algorithms/VLE_pure.hpp"
#include "teqp/algorithms/VLLE.hpp"

namespace teqp{
    namespace cppinterface{
    
        double AbstractModel::get_neff(const double T, const double rho, const EArrayd& molefracs) const {
            return -3.0*(this->get_Ar01(T, rho, molefracs) - this->get_Ar11(T, rho, molefracs) )/this->get_Ar20(T,rho,molefracs);
        };

        std::tuple<double, double> AbstractModel::solve_pure_critical(const double T, const double rho, const std::optional<nlohmann::json>& flags) const  {
            return teqp::solve_pure_critical(*this, T, rho, flags.value_or(nlohmann::json{}));
        }
        std::tuple<EArrayd, EMatrixd> AbstractModel::get_pure_critical_conditions_Jacobian(const double T, const double rho, int alternative_pure_index, int alternative_length) const {
            return teqp::get_pure_critical_conditions_Jacobian(*this, T, rho, alternative_pure_index, alternative_length);
        }
        std::tuple<double, double> AbstractModel::extrapolate_from_critical(const double Tc, const double rhoc, const double Tnew) const {
            return teqp::extrapolate_from_critical(*this, Tc, rhoc, Tnew);
        }

        EArray2 AbstractModel::pure_VLE_T(const double T, const double rhoL, const double rhoV, int maxiter) const {
            return teqp::pure_VLE_T(*this, T, rhoL, rhoV, maxiter);
        }

        double AbstractModel::dpsatdT_pure(const double T, const double rhoL, const double rhoV) const {
            return teqp::dpsatdT_pure(*this, T, rhoL, rhoV);
        }
    
        std::tuple<VLLE::VLLE_return_code,EArrayd,EArrayd,EArrayd> AbstractModel::mix_VLLE_T(const double T, const REArrayd& rhovecVinit, const REArrayd& rhovecL1init, const REArrayd& rhovecL2init, const double atol, const double reltol, const double axtol, const double relxtol, const int maxiter) const{
            
            return VLLE::mix_VLLE_T(*this, T, rhovecVinit, rhovecL1init, rhovecL2init, atol, reltol, axtol, relxtol, maxiter);
        }

        std::vector<nlohmann::json> AbstractModel::find_VLLE_T_binary(const std::vector<nlohmann::json>& traces, const std::optional<VLLE::VLLEFinderOptions> options) const{
            return VLLE::find_VLLE_T_binary(*this, traces, options);;
        }
    }
}
