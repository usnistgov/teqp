#include "teqp/cpp/teqpcpp.hpp"
#include "teqp/algorithms/critical_pure.hpp"
#include "teqp/algorithms/VLE_pure.hpp"
#include "teqp/algorithms/VLE.hpp"
#include "teqp/algorithms/VLLE.hpp"

namespace teqp{
    namespace cppinterface{
    
        double AbstractModel::get_neff(const double T, const double rho, const EArrayd& molefracs) const {
            return -3.0*(this->get_Ar01(T, rho, molefracs) - this->get_Ar11(T, rho, molefracs) )/this->get_Ar20(T,rho,molefracs);
        };

        std::tuple<double, double> AbstractModel::solve_pure_critical(const double T, const double rho, const std::optional<nlohmann::json>& flags) const  {
            return teqp::solve_pure_critical(*this, T, rho, flags.value_or(nlohmann::json{}));
        }
        std::tuple<EArrayd, EMatrixd> AbstractModel::get_pure_critical_conditions_Jacobian(const double T, const double rho, const std::optional<std::size_t>& alternative_pure_index, const std::optional<std::size_t>& alternative_length) const {
            return teqp::get_pure_critical_conditions_Jacobian(*this, T, rho, alternative_pure_index, alternative_length);
        }
        EArray2 AbstractModel::extrapolate_from_critical(const double Tc, const double rhoc, const double Tnew, const std::optional<Eigen::ArrayXd>& molefracs) const {
            return teqp::extrapolate_from_critical(*this, Tc, rhoc, Tnew, molefracs);
        }

        EArray2 AbstractModel::pure_VLE_T(const double T, const double rhoL, const double rhoV, int maxiter, const std::optional<Eigen::ArrayXd>& molefracs) const {
            return teqp::pure_VLE_T(*this, T, rhoL, rhoV, maxiter, molefracs);
        }

        double AbstractModel::dpsatdT_pure(const double T, const double rhoL, const double rhoV) const {
            return teqp::dpsatdT_pure(*this, T, rhoL, rhoV);
        }
    
        std::tuple<VLLE::VLLE_return_code,EArrayd,EArrayd,EArrayd> AbstractModel::mix_VLLE_T(const double T, const REArrayd& rhovecVinit, const REArrayd& rhovecL1init, const REArrayd& rhovecL2init, const double atol, const double reltol, const double axtol, const double relxtol, const int maxiter) const{
            
            return VLLE::mix_VLLE_T(*this, T, rhovecVinit, rhovecL1init, rhovecL2init, atol, reltol, axtol, relxtol, maxiter);
        }

        std::vector<nlohmann::json> AbstractModel::find_VLLE_T_binary(const std::vector<nlohmann::json>& traces, const std::optional<VLLE::VLLEFinderOptions> options) const{
            return VLLE::find_VLLE_T_binary(*this, traces, options);
        }
        std::vector<nlohmann::json> AbstractModel::find_VLLE_p_binary(const std::vector<nlohmann::json>& traces, const std::optional<VLLE::VLLEFinderOptions> options) const{
            return VLLE::find_VLLE_p_binary(*this, traces, options);
        }
    
        nlohmann::json AbstractModel::trace_VLLE_binary(const double T, const REArrayd& rhovecV, const REArrayd& rhovecL1, const REArrayd& rhovecL2, const std::optional<VLLE::VLLETracerOptions> options) const{
            return VLLE::trace_VLLE_binary(*this, T, rhovecV, rhovecL1, rhovecL2, options);
        }
    
    std::tuple<VLE_return_code,EArrayd,EArrayd> AbstractModel::mix_VLE_Tx(const double T, const REArrayd& rhovecL0, const REArrayd& rhovecV0, const REArrayd& xspec, const double atol, const double reltol, const double axtol, const double relxtol, const int maxiter) const{
        return teqp::mix_VLE_Tx(*this, T, rhovecL0, rhovecV0, xspec, atol, reltol, axtol, relxtol, maxiter);
    
    }
    MixVLEReturn AbstractModel::mix_VLE_Tp(const double T, const double pgiven, const REArrayd& rhovecL0, const REArrayd& rhovecV0, const std::optional<MixVLETpFlags> &flags) const{
        return teqp::mix_VLE_Tp(*this, T, pgiven, rhovecL0, rhovecV0, flags);
    }
    std::tuple<VLE_return_code,double,EArrayd,EArrayd> AbstractModel::mixture_VLE_px(const double p_spec, const REArrayd& xmolar_spec, const double T0, const REArrayd& rhovecL0, const REArrayd& rhovecV0, const std::optional<MixVLEpxFlags>& flags) const{
        return teqp::mixture_VLE_px(*this, p_spec, xmolar_spec, T0, rhovecL0, rhovecV0, flags);
    }
    
    std::tuple<EArrayd, EArrayd> AbstractModel::get_drhovecdp_Tsat(const double T, const REArrayd& rhovecL, const REArrayd& rhovecV) const {
        return teqp::get_drhovecdp_Tsat(*this, T, rhovecL, rhovecV);
    }
    std::tuple<EArrayd, EArrayd> AbstractModel::get_drhovecdT_psat(const double T, const REArrayd& rhovecL, const REArrayd& rhovecV) const {
        return teqp::get_drhovecdT_psat(*this, T, rhovecL, rhovecV);
    }
    double AbstractModel::get_dpsat_dTsat_isopleth(const double T, const REArrayd& rhovecL, const REArrayd& rhovecV) const {
        return teqp::get_dpsat_dTsat_isopleth(*this, T, rhovecL, rhovecV);
    }
    nlohmann::json AbstractModel::trace_VLE_isotherm_binary(const double T0, const EArrayd& rhovecL0, const EArrayd& rhovecV0, const std::optional<TVLEOptions> &options) const{
        return teqp::trace_VLE_isotherm_binary(*this, T0, rhovecL0, rhovecV0, options);
    }
    nlohmann::json AbstractModel::trace_VLE_isobar_binary(const double p, const double T0, const EArrayd& rhovecL0, const EArrayd& rhovecV0, const std::optional<PVLEOptions> &options) const{
        return teqp::trace_VLE_isobar_binary(*this, p, T0, rhovecL0, rhovecV0, options);
    }
    
    nlohmann::json AbstractModel::trace_critical_arclength_binary(const double T0, const EArrayd& rhovec0, const std::optional<std::string>& filename, const std::optional<TCABOptions> &options) const {
        using crit = teqp::CriticalTracing<decltype(*this), double, std::decay_t<decltype(rhovec0)>>;
        return crit::trace_critical_arclength_binary(*this, T0, rhovec0, filename , options);
    }
    EArrayd AbstractModel::get_drhovec_dT_crit(const double T, const REArrayd& rhovec) const {
        using crit = teqp::CriticalTracing<decltype(*this), double, std::decay_t<decltype(rhovec)>>;
        return crit::get_drhovec_dT_crit(*this, T, rhovec);
    }
    double AbstractModel::get_dp_dT_crit(const double T, const REArrayd& rhovec) const {
        using crit = teqp::CriticalTracing<decltype(*this), double, std::decay_t<decltype(rhovec)>>;
        return crit::get_dp_dT_crit(*this, T, rhovec);
    }
    EArray2 AbstractModel::get_criticality_conditions(const double T, const REArrayd& rhovec) const {
        using crit = teqp::CriticalTracing<decltype(*this), double, std::decay_t<decltype(rhovec)>>;
        return crit::get_criticality_conditions(*this, T, rhovec);
    }
    EigenData AbstractModel::eigen_problem(const double T, const REArrayd& rhovec, const std::optional<REArrayd>& alignment_v0) const {
        using crit = teqp::CriticalTracing<decltype(*this), double, std::decay_t<decltype(rhovec)>>;
        return crit::eigen_problem(*this, T, rhovec, alignment_v0.value_or(Eigen::ArrayXd()));
    }
    double AbstractModel::get_minimum_eigenvalue_Psi_Hessian(const double T, const REArrayd& rhovec) const {
        using crit = teqp::CriticalTracing<decltype(*this), double, std::decay_t<decltype(rhovec)>>;
        return crit::get_minimum_eigenvalue_Psi_Hessian(*this, T, rhovec);
    }
    
    }
}
