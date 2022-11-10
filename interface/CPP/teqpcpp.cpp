#include "teqp/cpp/teqpcpp.hpp"
#include "teqp/json_builder.hpp"

namespace teqp {
    namespace cppinterface {

        class ModelImplementer : public AbstractModel {
        private:
            template<typename cls>
            const cls& get_or_fail(const std::string& typestr) const{
                if (std::holds_alternative<cls>(m_model)){
                    return std::get<cls>(m_model);
                }
                else{
                    throw std::invalid_argument("This method is only available for models of the type " + std::string(typestr));
                }
            }
        protected:
            AllowedModels m_model;
            using RAX = Eigen::Ref<const Eigen::ArrayXd>;
            
        public:
            ModelImplementer(AllowedModels&& model) : m_model(model) {};
            
            const AllowedModels& get_model() const override{
                return m_model;
            }
            AllowedModels& get_mutable_model() override{
                return m_model;
            }
            
            double get_R(const EArrayd& molefracs) const override {
                return std::visit([&](const auto& model) {
                    return model.R(molefracs);
                }, m_model);
            }
            
            nlohmann::json trace_critical_arclength_binary(const double T0, const EArrayd& rhovec0, const std::optional<std::string>& filename, const std::optional<TCABOptions> &options) const override;
            EArrayd get_drhovec_dT_crit(const double T, const REArrayd& rhovec) const override;
            double get_dp_dT_crit(const double T, const REArrayd& rhovec) const override;
            EArray2 get_criticality_conditions(const double T, const REArrayd& rhovec) const override;
            EigenData eigen_problem(const double T, const REArrayd& rhovec, const std::optional<REArrayd>& alignment_v0) const override;
            double get_minimum_eigenvalue_Psi_Hessian(const double T, const REArrayd& rhovec) const override ;
            
            std::tuple<double, double> solve_pure_critical(const double T, const double rho, const std::optional<nlohmann::json>& flags) const override ;
            std::tuple<EArrayd, EMatrixd> get_pure_critical_conditions_Jacobian(const double T, const double rho, int alternative_pure_index, int alternative_length) const override ;
            std::tuple<double, double> extrapolate_from_critical(const double Tc, const double rhoc, const double Tnew) const override ;

            // Derivatives from isochoric thermodynamics (all have the same signature)
            #define X(f) \
            double f(const double T, const REArrayd& rhovec) const override ;
            ISOCHORIC_double_args
            #undef X

            #define X(f) \
            EArrayd f(const double T, const REArrayd& rhovec) const override ;
            ISOCHORIC_array_args
            #undef X

            #define X(f) \
            EMatrixd f(const double T, const REArrayd& rhovec) const override ;
            ISOCHORIC_matrix_args
            #undef X

            EArray33d get_deriv_mat2(const double T, double rho, const EArrayd& z) const override ;
            
            double get_B2vir(const double T, const EArrayd& molefrac) const override;
            double get_B12vir(const double T, const EArrayd& molefrac) const override;
            std::map<int, double> get_Bnvir(const int Nderiv, const double T, const EArrayd& molefrac) const override;
            double get_dmBnvirdTm(const int Nderiv, const int NTderiv, const double T, const EArrayd& molefrac) const override;

            double get_Arxy(const int NT, const int ND, const double T, const double rho, const EArrayd& molefracs) const override;
            // Here XMacros are used to create functions like get_Ar00, get_Ar01, ....
            #define X(i,j) \
            double get_Ar ## i ## j(const double T, const double rho, const REArrayd& molefracs) const override;
            ARXY_args
            #undef X
            // Here XMacros are used to create functions like get_Ar01n, get_Ar02n, ....
            #define X(i) \
            EArrayd get_Ar0 ## i ## n(const double T, const double rho, const REArrayd& molefracs) const override;
            AR0N_args
            #undef X

            double get_neff(const double T, const double rho, const EArrayd& molefracs) const override;

            EArray2 pure_VLE_T(const double T, const double rhoL, const double rhoV, int maxiter) const override;
            std::tuple<EArrayd, EArrayd> get_drhovecdp_Tsat(const double T, const REArrayd& rhovecL, const REArrayd& rhovecV) const override;
            std::tuple<EArrayd, EArrayd> get_drhovecdT_psat(const double T, const REArrayd& rhovecL, const REArrayd& rhovecV) const override;
            double get_dpsat_dTsat_isopleth(const double T, const REArrayd& rhovecL, const REArrayd& rhovecV) const override;

            nlohmann::json trace_VLE_isotherm_binary(const double T0, const EArrayd& rhovecL0, const EArrayd& rhovecV0, const std::optional<TVLEOptions> &options) const override;
            nlohmann::json trace_VLE_isobar_binary(const double p, const double T0, const EArrayd& rhovecL0, const EArrayd& rhovecV0, const std::optional<PVLEOptions> &options) const override;
            std::tuple<VLE_return_code,EArrayd,EArrayd> mix_VLE_Tx(const double T, const REArrayd& rhovecL0, const REArrayd& rhovecV0, const REArrayd& xspec, const double atol, const double reltol, const double axtol, const double relxtol, const int maxiter) const override;
            MixVLEReturn mix_VLE_Tp(const double T, const double pgiven, const REArrayd& rhovecL0, const REArrayd& rhovecV0, const std::optional<MixVLETpFlags> &flags) const override;
            std::tuple<VLE_return_code,double,EArrayd,EArrayd> mixture_VLE_px(const double p_spec, const REArrayd& xmolar_spec, const double T0, const REArrayd& rhovecL0, const REArrayd& rhovecV0, const std::optional<MixVLEpxFlags>& flags ) const override;

            std::tuple<VLLE::VLLE_return_code,EArrayd,EArrayd,EArrayd> mix_VLLE_T(const double T, const REArrayd& rhovecVinit, const REArrayd& rhovecL1init, const REArrayd& rhovecL2init, const double atol, const double reltol, const double axtol, const double relxtol, const int maxiter) const override;

            std::vector<nlohmann::json> find_VLLE_T_binary(const std::vector<nlohmann::json>& traces, const std::optional<VLLE::VLLEFinderOptions> options) const override;
        };

        
    }
}
