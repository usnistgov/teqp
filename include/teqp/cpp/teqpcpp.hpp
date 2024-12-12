#pragma once 
#include <memory>
#include <typeindex>
#include <optional>

#include <Eigen/Dense>
#include "nlohmann/json.hpp"

// The only headers that can be included here are
// ones that define and use POD (plain ole' data) types
#include "teqp/algorithms/critical_tracing_types.hpp"
#include "teqp/algorithms/VLE_types.hpp"
#include "teqp/algorithms/VLLE_types.hpp"

using EArray2 = Eigen::Array<double, 2, 1>;
using EArrayd = Eigen::ArrayX<double>;
using EArray33d = Eigen::Array<double, 3, 3>;
using REArrayd = Eigen::Ref<const EArrayd>;
using EMatrixd = Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic>;
using REMatrixd = Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic>>;

#define ARXY_args \
    X(0,0) \
    X(0,1) \
    X(0,2) \
    X(0,3) \
    X(0,4) \
    X(1,0) \
    X(1,1) \
    X(1,2) \
    X(1,3) \
    X(1,4) \
    X(2,0) \
    X(2,1) \
    X(2,2) \
    X(2,3) \
    X(2,4)

#define AR0N_args \
    X(1) \
    X(2) \
    X(3) \
    X(4) \
    X(5) \
    X(6)

// Note: the 0 index is not included because it is already present in the AR0N_args array
// otherwise you would have two copies of the 00
#define ARN0_args \
    X(1) \
    X(2) \
    X(3) \
    X(4)

// Functions that return a double, take T and rhovec as arguments
#define ISOCHORIC_double_args \
    X(get_pr) \
    X(get_splus) \
    X(get_dpdT_constrhovec)

#define ISOCHORIC_array_args \
    X(build_Psir_gradient_autodiff) \
    X(get_chempotVLE_autodiff) \
    X(get_dchempotdT_autodiff) \
    X(get_fugacity_coefficients) \
    X(get_partial_molar_volumes) \
    X(build_d2PsirdTdrhoi_autodiff) \
    X(get_dpdrhovec_constT)

#define ISOCHORIC_matrix_args \
    X(build_Psir_Hessian_autodiff) \
    X(build_Psi_Hessian_autodiff)

#define ISOCHORIC_multimatrix_args \
    X(build_Psir_fgradHessian_autodiff)
    
namespace teqp {
    namespace cppinterface {

        /**
        This class defines the public interface for a model.  Only native C++ types are passed through this inferface
         (as well as Eigen types and JSON data structures). Thus all the advanced derivative things can be hidden behind the C++ wall,
         yielding an interface that is still powerful, and very fast, but compilation times can be reduced to something more reasonable. Also,
         interfacing with other programming languages becomes much more convenient with this layer.  All the complicated
         routines are still available in the lower-level C++ code.
         
         Not allowed are:
         * Templated arguments to functions
         * Other numerical types (complex, multicomplex, autodiff, etc.)
         
         X-Macros can be used to wrap functions that take template arguments and expand them as multiple functions
         
        */
        class AbstractModel {
        public:
            
            virtual ~AbstractModel() = default;
            
            virtual const std::type_index& get_type_index() const = 0;
            
            virtual double get_R(const EArrayd&) const = 0;
            double R(const EArrayd& x) const { return get_R(x); };
            
            virtual double get_Arxy(const int, const int, const double, const double, const EArrayd&) const = 0;
            
            // Here X-Macros are used to create functions like get_Ar00, get_Ar01, ....
            #define X(i,j) virtual double get_Ar ## i ## j(const double T, const double rho, const REArrayd& molefrac) const = 0;
                ARXY_args
            #undef X
            // And like get_Ar01n, get_Ar02n, ....
            #define X(i) virtual EArrayd get_Ar0 ## i ## n(const double T, const double rho, const REArrayd& molefrac) const = 0;
                AR0N_args
            #undef X
            // And like get_Ar10n, get_Ar20n, ....
            #define X(i) virtual EArrayd get_Ar ## i ## 0n(const double T, const double rho, const REArrayd& molefrac) const = 0;
                ARN0_args
            #undef X
            
            // Extended precision evaluations, for testing of virial coefficients
            virtual double get_Ar01ep(const double, const double, const EArrayd&) const = 0;
            virtual double get_Ar02ep(const double, const double, const EArrayd&) const = 0;
            virtual double get_Ar03ep(const double, const double, const EArrayd&) const = 0;
            
            // Pass-through functions to give access to reducing function evaluations
            // for multi-fluid models in the corresponding-states formulation
            virtual double get_reducing_density(const EArrayd&) const = 0;
            virtual double get_reducing_temperature(const EArrayd&) const = 0;
            
            // Virial derivatives
            virtual double get_B2vir(const double T, const EArrayd& z) const = 0;
            virtual std::map<int, double> get_Bnvir(const int Nderiv, const double T, const EArrayd& z) const = 0;
            virtual double get_B12vir(const double T, const EArrayd& z) const = 0;
            virtual double get_dmBnvirdTm(const int Nderiv, const int NTderiv, const double T, const EArrayd& z) const = 0;
            
            // Composition derivatives
            virtual double get_ATrhoXi(const double T, const int NT, const double rhomolar, int ND, const EArrayd& molefrac, const int i, const int NXi) const = 0;
            virtual double get_ATrhoXiXj(const double T, const int NT, const double rhomolar, int ND, const EArrayd& molefrac, const int i, const int NXi, const int j, const int NXj) const = 0;
            virtual double get_ATrhoXiXjXk(const double T, const int NT, const double rhomolar, int ND, const EArrayd& molefrac, const int i, const int NXi, const int j, const int NXj, const int k, const int NXk) const = 0;
            
            virtual double get_AtaudeltaXi(const double tau, const int Ntau, const double delta, int Ndelta, const EArrayd& molefrac, const int i, const int NXi) const = 0;
            virtual double get_AtaudeltaXiXj(const double tau, const int Ntau, const double delta, int Ndelta, const EArrayd& molefrac, const int i, const int NXi, const int j, const int NXj) const = 0;
            virtual double get_AtaudeltaXiXjXk(const double tau, const int Ntau, const double delta, int Ndelta, const EArrayd& molefrac, const int i, const int NXi, const int j, const int NXj, const int k, const int NXk) const = 0;
            
            // Derivatives from isochoric thermodynamics (all have the same signature whithin each block)
            #define X(f) virtual double f(const double T, const EArrayd& rhovec) const = 0;
                ISOCHORIC_double_args
            #undef X
            #define X(f) virtual EArrayd f(const double T, const EArrayd& rhovec) const = 0;
                ISOCHORIC_array_args
            #undef X
            #define X(f) virtual EMatrixd f(const double T, const EArrayd& rhovec) const = 0;
                ISOCHORIC_matrix_args
            #undef X
            #define X(f) virtual std::tuple<double, Eigen::ArrayXd, Eigen::MatrixXd> f(const double T, const EArrayd& rhovec) const = 0;
                ISOCHORIC_multimatrix_args
            #undef X
            virtual Eigen::ArrayXd get_Psir_sigma_derivs(const double T, const EArrayd& rhovec, const EArrayd& v) const = 0;
            
            double get_neff(const double, const double, const EArrayd&) const;
            
            virtual EArray33d get_deriv_mat2(const double T, double rho, const EArrayd& z ) const = 0;
            
            std::tuple<double, double> solve_pure_critical(const double T, const double rho, const std::optional<nlohmann::json>& = std::nullopt) const ;
            EArray2 extrapolate_from_critical(const double Tc, const double rhoc, const double Tgiven, const std::optional<Eigen::ArrayXd>& molefracs = std::nullopt) const;
            std::tuple<EArrayd, EMatrixd> get_pure_critical_conditions_Jacobian(const double T, const double rho, const std::optional<std::size_t>& alternative_pure_index, const std::optional<std::size_t>& alternative_length) const;
            
            EArray2 pure_VLE_T(const double T, const double rhoL, const double rhoV, int maxiter, const std::optional<Eigen::ArrayXd>& molefracs = std::nullopt) const;
            double dpsatdT_pure(const double T, const double rhoL, const double rhoV) const;
            
            virtual std::tuple<EArrayd, EArrayd> get_drhovecdp_Tsat(const double T, const REArrayd& rhovecL, const REArrayd& rhovecV) const;
            virtual std::tuple<EArrayd, EArrayd> get_drhovecdT_psat(const double T, const REArrayd& rhovecL, const REArrayd& rhovecV) const;
            virtual double get_dpsat_dTsat_isopleth(const double T, const REArrayd& rhovecL, const REArrayd& rhovecV) const;
            virtual nlohmann::json trace_VLE_isotherm_binary(const double T0, const EArrayd& rhovec0, const EArrayd& rhovecV0, const std::optional<TVLEOptions> & = std::nullopt) const;
            virtual nlohmann::json trace_VLE_isobar_binary(const double p, const double T0, const EArrayd& rhovecL0, const EArrayd& rhovecV0, const std::optional<PVLEOptions> & = std::nullopt) const;
            virtual std::tuple<VLE_return_code,EArrayd,EArrayd> mix_VLE_Tx(const double T, const REArrayd& rhovecL0, const REArrayd& rhovecV0, const REArrayd& xspec, const double atol, const double reltol, const double axtol, const double relxtol, const int maxiter) const;
            virtual MixVLEReturn mix_VLE_Tp(const double T, const double pgiven, const REArrayd& rhovecL0, const REArrayd& rhovecV0, const std::optional<MixVLETpFlags> &flags = std::nullopt) const;
            virtual std::tuple<VLE_return_code,double,EArrayd,EArrayd> mixture_VLE_px(const double p_spec, const REArrayd& xmolar_spec, const double T0, const REArrayd& rhovecL0, const REArrayd& rhovecV0, const std::optional<MixVLEpxFlags>& flags = std::nullopt) const;
            
            std::tuple<VLLE::VLLE_return_code,EArrayd,EArrayd,EArrayd> mix_VLLE_T(const double T, const REArrayd& rhovecVinit, const REArrayd& rhovecL1init, const REArrayd& rhovecL2init, const double atol, const double reltol, const double axtol, const double relxtol, const int maxiter) const;
            std::vector<nlohmann::json> find_VLLE_T_binary(const std::vector<nlohmann::json>& traces, const std::optional<VLLE::VLLEFinderOptions> options = std::nullopt) const;
            std::vector<nlohmann::json> find_VLLE_p_binary(const std::vector<nlohmann::json>& traces, const std::optional<VLLE::VLLEFinderOptions> options = std::nullopt) const;
            nlohmann::json trace_VLLE_binary(const double T, const REArrayd& rhovecV, const REArrayd& rhovecL1, const REArrayd& rhovecL2, const std::optional<VLLE::VLLETracerOptions> options) const;
            
            virtual nlohmann::json trace_critical_arclength_binary(const double T0, const EArrayd& rhovec0, const std::optional<std::string>& = std::nullopt, const std::optional<TCABOptions> & = std::nullopt) const;
            virtual EArrayd get_drhovec_dT_crit(const double T, const REArrayd& rhovec) const;
            virtual double get_dp_dT_crit(const double T, const REArrayd& rhovec) const;
            virtual EArray2 get_criticality_conditions(const double T, const REArrayd& rhovec) const;
            virtual EigenData eigen_problem(const double T, const REArrayd& rhovec, const std::optional<REArrayd>& = std::nullopt) const;
            virtual double get_minimum_eigenvalue_Psi_Hessian(const double T, const REArrayd& rhovec) const;
            
        };
        
        // Generic JSON-based interface where the model description is encoded as JSON
        std::unique_ptr<AbstractModel> make_model(const nlohmann::json &, bool validate = true);

        // Expose specialized factory functions for different models
        // Mostly these are just adapter functions that prepare some
        // JSON and pass it to the make_model function
        // ....
        std::unique_ptr<AbstractModel> make_multifluid_model(
            const std::vector<std::string>& components, 
            const std::string& coolprop_root, 
            const std::string& BIPcollectionpath = {}, 
            const nlohmann::json& flags = {}, 
            const std::string& departurepath = {}
        );
    
        std::unique_ptr<AbstractModel> build_model_ptr(const nlohmann::json& json, bool validate = true);
    
        /// Return the schema for the given model kind
        nlohmann::json get_model_schema(const std::string& kind);
    
        
        using ModelPointerFactoryFunction = std::function<std::unique_ptr<teqp::cppinterface::AbstractModel>(const nlohmann::json &j)>;
    
        /**
         * \brief This function allows you to inject your own model factory function into the set of factory functions implemented in teqp. This allows you to add your own model at runtime. As an example of how to do this, see src/test_runtime_model_inclusion.cpp
         * \param key The key used to define the model
         * \param func The ModelPointerFactoryFunction factory function used to generate the wrapped model
         */
        void add_model_pointer_factory_function(const std::string& key, ModelPointerFactoryFunction& func);
        
    }
}
