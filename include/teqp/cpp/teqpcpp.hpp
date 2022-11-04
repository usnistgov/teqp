#pragma once 
#include <memory>

#include <Eigen/Dense>
#include "nlohmann/json.hpp"
#include "teqp/models/fwd.hpp"

// The only headers that can be included here are
// ones that define and use POD (plain ole' data) types
#include "teqp/algorithms/critical_tracing_types.hpp"

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
    X(0) \
    X(1) \
    X(2) \
    X(3) \
    X(4) \
    X(5) \
    X(6)

// Functions that return a double, take T and rhovec as arguments
#define ISOCHORIC_double_args \
    X(get_pr) \
    X(get_splus)

#define ISOCHORIC_array_args \
    X(build_Psir_gradient_autodiff) \
    X(get_chempotVLE_autodiff) \
    X(get_dchempotdT_autodiff) \
    X(get_fugacity_coefficients) \
    X(get_partial_molar_volumes) \
    X(build_d2PsirdTdrhoi_autodiff)

#define ISOCHORIC_matrix_args \
    X(build_Psir_Hessian_autodiff) \
    X(build_Psi_Hessian_autodiff)

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
            
            // This method allows for access to the contained variant in the subclass
            // You can access all its methods after the appropriate std::get call
            // with the right type
            virtual const AllowedModels& get_model() const = 0;
            virtual ~AbstractModel() = default;
            
            virtual double get_R(const EArrayd&) const = 0;
            
            virtual double get_Arxy(const int, const int, const double, const double, const EArrayd&) const = 0;
            
            // Here X-Macros are used to create functions like get_Ar00, get_Ar01, ....
            #define X(i,j) virtual double get_Ar ## i ## j(const double T, const double rho, const REArrayd& molefrac) const = 0;
                ARXY_args
            #undef X
            // And like get_Ar01n, get_Ar02n, ....
            #define X(i) virtual EArrayd get_Ar0 ## i ## n(const double T, const double rho, const REArrayd& molefrac) const = 0;
                AR0N_args
            #undef X
            virtual double get_neff(const double, const double, const EArrayd&) const = 0;
            
            // Virial derivatives
            virtual double get_B2vir(const double T, const EArrayd& z) const = 0;
            virtual std::map<int, double> get_Bnvir(const int Nderiv, const double T, const EArrayd& z) const = 0;
            virtual double get_B12vir(const double T, const EArrayd& z) const = 0;
            virtual double get_dmBnvirdTm(const int Nderiv, const int NTderiv, const double T, const EArrayd& z) const = 0;
            
            // Derivatives from isochoric thermodynamics (all have the same signature whithin each block)
            #define X(f) virtual double f(const double T, const REArrayd& rhovec) const = 0;
                ISOCHORIC_double_args
            #undef X
            #define X(f) virtual EArrayd f(const double T, const REArrayd& rhovec) const = 0;
                ISOCHORIC_array_args
            #undef X
            #define X(f) virtual EMatrixd f(const double T, const REArrayd& rhovec) const = 0;
                ISOCHORIC_matrix_args
            #undef X
            
            virtual EArray33d get_deriv_mat2(const double T, double rho, const EArrayd& z ) const = 0;
            
            virtual std::tuple<double, double> solve_pure_critical(const double T, const double rho, const std::optional<nlohmann::json>&) const = 0;
            virtual std::tuple<double, double> extrapolate_from_critical(const double Tc, const double rhoc, const double Tgiven) const = 0;
            virtual EArray2 pure_VLE_T(const double T, const double rhoL, const double rhoV, int maxiter) const = 0;
            
            virtual std::tuple<EArrayd, EArrayd> get_drhovecdp_Tsat(const double T, const REArrayd& rhovecL, const REArrayd& rhovecV) const = 0;
            virtual std::tuple<EArrayd, EArrayd> get_drhovecdT_psat(const double T, const REArrayd& rhovecL, const REArrayd& rhovecV) const = 0;
            virtual double get_dpsat_dTsat_isopleth(const double T, const REArrayd& rhovecL, const REArrayd& rhovecV) const = 0;
            
            virtual nlohmann::json trace_critical_arclength_binary(const double T0, const EArrayd& rhovec0, const std::optional<std::string>&, const std::optional<TCABOptions> &) const = 0;
            virtual EArrayd get_drhovec_dT_crit(const double T, const REArrayd& rhovec) const = 0;
            virtual double get_dp_dT_crit(const double T, const REArrayd& rhovec) const = 0;
            virtual EArray2 get_criticality_conditions(const double T, const REArrayd& rhovec) const = 0;
            virtual EigenData eigen_problem(const double T, const REArrayd& rhovec, const std::optional<REArrayd>&) const = 0;
            virtual double get_minimum_eigenvalue_Psi_Hessian(const double T, const REArrayd& rhovec) const = 0;
            
        };
        
        // Generic JSON-based interface where the model description is encoded as JSON
        std::shared_ptr<AbstractModel> make_model(const nlohmann::json &);

        // Expose specialized factory functions for different models
        // Mostly these are just adapter functions that prepare some
        // JSON and pass it to the make_model function
        // ....
        std::shared_ptr<AbstractModel> make_multifluid_model(
            const std::vector<std::string>& components, 
            const std::string& coolprop_root, 
            const std::string& BIPcollectionpath = {}, 
            const nlohmann::json& flags = {}, 
            const std::string& departurepath = {}
        );
    }
}
