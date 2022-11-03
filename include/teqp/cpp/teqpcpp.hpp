#pragma once 
#include <memory>

#include <Eigen/Dense>
#include "nlohmann/json.hpp"

// The only headers that can be included here are
// ones that define and use POD (plain ole' data) types
#include "teqp/algorithms/critical_tracing_types.hpp"

using EArray2 = Eigen::Array<double, 2, 1>;
using EArrayd = Eigen::ArrayX<double>;
using EArray33d = Eigen::Array<double, 3, 3>;

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
         
        */
        class AbstractModel {
        public:
            
            virtual nlohmann::json trace_critical_arclength_binary(const double T0, const EArrayd& rhovec0, const std::optional<std::string>&, const std::optional<TCABOptions> &) const = 0;
            virtual EArray2 pure_VLE_T(const double T, const double rhoL, const double rhoV, int maxiter) const = 0;
            virtual EArrayd get_fugacity_coefficients(const double T, const EArrayd& rhovec) const = 0;
            virtual EArrayd get_partial_molar_volumes(const double T, const EArrayd& rhovec) const = 0;
            virtual EArray33d get_deriv_mat2(const double T, double rho, const EArrayd& z) const = 0;
            
            virtual double get_Arxy(const int, const int, const double, const double, const EArrayd&) const = 0;
            
            // Here XMacros are used to create functions like get_Ar00, get_Ar01, ....
            #define X(i,j) virtual double get_Ar ## i ## j(const double T, const double rho, const EArrayd& molefrac) const = 0;
                ARXY_args
            #undef X
            // And like get_Ar01n, get_Ar02n, ....
            #define X(i) virtual EArrayd get_Ar0 ## i ## n(const double T, const double rho, const EArrayd& molefrac) const = 0;
                AR0N_args
            #undef X
            
            
            // Virial derivatives
            virtual double get_B2vir(const double T, const EArrayd& z) const = 0;
            virtual std::map<int, double> get_Bnvir(const int Nderiv, const double T, const EArrayd& z) const = 0;
            virtual double get_B12vir(const double T, const EArrayd& z) const = 0;
            virtual double get_dmBnvirdTm(const int Nderiv, const int NTderiv, const double T, const EArrayd& z) const = 0;
            
            // Methods only available for PC-SAFT
            virtual EArrayd get_m() const = 0;
            virtual EArrayd get_sigma_Angstrom() const = 0;
            virtual EArrayd get_epsilon_over_k_K() const = 0;
            virtual double max_rhoN(const double, const EArrayd&) const = 0;
            
            // Methods only available for canonical cubics
            
            // Methods only available for multifluid
            
            virtual ~AbstractModel() = default;
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
        std::shared_ptr<AbstractModel> make_vdW1(double a, double b);
        std::shared_ptr<AbstractModel> make_canonical_PR(const std::valarray<double>& Tcrit, const std::valarray<double>&pcrit, const std::valarray<double>&acentric, const std::valarray<std::valarray<double>>& kmat);
        std::shared_ptr<AbstractModel> make_canonical_SRK(const std::valarray<double>&Tcrit, const std::valarray<double>&pcrit, const std::valarray<double>& acentric, const std::valarray<std::valarray<double>>& kmat);
    }
}
