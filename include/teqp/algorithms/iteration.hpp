#pragma once

#include <optional>

#include "teqp/cpp/teqpcpp.hpp"
#include "teqp/cpp/derivs.hpp"
#include "teqp/exceptions.hpp"

#include <boost/multiprecision/cpp_bin_float.hpp>

namespace teqp {
namespace iteration {

using namespace cppinterface;

enum class StoppingConditionReason{not_specified, keep_going, success, fatal};

/**
 A holder class that holds references(!) to the data that is then passed to the stopping condition testing functions.
 
 This approach is used to keep the calling signature of the stopping condition functions more concise
 */
struct StoppingData{
    const int N; ///< The number of steps that have been taken so far
    const Eigen::Array2d& x; ///< The complete set of independent variables
    const Eigen::Array2d& dx; ///< The complete set of steps in independent variables
    const Eigen::Array2d& r; ///< The set of residual functions
    const std::vector<int>& nonconstant_indices; ///< The indices where the independent variable can be varied (are not T or rho)
};

/**
 The base class for a stopping condition test in iteration
 
 The function stop returns either a StoppingConditionReason::keep_going to indicate that the stopping condition is not satisfied
 or another enumerated value to indicate whether successful or a failure
 */
class StoppingCondition{
public:
    virtual StoppingConditionReason stop(const StoppingData&) = 0;
    virtual std::string desc() = 0;
    virtual ~StoppingCondition() = default;
};

class MaxAbsErrorCondition : public StoppingCondition{
private:
    double threshold;
public:
    MaxAbsErrorCondition(double threshold) : threshold(threshold) {};
    StoppingConditionReason stop(const StoppingData& data) override{
        using s = StoppingConditionReason;
        return (data.r.abs().maxCoeff() < threshold) ? s::success : s::keep_going;
    };
    std::string desc() override{
        return "MaxAbsErrorCondition";
    };
};

class MinRelStepsizeCondition : public StoppingCondition{
private:
    double threshold;
public:
    MinRelStepsizeCondition(double threshold) : threshold(threshold) {};
    
    /**
    \note Initially this was implemented as Eigen operations with vector indexing, but the manual approach is MUCH faster
     */
    StoppingConditionReason stop(const StoppingData& data) override{
        using s = StoppingConditionReason;
        double minval = 1e99;
        for (auto idx : data.nonconstant_indices){
            double val = std::abs(data.dx(idx)/data.x(idx));
            minval = std::min(val, minval);
        }
        return (minval < threshold) ? s::success : s::keep_going;
    };
    std::string desc() override{
        return "MinRelStepsizeCondition";
    };
};

class NanXDXErrorCondition : public StoppingCondition{
public:
    /**
    \note Initially this was implemented as Eigen operations with vector indexing, but the manual approach is MUCH faster
     */
    StoppingConditionReason stop(const StoppingData& data) override{
        using s = StoppingConditionReason;
        bool allfinite = true;
        for (auto idx : data.nonconstant_indices){
            if (!std::isfinite(data.x(idx)) || !std::isfinite(data.dx(idx))){
                allfinite = false; break;
            }
        }
        return (!allfinite) ? s::fatal : s::keep_going;
    };
    std::string desc() override{
        return "NanXDXErrorCondition";
    };
};

class NegativeXErrorCondition : public StoppingCondition{
public:
    StoppingConditionReason stop(const StoppingData& data) override{
        using s = StoppingConditionReason;
        bool allpositive = true;
        for (auto idx : data.nonconstant_indices){
            if (data.x(idx) < 0){
                allpositive = false; break;
            }
        }
        return (!allpositive) ? s::fatal : s::keep_going;
    };
    std::string desc() override{
        return "NegativeXErrorCondition";
    };
};

class StepCountErrorCondition : public StoppingCondition{
private:
    const std::size_t Nmax;
public:
    StepCountErrorCondition(int Nmax) : Nmax(Nmax){}
    StoppingConditionReason stop(const StoppingData& data) override{
        using s = StoppingConditionReason;
        return (data.N >= Nmax) ? s::fatal : s::keep_going;
    };
    std::string desc() override{
        return "StepCountErrorCondition";
    };
};




class AlphaModel{
public:
    std::shared_ptr<AbstractModel> model_ideal_gas, model_residual;
    
    template<typename Z>
    auto get_R(const Z& z) const{
        return model_residual->get_R(z);
    }
    
    auto get_R(const std::vector<double>& z) const{
        const auto zz = Eigen::Map<const Eigen::ArrayXd>(&z[0], z.size());
        return model_residual->get_R(zz);
    }
    
    template<typename Z>
    Eigen::Array33d get_deriv_mat2(double T, double rho, const Z& z) const{
        return model_ideal_gas->get_deriv_mat2(T, rho, z) + model_residual->get_deriv_mat2(T, rho, z);
    }
    
    Eigen::Array33d get_deriv_mat2(double T, double rho, const std::vector<double>& z) const{
        const auto zz = Eigen::Map<const Eigen::ArrayXd>(&z[0], z.size());
        return model_ideal_gas->get_deriv_mat2(T, rho, zz) + model_residual->get_deriv_mat2(T, rho, zz);
    }
    
    template<typename Z>
    auto get_A00A10A01(double T, double rho, const Z& z) const{
        auto A10 = model_ideal_gas->get_Ar10(T, rho, z) + model_residual->get_Ar10(T, rho, z);
        Eigen::Array2d A00A01 = model_ideal_gas->get_Ar01n(T, rho, z) + model_residual->get_Ar01n(T, rho, z); // This gives [A00, A01] at the same time
        return std::make_tuple(A00A01[0], A10, A00A01[1]);
    }
    
    /**
     \brief A convenience function for calculation of Jacobian terms of the form \f$ J_{i0} = \frac{\partial y}{\partial T} \f$  and \f$ J_{i1} = \frac{\partial y}{\partial \rho} \f$ where \f$y\f$ is one of the thermodynamic variables in vars
     
     \param vars A set of chars, allowed are 'H','S','U','P','T','D'
     \param A The matrix of derivatives of \f$\alpha^{\rm ig} + \alpha^{\rm r}\f$, perhaps obtained from teqp::DerivativeHolderSquare, or via get_deriv_mat2 of the AbstractModel
     \param R The molar gas constant
     \param T Temperature
     \param rho Molar density
     \param z Mole fractions
     */
    template<typename Array>
    auto get_vals(const std::vector<char>& vars, const double R, const double T, const double rho, const Array &z) const{
        
        Array v(2);
        auto Trecip = 1.0/T;
        auto dTrecipdT = -Trecip*Trecip;
        
        // Define some lambda functions for things that *might* be needed
        // The lambdas are used to get a sort of lazy evaluation. The lambdas are
        // only evaluated on an as-needed basis.  If the lambda is not called,
        // no cost is incurred.
        //
        // Probably the compiler will inline these functions anyhow.
        //
        auto [A00, A10, A01] = get_A00A10A01(T, rho, z);
        // Derivatives of total alpha(Trecip, rho)
        auto alpha = [&](){ return A00; };
        auto dalphadTrecip = [&](){ return A10/Trecip; };
        auto dalphadrho = [&](){ return A01/rho; };
        // Derivatives of total Helmholtz energy a = alpha*R/Trecip in
        // terms of derivatives of alpha(Trecip, rho)
        auto a = [&](){ return alpha()*R*T; };
        auto dadTrecip = [&](){ return R/(Trecip*Trecip)*(Trecip*dalphadTrecip()-alpha()); };
        auto dadrho = [&](){ return R/Trecip*(dalphadrho()); };
        
        for (auto i = 0; i < vars.size(); ++i){
            switch(vars[i]){
                case 'T':
                    v(i) = T; break;
                case 'D':
                    v(i) = rho; break;
                case 'P':
                    v(i) = rho*rho*dadrho(); break;
                case 'S':
                    v(i) = Trecip*Trecip*dadTrecip(); break;
                case 'H':
                    v(i) = a() + Trecip*dadTrecip() + rho*dadrho(); break;
                case 'U':
                    v(i) = a() + Trecip*dadTrecip(); break;
                default:
                    throw std::invalid_argument("bad var: " + std::to_string(vars[i]));
            }
        }
        return v;
    }
};
static_assert(std::is_copy_constructible_v<AlphaModel>);

struct NRIteratorResult{
    bool success;
    StoppingConditionReason reason;
    std::string msg;
    Eigen::Array2d Trho;
};

/**
 A class for doing Newton-Raphson steps to solve for two unknown thermodynamic variables
 
 */
class NRIterator{
private:
    const AlphaModel alphamodel;
    const std::vector<char>  vars;
    const Eigen::Array2d vals;
    
    const Eigen::ArrayXd z;
    Eigen::Array2d Trho, r;
    double R;
    
    std::tuple<bool, bool> relative_error;
    const std::vector<std::shared_ptr<StoppingCondition>> stopping_conditions;
    
    int step_counter = 0;
    std::string msg;
    std::vector<int> nonconstant_indices;
    bool isTD;
    
public:
    NRIterator(const AlphaModel& alphamodel, const std::vector<char>& vars, const Eigen::Array2d& vals, double T, double rho, const Eigen::Ref<const Eigen::ArrayXd>& z, const std::tuple<bool, bool> &relative_error, const std::vector<std::shared_ptr<StoppingCondition>> stopping_conditions) : alphamodel(alphamodel), vars(vars), vals(vals), Trho((Eigen::Array2d() << T,rho).finished()), z(z), R(alphamodel.get_R(z)), relative_error(relative_error), stopping_conditions(stopping_conditions) {
        if(!(vars[0] == 'T' || vars[1] == 'T')){ nonconstant_indices.push_back(0); }
        if(!(vars[0] == 'D' || vars[1] == 'D')){ nonconstant_indices.push_back(1); }
        isTD = (nonconstant_indices.size() == 0);
    }
    bool verbose = false;
    
    /// Return the variables that are being used in the iteration
    std::vector<char> get_vars() const { return vars; }
    /// Return the target values to be obtained
    Eigen::Array2d get_vals() const { return vals; }
    /// Return the current temperature
    auto get_T() const { return Trho(0); }
    /// Return the current molar density
    auto get_rho() const { return Trho(1); }
    /// Return the current mole fractions
    Eigen::ArrayXd get_molefrac() const { return z; }
    /// Return the current step counter
    int get_step_count() const { return step_counter; }
    /// Return the current message relaying information about success or failure of the iteration
    std::string get_message() const { return msg; }
    /// Get the indices of the nonconstant variables
    auto get_nonconstant_indices() const { return nonconstant_indices; }
    
    void reset(double T, double rho){
        Trho = (Eigen::Array2d() << T, rho).finished();
        step_counter = 0;
    }
    
    /** Do the calculations needed for the step and return the step and the other data
    *  In C++, the copy will be elided (the return value will be moved)
    * \param T Temperature
    * \param rho Molar density
    */
    auto calc_matrices(double T, double rho) const{
        auto A = alphamodel.get_deriv_mat2(T, rho, z);
        return build_iteration_Jv(vars, A, R, T, rho, z);
    }
    auto calc_step(double T, double rho) const{
        auto im = calc_matrices(T, rho);
        return std::make_tuple((im.J.matrix().fullPivLu().solve((-(im.v-vals)).matrix())).eval(), im);
    }
    auto calc_just_step(double T, double rho) const {
        return std::get<0>(calc_step(T, rho));
    }
    auto calc_vals(double T, double rho) const{
        return calc_matrices(T, rho).v;
    }
    auto calc_r(double T, double rho) const{
        auto im = calc_matrices(T, rho);
        Eigen::Array2d r = im.v-vals;
        if (std::get<0>(relative_error)){ r(0) /= vals(0);}
        if (std::get<1>(relative_error)){ r(1) /= vals(1);}
        return r;
    }
    auto calc_J(double T, double rho) const{
        auto im = calc_matrices(T, rho);
        if (std::get<0>(relative_error)){ im.J.row(0) /= vals(0);}
        if (std::get<1>(relative_error)){ im.J.row(1) /= vals(1);}
        return im.J;
    }
    /// Calculate the maximum absolute value of the values in r
    auto calc_maxabsr(double T, double rho) const{
        Eigen::Array2d r = alphamodel.get_vals(vars, R, T, rho, z)-vals;
        if (std::get<0>(relative_error)){ r(0) /= vals(0);}
        if (std::get<1>(relative_error)){ r(1) /= vals(1);}
        return r.abs().maxCoeff();
    }
    /// Get the maximum absolute value of residual vector, using the cached value for r
    auto get_maxabsr() const{
        return r.abs().maxCoeff();
    }
    
    /** Take a given number of steps
     * \param N The number of steps to take
     * \param apply_stopping True to apply the stopping conditions
     */
    auto take_steps(int N, bool apply_stopping=true){
        if (N <= 0){
            throw teqp::InvalidArgument("N must be greater than 0");
        }
        StoppingConditionReason reason = StoppingConditionReason::fatal;
        if(isTD){
            /// Special-case temperature-density inputs, which require only one step
            auto im = calc_matrices(Trho(0), Trho(1));
            r = im.v-vals;
            
            if (std::get<0>(relative_error)){ r(0) /= vals(0); im.J.row(0) /= vals(0); }
            if (std::get<1>(relative_error)){ r(1) /= vals(1); im.J.row(1) /= vals(1); }
            
            Eigen::Array2d dTrho = im.J.matrix().fullPivLu().solve((-r).matrix());
            Trho += dTrho;
            step_counter++;
            reason = StoppingConditionReason::success; msg = "Only one step is needed for DT inputs";
            return reason;
        }
        
        for (auto K = 0; K < N; ++K){
            auto im = calc_matrices(Trho(0), Trho(1));
            r = im.v-vals;
            
            if (std::get<0>(relative_error)){ r(0) /= vals(0); im.J.row(0) /= vals(0); }
            if (std::get<1>(relative_error)){ r(1) /= vals(1); im.J.row(1) /= vals(1); }
            
            Eigen::Array2d dTrho = im.J.matrix().fullPivLu().solve((-r).matrix());
            
            bool stop = false;
            if (apply_stopping){
                // Check whether a stopping condition (either good[complete] or bad[error])
                const StoppingData data{K, Trho, dTrho, r, nonconstant_indices};
                for (auto& condition : stopping_conditions){
                    using s = StoppingConditionReason;
                    auto this_reason = condition->stop(data);
                    if (this_reason != s::keep_going){
                        stop = true; reason = this_reason; msg = condition->desc(); break;
                    }
                }
            }
            
            Trho += dTrho;
            step_counter++;
            if (stop){
                break;
            }
        }
        return reason;
    }
    
    
    /** Take a given number of steps, with log(rho) as the density variable rather than rho, which helps when rho is VERY small to ensure that you can't have negative density
     * \param N The number of steps to take
     */
    auto take_steps_logrho(int N){
        if (N <= 0){
            throw teqp::InvalidArgument("N must be greater than 0");
        }
        auto tic = std::chrono::steady_clock::now();
        StoppingConditionReason reason = StoppingConditionReason::fatal;
        
        for (auto K = 0; K < N; ++K){
            auto im = calc_matrices(Trho(0), Trho(1));
            r = im.v-vals;
            
            im.J.col(1) *= Trho(1); // This will make the step be [dT, dln(rho)] instead of [dT, drho]
            if (std::get<0>(relative_error)){ r(0) /= vals(0); im.J.row(0) /= vals(0); }
            if (std::get<1>(relative_error)){ r(1) /= vals(1); im.J.row(1) /= vals(1); }
            
//            Eigen::JacobiSVD<Eigen::Matrix2d> svd(im.J);
//            double cond = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size()-1);
            
            Eigen::Array2d step;
            if (false){//cond > 1){
                // The idea in this block was to use extended precision to obtain the step
                // but it seems to make no difference at all, because the condition number is a
                // function of the problem, but the loss in precision is only meaningful relative to
                // the number of digits of working precision available via the lost digits approximated
                // by log10(kappa), where kappa is the condition number
                using my_float = boost::multiprecision::number<boost::multiprecision::cpp_bin_float<164U>>;
                Eigen::Matrix<my_float, 2, 2> Jmp = im.J.cast<my_float>();
                Eigen::Vector<my_float, 2> rmp = r.cast<my_float>();
                step = Jmp.fullPivLu().solve(-rmp).cast<double>();
                Eigen::Array<my_float, 2, -1> newvals = (Trho.cast<my_float>() + Jmp.fullPivLu().solve(-rmp).array());
                
                Eigen::Array2d new_double = (Trho + im.J.matrix().fullPivLu().solve((-r).matrix()).array());
                std::cout << "-- " << ((newvals-new_double.cast<my_float>())/Trho.cast<my_float>()).cast<double>() << std::endl;
            }
            else{
                step = im.J.matrix().fullPivLu().solve((-r).matrix());
            }
            Eigen::Array2d dTrho = (Eigen::Array2d() << step(0), Trho(1)*(exp(step(1))-1)).finished();
            
            // Check whether a stopping condition (either good[complete] or bad[error])
            bool stop = false;
            const StoppingData data{K, Trho, dTrho, r, nonconstant_indices};
            for (auto& condition : stopping_conditions){
                using s = StoppingConditionReason;
                auto this_reason = condition->stop(data);
                if (this_reason != s::keep_going){
                    stop = true; reason = this_reason; msg = condition->desc(); break;
                }
            }
            
            Trho += dTrho;
            step_counter++;
            
            if(nonconstant_indices.size() == 0){
                stop = true; reason = StoppingConditionReason::success; msg = "Only one step is needed for DT inputs";
            }
//            if(nonconstant_indices.size() == 2){
//                // If the step size gets too small on a relative basis then you
//                // need to stop
//                // There is a loss in precision caused by the Jacobian having a large condition number
//                // and use that to determine whether you need to stop stepping
//                //
//                // The condition number only is meaningful for 2x2 systems where both variables
//                // are being iterated for.
//                if ((dTrho/Trho)(nonconstant_indices).abs().minCoeff() < 1e-15*cond){
//                    std::stringstream ss; ss << im.J;
//                    stop = true; reason = StoppingConditionReason::success; msg = "MinRelStepSize ~= cond of " + std::to_string(cond) + ": J" + ss.str();
//                }
//            }
            if (verbose){
//                std::cout << step_counter << "," << r.abs().maxCoeff() << "," << Trho << "|" << (dTrho/Trho).abs().minCoeff() << "cond: " << cond << std::endl;
//                std::cout << step_counter << "," << r.abs().maxCoeff() << "," << Trho << "|" << (dTrho/Trho).abs().minCoeff() << im.J.matrix() << im.v-vals << "cond: " << cond << std::endl;
            }
            if (stop){
                break;
            }
        }
        auto toc = std::chrono::steady_clock::now();
//        std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::nanoseconds>(toc - tic).count()/1e3/step_counter << "[µs/call]" << std::endl;
        return reason;
    }
    auto path_integration(double T, double rho, std::size_t N){
        // Start at the given value of T, rho
        
        // Calculate the given variables
        auto im = calc_matrices(T, rho);

        Eigen::Array2d vals_current = im.v;
        double x_current = vals_current(0);
        double y_current = vals_current(1);
        
        // Assume (for now) a linear path between starting point and the target point
        double x_target = vals(0);
        double y_target = vals(1);
        auto dxdt = (x_target-x_current)/(1.0-0.0); // The 1.0 to make clear what is going on; we are integrating from 0 -> 1
        auto dydt = (y_target-y_current)/(1.0-0.0);
        double dt = 1.0/N;
        
        // Determine the value of the residual function at the end point
        for (auto i = 0U; i < N; ++i){
            
            auto dxdT__rho = im.J(0, 0);
            auto dxdrho__T = im.J(0, 1);
            auto dydT__rho = im.J(1, 0);
            auto dydrho__T = im.J(1, 1);
            
            auto dxdT__y = dxdT__rho - dxdrho__T*dydT__rho/dydrho__T;
            auto dxdrho__y = dxdrho__T - dxdT__rho*dydrho__T/dydT__rho;
            
            auto dydT__x = dydT__rho - dydrho__T*dxdT__rho/dxdrho__T;
            auto dydrho__x = dydrho__T - dydT__rho*dxdrho__T/dxdT__rho;
            
            // Calculate the increment in temperature and density along the integration path (needs to be adaptive eventually)
            double dTdt = dxdt/dxdT__y + dydt/dydT__x;
            double drhodt = dxdt/dxdrho__y + dydt/dydrho__x;
            
            T += dTdt*dt;
            rho += drhodt*dt;
            im = calc_matrices(T, rho);
            if (verbose){
                std::cout << i << "," << (im.v-vals).abs().maxCoeff() << std::endl;
            }
        }
        return std::make_tuple(T, rho, im.v(0), im.v(1));
    }
};


}
}
