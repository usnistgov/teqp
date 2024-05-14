#pragma once

#include "teqp/cpp/teqpcpp.hpp"
#include "teqp/cpp/derivs.hpp"

namespace teqp {
namespace iteration {

using namespace cppinterface;
/**
 A class for doing Newton-Raphson steps to solve for two unknown thermodynamic variables
 
 */
class NRIterator{
private:
    const AbstractModel *ar, *aig;
    const std::vector<char>  vars;
    const Eigen::ArrayXd vals;
    double T, rho;
    const Eigen::Ref<const Eigen::ArrayXd> z;
    
public:
    NRIterator(const AbstractModel* ar, const AbstractModel* aig, const std::vector<char>& vars, const Eigen::ArrayXd& vals, double T, double rho, const Eigen::Ref<const Eigen::ArrayXd>& z) : ar(ar), aig(aig), vars(vars), vals(vals), T(T), rho(rho), z(z){}
    
    /// Return the variables that are being used in the iteration
    std::vector<char> get_vars() const { return vars; }
    /// Return the target values to be obtained
    Eigen::ArrayXd get_vals() const { return vals; }
    /// Return the current temperature
    auto get_T() const { return T; }
    /// Return the current molar density
    auto get_rho() const { return rho; }
    /// Return the current mole fractions
    Eigen::ArrayXd get_molefrac() const { return z; }
    
    /** Do the calculations needed for the step and return the step and the other data
    *  In C++, the copy will be elided (the return value will be moved)
    * \param T Temperature
    * \param rho Molar density
    */
    auto calc_step(double T, double rho) const{
        auto Ar = ar->get_deriv_mat2(T, rho, z);
        auto Aig = aig->get_deriv_mat2(T, rho, z);
        auto R = ar->get_R(z);
        auto im = build_iteration_Jv(vars, Ar, Aig, R, T, rho, z);
        return std::make_tuple((im.J.matrix().colPivHouseholderQr().solve((-(im.v-vals)).matrix())).eval(), im);
    }
    
    /// Take one step, return the residuals
    auto take_step(){
        auto [dx, im] = calc_step(T, rho);
        T += dx(0);
        rho += dx(1);
        return (im.v-vals).eval();
    }
    
    /// Take one step, return the max(abs(residuals))
    auto take_step_getmaxabsr(){
        auto [dx, im] = calc_step(T, rho);
        T += dx(0);
        rho += dx(1);
        auto r = (im.v-vals).eval();
        return r.abs().maxCoeff();
    }
    
    /** Take a given number of steps
     * \param N The number of steps to take
     */
    auto take_steps(int N){
        if (N <= 0){
            throw teqp::InvalidArgument("N must be greater than 0");
        }
        for (auto i = 0; i < N; ++i){
            take_step();
        }
    }
};


}
}
