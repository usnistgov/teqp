//
//  VLE_pure.hpp
//  teqp
//
//  Created by Bell, Ian H. (Fed) on 5/3/23.
//
#pragma once
#include <type_traits>
#include "teqp/cpp/teqpcpp.hpp"
#include "teqp/derivs.hpp"
#include "teqp/algorithms/critical_pure.hpp"

namespace teqp{

template<typename Model> using is_AbstractModel = typename std::is_base_of<teqp::cppinterface::AbstractModel, Model>;
template<typename Model> using is_not_AbstractModel = std::negation<is_AbstractModel<Model>>;

template<typename Model, typename TYPE=double, ADBackends backend = ADBackends::autodiff>
class IsothermPureVLEResiduals  {
public:
    using EigenArray = Eigen::Array<TYPE, 2, 1>;
    using EigenArray1 = Eigen::Array<TYPE, 1, 1>;
    using EigenMatrix = Eigen::Array<TYPE, 2, 2>;
private:
    const Model& m_model;
    const TYPE m_T;
    const Eigen::ArrayXd molefracs;
    EigenMatrix J;
    EigenArray y;
public:
    std::size_t icall = 0;
    double Rr, R0;

    IsothermPureVLEResiduals(const Model& model, const TYPE& T, const std::optional<Eigen::ArrayXd>& molefracs_ = std::nullopt) : m_model(model), m_T(T),
        molefracs( (molefracs_) ? molefracs_.value() : Eigen::ArrayXd::Ones(1,1)) {
            if constexpr(is_not_AbstractModel<Model>::value){
                Rr = m_model.R(molefracs);
                R0 = m_model.R(molefracs);
            }
            else{
                Rr = m_model.get_R(molefracs);
                R0 = m_model.get_R(molefracs);
            }
    };

    const auto& get_errors() { return y; };
    
    template<typename Rho>
    auto get_der(const Rho& rho){
        if constexpr(is_not_AbstractModel<Model>::value){
            using tdx = TDXDerivatives<Model,TYPE,Eigen::ArrayXd>;
            return tdx::template get_Ar0n<2, backend>(m_model, m_T, rho, molefracs);
        }
        else{
            return m_model.get_Ar02n(m_T, rho, molefracs);
        }
    }

    auto call(const EigenArray& rhovec) {
        assert(rhovec.size() == 2);

        const EigenArray1 rhovecL = rhovec.head(1);
        const EigenArray1 rhovecV = rhovec.tail(1);
        const auto rhomolarL = rhovecL.sum(), rhomolarV = rhovecV.sum();

        //const TYPE R = m_model.R(molefracs);
        double R0_over_Rr = R0 / Rr;
        
        auto derL = get_der(rhomolarL);
        auto pRTL = rhomolarL*(R0_over_Rr + derL[1]); // p/(R*T)
        auto dpRTLdrhoL = R0_over_Rr + 2*derL[1] + derL[2];
        auto hatmurL = derL[1] + derL[0] + R0_over_Rr*log(rhomolarL);
        auto dhatmurLdrho = (2*derL[1] + derL[2])/rhomolarL + R0_over_Rr/rhomolarL;

        auto derV = get_der(rhomolarV);
        auto pRTV = rhomolarV*(R0_over_Rr + derV[1]); // p/(R*T)
        auto dpRTVdrhoV = R0_over_Rr + 2*derV[1] + derV[2];
        auto hatmurV = derV[1] + derV[0] + R0_over_Rr *log(rhomolarV);
        auto dhatmurVdrho = (2*derV[1] + derV[2])/rhomolarV + R0_over_Rr/rhomolarV;

        y(0) = pRTL - pRTV;
        J(0, 0) = dpRTLdrhoL;
        J(0, 1) = -dpRTVdrhoV;

        y(1) = hatmurL - hatmurV;
        J(1, 0) = dhatmurLdrho;
        J(1, 1) = -dhatmurVdrho;

        icall++;
        return y;
    }
    auto Jacobian(const EigenArray& /*rhovec*/){
        return J;
    }
    //auto numJacobian(const EigenArray& rhovec) {
    //    EigenArray plus0 = rhovec, plus1 = rhovec;
    //    double dr = 1e-6 * rhovec[0];
    //    plus0[0] += dr; plus1[1] += dr;
    //    EigenMatrix J;
    //    J.col(0) = (call(plus0) - call(rhovec))/dr;
    //    J.col(1) = (call(plus1) - call(rhovec))/dr;
    //    return J;
    //}
};

template<typename Residual, typename Scalar=double>
auto do_pure_VLE_T(Residual &resid, Scalar rhoL, Scalar rhoV, int maxiter) {
    using EArray = Eigen::Array<Scalar, 2, 1>;
    auto rhovec = (EArray() << rhoL, rhoV).finished();
    auto r0 = resid.call(rhovec);
    auto J = resid.Jacobian(rhovec);
    for (int iter = 0; iter < maxiter; ++iter){
        if (iter > 0) {
            r0 = resid.call(rhovec);
            J = resid.Jacobian(rhovec);
        }
        auto v = J.matrix().colPivHouseholderQr().solve(-r0.matrix()).array().eval();
        auto rhovecnew = (rhovec + v).eval();
        //double r00 = static_cast<double>(r0[0]);
        //double r01 = static_cast<double>(r0[1]);
        
        // If the solution has stopped improving, stop. The change in rhovec is equal to v in infinite precision, but
        // not when finite precision is involved, use the minimum non-denormal float as the determination of whether
        // the values are done changing
        auto minval = std::numeric_limits<Scalar>::epsilon();
        //double minvaldbl = static_cast<double>(minval);
        if (((rhovecnew - rhovec).cwiseAbs() < minval).all()) {
            break;
        }
        if ((r0.cwiseAbs() < minval).all()) {
            break;
        }
        rhovec = rhovecnew;
    }
    return rhovec;
}

inline auto pure_VLE_T(const teqp::cppinterface::AbstractModel& model, double T, double rhoL, double rhoV, int maxiter, const std::optional<Eigen::ArrayXd>& molefracs = std::nullopt) {
    Eigen::ArrayXd molefracs_{Eigen::ArrayXd::Ones(1,1)};
    if (molefracs){ molefracs_ = molefracs.value(); }
    auto res = IsothermPureVLEResiduals<teqp::cppinterface::AbstractModel>(model, T, molefracs_);
    return do_pure_VLE_T(res, rhoL, rhoV, maxiter);
}

/***
 * \brief Calculate the derivative of vapor pressure with respect to temperature
 * \param model The model to operate on
 * \param T Temperature
 * \param rhoL Liquid density
 * \param rhoV Vapor density
 *
 *  Based upon
 *  \f[
 * \frac{dp_{\sigma}}{dT} = \frac{h''-h'}{T(v''-v')} = \frac{s''-s'}{v''-v'}
 *  \f]
 *  where the \f$h''-h'\f$ is given by the difference in residual enthalpy \f$h''-h' = h^r''-h^r'\f$ because the ideal-gas parts cancel
 */
inline auto dpsatdT_pure(const teqp::cppinterface::AbstractModel& model, double T, double rhoL, double rhoV, const std::optional<Eigen::ArrayXd>& molefracs = std::nullopt) {
    
    auto molefrac = (Eigen::ArrayXd(1) << 1.0).finished();
    if (molefracs){ molefrac = molefracs.value(); }
    
    auto R = model.get_R(molefrac);
    
    auto hrVLERTV = model.get_Ar01(T, rhoV, molefrac) + model.get_Ar10(T, rhoV, molefrac);
    auto hrVLERTL = model.get_Ar01(T, rhoL, molefrac) + model.get_Ar10(T, rhoL, molefrac);
    auto deltahr_over_T = R*(hrVLERTV-hrVLERTL);
    auto dpsatdT = deltahr_over_T/(1/rhoV-1/rhoL); // From Clapeyron; dp/dT = Deltas/Deltav = Deltah/(T*Deltav); Delta=V-L
    return dpsatdT;
}

/***
 \brief Starting at the critical point, trace the VLE down to a temperature of interest
 
 \note This method only works for well-behaved EOS, notably absent from that category are EOS in the multiparameter category with orthobaric scaling exponent not equal to 0.5 at the critical point. Most other analytical EOS work fine
 
 The JSON data structure defines the variables that need to be specified.
 
 In the current implementation, there are a few steps:
 1. Solve for the true critical point satisfying \f$(\partial p/\partial \rho)_{T}=(\partial^2p/\partial\rho^2)_{T}=0\f$
 2. Take a small step away from the critical point (this is where the beta=0.5 assumption is invoked)
 3. Integrate from the near critical temperature to the temperature of interest
 */
inline auto pure_trace_VLE(const teqp::cppinterface::AbstractModel& model, const double T, const nlohmann::json &spec){
    // Start at the true critical point, from the specified guess value
    nlohmann::json pure_spec;
    Eigen::ArrayXd z{Eigen::ArrayXd::Ones(1,1)};
    if (spec.contains("pure_spec")){
        pure_spec = spec.at("pure_spec");
        z = Eigen::ArrayXd(pure_spec.at("alternative_length").get<int>()); z.setZero();
        z(pure_spec.at("alternative_pure_index").get<int>()) = 1;
    }
    
    auto [Tc, rhoc] = solve_pure_critical(model, spec.at("Tcguess").get<double>(), spec.at("rhocguess").get<double>(), pure_spec);
    
    // Small step towards lower temperature close to critical temperature
    double Tclose = spec.at("Tred").get<double>()*Tc;
    auto rhoLrhoV = extrapolate_from_critical(model, Tc, rhoc, Tclose, z);
    auto rhoLrhoVpolished = pure_VLE_T(model, Tclose, rhoLrhoV[0], rhoLrhoV[1], spec.value("NVLE", 10), z);
    if (rhoLrhoVpolished[0] == rhoLrhoVpolished[1]){
        throw teqp::IterationError("Converged to trivial solution");
    }
    
    // "Integrate" down to temperature of interest
    int Nstep = spec.at("Nstep");
    double R = model.R(z);
    bool with_deriv = spec.at("with_deriv");
    double dT = -(Tclose-T)/(Nstep-1);
    
    for (auto T_: Eigen::ArrayXd::LinSpaced(Nstep, Tclose, T)){
        rhoLrhoVpolished = pure_VLE_T(model, T_, rhoLrhoVpolished[0], rhoLrhoVpolished[1], spec.value("NVLE", 10), z);
        
        //auto pL = rhoLrhoVpolished[0]*R*T_*(1 + tdx::get_Ar01(model, T_, rhoLrhoVpolished[0], z));
        //auto pV = rhoLrhoVpolished[1]*R*T_*(1 + tdx::get_Ar01(model, T_, rhoLrhoVpolished[1], z));
        //std::cout << pL << " " << pV << " " << pL/pV-1 <<  std::endl;
        if (with_deriv){
            // Get drho/dT for both phases
            auto dpsatdT = dpsatdT_pure(model, T_, rhoLrhoVpolished[0], rhoLrhoVpolished[1], z);
            auto get_drhodT = [&z, R, dpsatdT](const AbstractModel& model, double T, double rho){
                auto dpdrho = R*T*(1 + 2*model.get_Ar01(T, rho, z) + model.get_Ar02(T, rho, z));
                auto dpdT = R*rho*(1 + model.get_Ar01(T, rho, z) - model.get_Ar11(T, rho, z));
                return -dpdT/dpdrho + dpsatdT/dpdrho;
            };
            auto drhodTL = get_drhodT(model, T_, rhoLrhoVpolished[0]);
            auto drhodTV = get_drhodT(model, T_, rhoLrhoVpolished[1]);
            // Use the obtained derivative to calculate the step in rho from deltarho = (drhodT)*dT
            auto DeltarhoL = dT*drhodTL, DeltarhoV = dT*drhodTV;
            rhoLrhoVpolished[0] += DeltarhoL;
            rhoLrhoVpolished[1] += DeltarhoV;
        }
        
        // Updated values for densities at new T
        if (!std::isfinite(rhoLrhoVpolished[0])){
            throw teqp::IterationError("The density is no longer valid; try increasing Nstep");
        }
        if (rhoLrhoVpolished[0] == rhoLrhoVpolished[1]){
            throw teqp::IterationError("Converged to trivial solution; try increasing Nstep");
        }
    }
    return rhoLrhoVpolished;
}

#define VLE_PURE_FUNCTIONS_TO_WRAP \
    X(dpsatdT_pure) \
    X(pure_VLE_T) \
    X(pure_trace_VLE)

#define X(f) template <typename TemplatedModel, typename ...Params, \
typename = typename std::enable_if<is_not_AbstractModel<TemplatedModel>::value>::type> \
inline auto f(const TemplatedModel& model, Params&&... params){ \
    auto view = teqp::cppinterface::adapter::make_cview(model); \
    const AbstractModel& am = *view.get(); \
    return f(am, std::forward<Params>(params)...); \
}
VLE_PURE_FUNCTIONS_TO_WRAP
#undef X
#undef VLE_PURE_FUNCTIONS_TO_WRAP

}
