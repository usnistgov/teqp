#pragma once

#include <optional>
#include "teqp/derivs.hpp"
#include <Eigen/Dense>

// Imports from boost for numerical integration
#include <boost/numeric/odeint/stepper/controlled_runge_kutta.hpp>
#include <boost/numeric/odeint/stepper/runge_kutta_cash_karp54.hpp>
#include <boost/numeric/odeint/stepper/euler.hpp>

template<typename Model, typename TYPE = double>
class IsothermPureVLEResiduals  {
    typedef Eigen::Array<TYPE, 2, 1> EigenArray;
    typedef Eigen::Array<TYPE, 1, 1> EigenArray1;
    typedef Eigen::Array<TYPE, 2, 2> EigenMatrix;
private:
    const Model& m_model;
    TYPE m_T;
    EigenMatrix J;
    EigenArray y;
    
public:
    std::size_t icall = 0;
    double Rr, R0;

    IsothermPureVLEResiduals(const Model& model, TYPE T) : m_model(model), m_T(T) {
        std::valarray<double> molefrac = { 1.0 };
        Rr = m_model.R(molefrac);
        R0 = m_model.R(molefrac);
    };

    const auto& get_errors() { return y; };

    auto call(const EigenArray& rhovec) {
        assert(rhovec.size() == 2);

        const EigenArray1 rhovecL = rhovec.head(1);
        const EigenArray1 rhovecV = rhovec.tail(1);
        const auto rhomolarL = rhovecL.sum(), rhomolarV = rhovecV.sum();
        const auto molefracs = (EigenArray1() << 1.0).finished();

        using id = IsochoricDerivatives<Model,TYPE,EigenArray1>;
        using tdx = TDXDerivatives<Model,TYPE,EigenArray1>;

        const TYPE &T = m_T;
        const TYPE R = m_model.R(molefracs);
        double R0_over_Rr = R0 / Rr;
        
        auto derL = tdx::template get_Ar0n<2>(m_model, T, rhomolarL, molefracs);
        auto pRTL = rhomolarL*(R0_over_Rr + derL[1]); // p/(R*T)
        auto dpRTLdrhoL = R0_over_Rr + 2*derL[1] + derL[2];
        auto hatmurL = derL[1] + derL[0] + R0_over_Rr*log(rhomolarL);
        auto dhatmurLdrho = (2*derL[1] + derL[2])/rhomolarL + R0_over_Rr/rhomolarL;

        auto derV = tdx::template get_Ar0n<2>(m_model, T, rhomolarV, molefracs);
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
    auto Jacobian(const EigenArray& rhovec){
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

template<typename Residual, typename Scalar>
Eigen::ArrayXd do_pure_VLE_T(Residual &resid, Scalar rhoL, Scalar rhoV, int maxiter) {
    auto rhovec = (Eigen::ArrayXd(2) << rhoL, rhoV).finished();
    auto r0 = resid.call(rhovec);
    auto J = resid.Jacobian(rhovec);
    for (int iter = 0; iter < maxiter; ++iter){
        if (iter > 0) {
            r0 = resid.call(rhovec);
            J = resid.Jacobian(rhovec); 
        }
        auto v = J.matrix().colPivHouseholderQr().solve(-r0.matrix()).array().eval();
        auto rhovecnew = (rhovec + v).eval();
        
        // If the solution has stopped improving, stop. The change in rhovec is equal to v in infinite precision, but 
        // not when finite precision is involved, use the minimum non-denormal float as the determination of whether
        // the values are done changing
        if (((rhovecnew - rhovec).cwiseAbs() < std::numeric_limits<Scalar>::min()).all()) {
            break;
        }
        rhovec = rhovecnew;
    }
    return (Eigen::ArrayXd(2) << rhovec[0], rhovec[1]).finished();
}

template<typename Model, typename Scalar>
Eigen::ArrayXd pure_VLE_T(const Model& model, Scalar T, Scalar rhoL, Scalar rhoV, int maxiter) {
    auto res = IsothermPureVLEResiduals(model, T);
    return do_pure_VLE_T(res, rhoL, rhoV, maxiter);
}

template<typename Model, typename Scalar>
Eigen::ArrayXd extrapolate_from_critical(const Model& model, const Scalar Tc, const Scalar rhoc, const Scalar T) {
    
    using tdx = TDXDerivatives<Model>;
    auto z = (Eigen::ArrayXd(1) << 1.0).finished();
    auto R = model.R(z);
    auto ders = tdx::template get_Ar0n<4>(model, Tc, rhoc, z);
    auto dpdrho = R*Tc*(1 + 2 * ders[1] + ders[2]); // Should be zero
    auto d2pdrho2 = R*Tc/rhoc*(2 * ders[1] + 4 * ders[2] + ders[3]); // Should be zero
    auto d3pdrho3 = R*Tc/(rhoc*rhoc)*(6 * ders[2] + 6 * ders[3] + ders[4]);
    auto Ar11 = tdx::template get_Ar11(model, Tc, rhoc, z);
    auto Ar12 = tdx::template get_Ar12(model, Tc, rhoc, z);
    auto d2pdrhodT = R * (1 + 2 * ders[1] + ders[2] - 2 * Ar11 - Ar12);
    auto Brho = sqrt(6*d2pdrhodT*Tc/d3pdrho3);

    auto drhohat_dT = Brho / Tc;
    auto dT = T - Tc;

    auto drhohat = dT * drhohat_dT;
    auto rholiq = -drhohat/sqrt(1 - T/Tc) + rhoc;
    auto rhovap = drhohat/sqrt(1 - T/Tc) + rhoc;
    return (Eigen::ArrayXd(2) << rholiq, rhovap).finished();
}

template<class A, class B>
auto linsolve(const A &a, const B& b) {
    return a.matrix().colPivHouseholderQr().solve(b.matrix()).array().eval();
}

template<class Model, class Scalar, class VecType>
auto get_drhovecdp_Tsat(const Model& model, const Scalar &T, const VecType& rhovecL, const VecType& rhovecV) {
    //tic = timeit.default_timer();
    using id = IsochoricDerivatives<Model, Scalar, VecType>;
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Hliq = id::build_Psi_Hessian_autodiff(model, T, rhovecL).eval();
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Hvap = id::build_Psi_Hessian_autodiff(model, T, rhovecV).eval();
    //Hvap[~np.isfinite(Hvap)] = 1e20;
    //Hliq[~np.isfinite(Hliq)] = 1e20;

    auto N = rhovecL.size();
    Eigen::MatrixXd A = decltype(Hliq)::Zero(N, N);
    auto b = decltype(Hliq)::Ones(N, 1);
    decltype(Hliq) drhodp_liq, drhodp_vap;
    assert(rhovecL.size() == rhovecV.size());
    if ((rhovecL != 0).all() && (rhovecV != 0).all()) {
        // Normal treatment for all concentrations not equal to zero
        A(0, 0) = Hliq.row(0).dot(rhovecV.matrix());
        A(0, 1) = Hliq.row(1).dot(rhovecV.matrix());
        A(1, 0) = Hliq.row(0).dot(rhovecL.matrix());
        A(1, 1) = Hliq.row(1).dot(rhovecL.matrix());

        drhodp_liq = linsolve(A, b);
        drhodp_vap = linsolve(Hvap, Hliq*drhodp_liq);
    }
    else{
        // Special treatment for infinite dilution
        auto murL = id::build_Psir_gradient_autodiff(model, T, rhovecL);
        auto murV = id::build_Psir_gradient_autodiff(model, T, rhovecV);
        auto RL = model.R(rhovecL / rhovecL.sum());
        auto RV = model.R(rhovecV / rhovecV.sum());

        // First, for the liquid part
        for (auto i = 0; i < N; ++i) {
            for (auto j = 0; j < N; ++j) {
                if (rhovecL[j] == 0) {
                    // Analysis is special if j is the index that is a zero concentration.If you are multiplying by the vector
                    // of liquid concentrations, a different treatment than the case where you multiply by the vector
                    // of vapor concentrations is required
                    // ...
                    // Initial values
                    auto Aij = (Hliq.row(j).array().cwiseProduct(((i == 0) ? rhovecV : rhovecL).array().transpose())).eval(); // coefficient - wise product
                    // A throwaway boolean for clarity
                    bool is_liq = (i == 1);
                    // Apply correction to the j term (RT if liquid, RT*phi for vapor)
                    Aij[j] = (is_liq) ? RL*T : RL*T*exp(-(murV[j] - murL[j])/(RL*T));
                    // Fill in entry
                    A(i, j) = Aij.sum();
                }
                else{
                    // Normal
                    A(i, j) = Hliq.row(j).dot(((i==0) ? rhovecV : rhovecL).matrix());
                }
            }
        }
        drhodp_liq = linsolve(A, b);

        // Then, for the vapor part, also requiring special treatment
        // Left - multiplication of both sides of equation by diagonal matrix with liquid concentrations along diagonal, all others zero
        auto diagrhovecL = rhovecL.matrix().asDiagonal();
        auto PSIVstar = (diagrhovecL*Hvap).eval();
        auto PSILstar = (diagrhovecL*Hliq).eval();
        for (auto j = 0; j < N; ++j) {
            if (rhovecL[j] == 0) {
                PSILstar(j, j) = RL*T;
                PSIVstar(j, j) = RV*T/exp(-(murV[j] - murL[j]) / (RV * T));
            }
        }
        drhodp_vap = linsolve(PSIVstar, PSILstar*drhodp_liq);
    }
    return std::make_tuple(drhodp_liq, drhodp_vap);
}

struct TVLEOptions {
    double init_dt = 1e-5, abs_err = 1e-8, rel_err = 1e-8, max_dt = 100000;
    int max_steps = 1000, integration_order = 5;
    bool polish = false;
};

/***
* \brief Trace an isotherm with parametric tracing
*/
template<typename Model, typename Scalar, typename VecType>
auto trace_VLE_isotherm_binary(const Model &model, Scalar T, VecType rhovecL0, VecType rhovecV0, const std::optional<TVLEOptions>& options = std::nullopt) 
{
    // Get the options, or the default values if not provided
    TVLEOptions opt = options.value_or(TVLEOptions{});
    auto N = rhovecL0.size();
    if (N != 2) {
        throw std::invalid_argument("Size must be 2");
    }
    if (rhovecL0.size() != rhovecV0.size()) {
        throw std::invalid_argument("Both molar concentration arrays must be of the same size");
    }

    auto norm = [](const auto& v) { return sqrt((v * v).sum()); };

    // Define datatypes and functions for tracing tools
    auto JSONdata = nlohmann::json::array();

    // Typedefs for the types
    using namespace boost::numeric::odeint;
    using state_type = std::vector<double>;

    // Class for simple Euler integration
    euler<state_type> eul;
    typedef runge_kutta_cash_karp54< state_type > error_stepper_type;
    typedef controlled_runge_kutta< error_stepper_type > controlled_stepper_type;

    // Define the tolerances
    double abs_err = opt.abs_err, rel_err = opt.rel_err, a_x = 1.0, a_dxdt = 1.0;
    controlled_stepper_type controlled_stepper(default_error_checker< double, range_algebra, default_operations >(abs_err, rel_err, a_x, a_dxdt));


    double c = 1.0;

    // The function to be integrated by odeint
    auto xprime = [&](const state_type& X, state_type& Xprime, double /*t*/) {
        // Memory maps into the state vector for inputs and their derivatives
        // These are views, not copies!
        auto rhovecL = Eigen::Map<const Eigen::ArrayXd>(&(X[0]), N);
        auto rhovecV = Eigen::Map<const Eigen::ArrayXd>(&(X[0]) + N, N);
        auto drhovecdtL = Eigen::Map<Eigen::ArrayXd>(&(Xprime[0]), N);
        auto drhovecdtV = Eigen::Map<Eigen::ArrayXd>(&(Xprime[0]) + N, N);
        // Get the derivatives with respect to pressure along the isotherm of the phase envelope
        auto [drhovecdpL, drhovecdpV] = get_drhovecdp_Tsat(model, T, rhovecL, rhovecV);
        // Get the derivative of p w.r.t. parameter
        auto dpdt = 1.0/sqrt(norm(drhovecdpL.array()) + norm(drhovecdpV.array()));
        // And finally the derivatives with respect to the tracing variable
        drhovecdtL = c*drhovecdpL*dpdt;
        drhovecdtV = c*drhovecdpV*dpdt;
    };

    // Set up the initial state vector
    state_type x0(2*N), last_drhodt(2*N);
    auto set_init_state = [&](state_type& X) {
        auto rhovecL = Eigen::Map<Eigen::ArrayXd>(&(X[0]), N);
        auto rhovecV = Eigen::Map<Eigen::ArrayXd>(&(X[0]) + N, N);
        rhovecL = rhovecL0;
        rhovecV = rhovecV0;
    };
    set_init_state(x0);
    
    // Figure out which direction to trace initially
    double t = 0, dt = opt.init_dt;
    {
        auto dxdt = x0;
        xprime(x0, dxdt, -1.0);
        const auto dXdt = Eigen::Map<const Eigen::ArrayXd>(&(dxdt[0]), dxdt.size());
        const auto X = Eigen::Map<const Eigen::ArrayXd>(&(x0[0]), x0.size());

        const Eigen::ArrayXd step = X + dXdt * dt;
        Eigen::ArrayX<bool> negativestepvals = (step < 0).eval();
        // Flip the sign if the first step would yield any negative concentrations
        if (negativestepvals.any()) {
            c *= -1;
        }
    }
    std::string termination_reason;
    
    // Then trace...
    int retry_count = 0;
    for (auto istep = 0; istep < opt.max_steps; ++istep) {

        auto store_point = [&]() {
            //// Calculate some other parameters, for debugging
            auto N = x0.size() / 2;
            auto rhovecL = Eigen::Map<const Eigen::ArrayXd>(&(x0[0]), N);
            auto rhovecV = Eigen::Map<const Eigen::ArrayXd>(&(x0[0]) + N, N);
            auto rhotot = rhovecL.sum();
            using id = IsochoricDerivatives<decltype(model), Scalar, VecType>;
            double pL = rhotot * model.R(rhovecL / rhovecL.sum())*T + id::get_pr(model, T, rhovecL);

            // Store the data in a JSON structure
            nlohmann::json point = {
                {"t", t},
                {"T / K", T},
                {"pL / Pa", pL},
                {"c", c},
                {"rhoL / mol/m^3", rhovecL},
                {"rhoV / mol/m^3", rhovecV},
                
            };
            JSONdata.push_back(point);
        };
        if (istep == 0 && retry_count == 0) {
            store_point();
        }

        double dtold = dt;
        auto x0_previous = x0;

        // Call the derivative function to get drho/dt in order 
        // to cache it at the *beginning* of the step
        auto dxdt = x0;
        try {
            xprime(x0, dxdt, -1.0);
        }
        catch (...) {
            break;
        }
        last_drhodt = dxdt;

        if (opt.integration_order == 5) {
            controlled_step_result res = controlled_step_result::fail;
            try {
                res = controlled_stepper.try_step(xprime, x0, t, dt);
            }
            catch (...) {
                break;
            }

            if (res != controlled_step_result::success) {
                // Try again, with a smaller step size
                istep--;
                retry_count++;
                continue;
            }
            else {
                retry_count = 0;
            }
            // Reduce step size if greater than the specified max step size
            dt = std::min(dt, opt.max_dt);
        }
        else if (opt.integration_order == 1) {
            try {
                eul.do_step(xprime, x0, t, dt);
            }
            catch (...) {
                break;
            }
        }
        else {
            throw std::invalid_argument("integration order is invalid:" + std::to_string(opt.integration_order));
        }
        auto stop_requested = [&]() {
            //// Calculate some other parameters, for debugging
            auto N = x0.size() / 2;
            auto rhovecL = Eigen::Map<const Eigen::ArrayXd>(&(x0[0]), N);
            auto rhovecV = Eigen::Map<const Eigen::ArrayXd>(&(x0[0]) + N, N);
            auto x = rhovecL / rhovecL.sum();
            auto y = rhovecV / rhovecV.sum();
            if ((x < 0).any() || (x > 1).any() || (y < 0).any() || (y > 1).any()) {
                return true;
            }
            else {
                return false;
            }
        };
        if (stop_requested()) {
            break;
        }

        store_point();
    }
    return JSONdata;
}