#pragma once

#include "teqp/derivs.hpp"
#include "teqp/exceptions.hpp"
#include "teqp/algorithms/VLLE_types.hpp"
#include "teqp/cpp/teqpcpp.hpp"

// Imports from boost
#include <boost/numeric/odeint/stepper/controlled_runge_kutta.hpp>
#include <boost/numeric/odeint/stepper/runge_kutta_cash_karp54.hpp>

namespace teqp {
namespace VLLE {
    
    using namespace teqp::cppinterface;
    /***
    * \brief Do a vapor-liquid-liquid phase equilibrium problem for a mixture (binary only for now)
    * \param model The model to operate on
    * \param T Temperature
    * \param rhovecVinit Initial values for vapor mole concentrations
    * \param rhovecL1init Initial values for liquid #1 mole concentrations
    * \param rhovecL2init Initial values for liquid #2 mole concentrations
    * \param atol Absolute tolerance on function values
    * \param reltol Relative tolerance on function values
    * \param axtol Absolute tolerance on steps in independent variables
    * \param relxtol Relative tolerance on steps in independent variables
    * \param maxiter Maximum number of iterations permitted
    */
    
    inline auto mix_VLLE_T(const AbstractModel& model, double T, const EArrayd& rhovecVinit, const EArrayd& rhovecL1init, const EArrayd& rhovecL2init, double atol, double reltol, double axtol, double relxtol, int maxiter) {

        const Eigen::Index N = rhovecVinit.size();
        Eigen::MatrixXd J(3 * N, 3 * N); J.setZero();
        Eigen::VectorXd r(3 * N), x(3 * N);

        x.head(N) = rhovecVinit;
        x.segment(N, N) = rhovecL1init;
        x.tail(N) = rhovecL2init;

        Eigen::Map<Eigen::ArrayXd> rhovecV (&(x(0)), N);
        Eigen::Map<Eigen::ArrayXd> rhovecL1(&(x(0+N)), N);
        Eigen::Map<Eigen::ArrayXd> rhovecL2(&(x(0+2*N)), N);

        VLLE_return_code return_code = VLLE_return_code::unset;

        for (int iter = 0; iter < maxiter; ++iter) {

            auto [PsirV, PsirgradV, hessianV] = model.build_Psir_fgradHessian_autodiff(T, rhovecV);
            auto [PsirL1, PsirgradL1, hessianL1] = model.build_Psir_fgradHessian_autodiff(T, rhovecL1);
            auto [PsirL2, PsirgradL2, hessianL2] = model.build_Psir_fgradHessian_autodiff(T, rhovecL2);
            
            auto HtotV = model.build_Psi_Hessian_autodiff(T, rhovecV);
            auto HtotL1 = model.build_Psi_Hessian_autodiff(T, rhovecL1);
            auto HtotL2 = model.build_Psi_Hessian_autodiff(T, rhovecL2);

            auto zV = rhovecV/rhovecV.sum(), zL1 = rhovecL1 / rhovecL1.sum(), zL2 = rhovecL2 / rhovecL2.sum();
            double RTL1 = model.get_R(zL1)*T, RTL2 = model.get_R(zL2)*T, RTV = model.get_R(zV)*T;

            auto rhoL1 = rhovecL1.sum();
            auto rhoL2 = rhovecL2.sum();
            auto rhoV = rhovecV.sum();
            double pL1 = rhoL1 * RTL1 - PsirL1 + (rhovecL1.array() * PsirgradL1.array()).sum(); // The (array*array).sum is a dot product
            double pL2 = rhoL2 * RTL2 - PsirL2 + (rhovecL2.array() * PsirgradL2.array()).sum(); // The (array*array).sum is a dot product
            double pV = rhoV * RTV - PsirV + (rhovecV.array() * PsirgradV.array()).sum();
            auto dpdrhovecL1 = RTL1 + (hessianL1 * rhovecL1.matrix()).array();
            auto dpdrhovecL2 = RTL2 + (hessianL2 * rhovecL2.matrix()).array();
            auto dpdrhovecV = RTV + (hessianV * rhovecV.matrix()).array();

            // 2N rows are equality of chemical equilibria
            r.head(N) = PsirgradV + RTV*log(rhovecV) - (PsirgradL1 + RTL1*log(rhovecL1));
            r.segment(N,N) = PsirgradL1 + RTL1 * log(rhovecL1) - (PsirgradL2 + RTL2 * log(rhovecL2));
            // Followed by N pressure equilibria
            r(2*N) = pV - pL1;
            r(2*N+1) = pL1 - pL2;

            // Chemical potential contributions in Jacobian
            J.block(0,0,N,N) = HtotV;
            J.block(0,N,N,N) = -HtotL1;
            //J.block(0,2*N,N,N) = 0;  (following the pattern, to make clear the structure)
            //J.block(N,0,N,N) = 0;   (following the pattern, to make clear the structure)
            J.block(N, N, N, N) = HtotL1;
            J.block(N, 2 * N, N, N) = -HtotL2;
            // Pressure contributions in Jacobian
            J.block(2 * N, 0, 1, N) = dpdrhovecV.transpose();
            J.block(2 * N, N, 1, N) = -dpdrhovecL1.transpose();
            J.block(2 * N + 1, N, 1, N) = dpdrhovecL1.transpose();
            J.block(2 * N + 1, 2 * N, 1, N) = -dpdrhovecL2.transpose();

            // Solve for the step
            Eigen::ArrayXd dx = J.colPivHouseholderQr().solve(-r);
            x.array() += dx;

            auto xtol_threshold = (axtol + relxtol * x.array().cwiseAbs()).eval();
            if ((dx.array() < xtol_threshold).all()) {
                return_code = VLLE_return_code::xtol_satisfied;
                break;
            }

            auto error_threshold = (atol + reltol * r.array().cwiseAbs()).eval();
            if ((r.array().cwiseAbs() < error_threshold).all()) {
                return_code = VLLE_return_code::functol_satisfied;
                break;
            }

            // If the solution has stopped improving, stop. The change in x is equal to dx in infinite precision, but
            // not when finite precision is involved, use the minimum non-denormal float as the determination of whether
            // the values are done changing
            if (((x.array() - dx.array()).cwiseAbs() < std::numeric_limits<double>::min()).all()) {
                return_code = VLLE_return_code::xtol_satisfied;
                break;
            }
            if (iter == maxiter - 1) {
                return_code = VLLE_return_code::maxiter_met;
            }
        }
        Eigen::ArrayXd rhovecVfinal = rhovecV, rhovecL1final = rhovecL1, rhovecL2final = rhovecL2;
        return std::make_tuple(return_code, rhovecVfinal, rhovecL1final, rhovecL2final);
    }

    /***
    * \brief Do a vapor-liquid-liquid phase equilibrium problem for a mixture (binary only for now)
    * \param model The model to operate on
    * \param T Temperature
    * \param rhovecVinit Initial values for vapor mole concentrations
    * \param rhovecL1init Initial values for liquid #1 mole concentrations
    * \param rhovecL2init Initial values for liquid #2 mole concentrations
    * \param atol Absolute tolerance on function values
    * \param reltol Relative tolerance on function values
    * \param axtol Absolute tolerance on steps in independent variables
    * \param relxtol Relative tolerance on steps in independent variables
    * \param maxiter Maximum number of iterations permitted
    */

    inline auto mix_VLLE_p(const AbstractModel& model, double p, double Tinit, const EArrayd& rhovecVinit, const EArrayd& rhovecL1init, const EArrayd& rhovecL2init, double atol, double reltol, double axtol, double relxtol, int maxiter) {

        const Eigen::Index N = rhovecVinit.size();
        Eigen::MatrixXd J(3*N+1, 3*N+1); J.setZero();
        Eigen::VectorXd r(3*N+1), x(3*N+1);

        x.head(N) = rhovecVinit;
        x.segment(N, N) = rhovecL1init;
        x.segment(2*N, N) = rhovecL2init;
        x(x.size()-1) = Tinit;

        Eigen::Map<Eigen::ArrayXd> rhovecV (&(x(0)), N);
        Eigen::Map<Eigen::ArrayXd> rhovecL1(&(x(0+N)), N);
        Eigen::Map<Eigen::ArrayXd> rhovecL2(&(x(0+2*N)), N);

        VLLE_return_code return_code = VLLE_return_code::unset;

        double T = -1;
        for (int iter = 0; iter < maxiter; ++iter) {
            T = x(x.size()-1);

            auto [PsirV, PsirgradV, hessianV] = model.build_Psir_fgradHessian_autodiff(T, rhovecV);
            auto [PsirL1, PsirgradL1, hessianL1] = model.build_Psir_fgradHessian_autodiff(T, rhovecL1);
            auto [PsirL2, PsirgradL2, hessianL2] = model.build_Psir_fgradHessian_autodiff(T, rhovecL2);
            
            auto HtotV = model.build_Psi_Hessian_autodiff(T, rhovecV);
            auto HtotL1 = model.build_Psi_Hessian_autodiff(T, rhovecL1);
            auto HtotL2 = model.build_Psi_Hessian_autodiff(T, rhovecL2);

            auto zV = rhovecV/rhovecV.sum(), zL1 = rhovecL1 / rhovecL1.sum(), zL2 = rhovecL2 / rhovecL2.sum();
            double RL1 = model.get_R(zL1), RL2 = model.get_R(zL2), RV = model.get_R(zV);
            double RTL1 = RL1*T, RTL2 = RL2*T, RTV = RV*T;

            auto rhoL1 = rhovecL1.sum();
            auto rhoL2 = rhovecL2.sum();
            auto rhoV = rhovecV.sum();
            double pL1 = rhoL1 * RTL1 - PsirL1 + (rhovecL1.array() * PsirgradL1.array()).sum(); // The (array*array).sum is a dot product
            double pL2 = rhoL2 * RTL2 - PsirL2 + (rhovecL2.array() * PsirgradL2.array()).sum(); // The (array*array).sum is a dot product
            double pV = rhoV * RTV - PsirV + (rhovecV.array() * PsirgradV.array()).sum();
            auto dpdrhovecL1 = RTL1 + (hessianL1 * rhovecL1.matrix()).array();
            auto dpdrhovecL2 = RTL2 + (hessianL2 * rhovecL2.matrix()).array();
            auto dpdrhovecV = RTV + (hessianV * rhovecV.matrix()).array();
            
            auto DELTAVL1dmu_dT_res = (model.build_d2PsirdTdrhoi_autodiff(T, rhovecV.eval())
                                  - model.build_d2PsirdTdrhoi_autodiff(T, rhovecL1.eval())).eval();
            auto DELTAL1L2dmu_dT_res = (model.build_d2PsirdTdrhoi_autodiff(T, rhovecL1.eval())
                                  - model.build_d2PsirdTdrhoi_autodiff(T, rhovecL2.eval())).eval();
            auto DELTAVL1_dchempot_dT = (DELTAVL1dmu_dT_res + RV*log(rhovecV) - RL1*log(rhovecL1)).eval();
            auto DELTAL1L2_dchempot_dT = (DELTAL1L2dmu_dT_res + RL1*log(rhovecL1) - RL2*log(rhovecL2)).eval();

            // 2N rows are equality of chemical equilibria
            r.head(N) = PsirgradV + RTV*log(rhovecV) - (PsirgradL1 + RTL1*log(rhovecL1));
            r.segment(N,N) = PsirgradL1 + RTL1 * log(rhovecL1) - (PsirgradL2 + RTL2 * log(rhovecL2));
            // Followed by 2 pressure equilibria for the phases
            r(2*N) = pV - pL1;
            r(2*N+1) = pL1 - pL2;
            // And finally, the pressure must equal the specified one
            r(2*N+2) = pV - p;
            

            // Chemical potential contributions in Jacobian
            J.block(0,0,N,N) = HtotV;
            J.block(0,N,N,N) = -HtotL1;
            //J.block(0,2*N,N,N) = 0;  // For L2, following the pattern, to make clear the structure
            J.block(0,2*N+2, N, 1) = DELTAVL1_dchempot_dT;
            
            //J.block(N,0,N,N) = 0;   // For V, following the pattern, to make clear the structure)
            J.block(N,N, N, N) = HtotL1;
            J.block(N,2* N, N, N) = -HtotL2;
            J.block(N,2*N+2, N, 1) = DELTAL1L2_dchempot_dT;
            // So far, 2*N constraints...
            
            // Pressure contributions in Jacobian
            J.block(2*N, 0, 1, N) = dpdrhovecV.transpose();
            J.block(2*N, N, 1, N) = -dpdrhovecL1.transpose();
            J(2*N, 2*N+2) = model.get_dpdT_constrhovec(T, rhovecV) - model.get_dpdT_constrhovec(T, rhovecL1);
            J.block(2 * N + 1, N, 1, N) = dpdrhovecL1.transpose();
            J.block(2 * N + 1, 2 * N, 1, N) = -dpdrhovecL2.transpose();
            J(2*N+1, 2*N+2) = model.get_dpdT_constrhovec(T, rhovecL1) - model.get_dpdT_constrhovec(T, rhovecL2);
            
            J.block(2*N+2, 0, 1, N) = dpdrhovecV.transpose();
            J(2*N+2, 2*N+2) = model.get_dpdT_constrhovec(T, rhovecV);
            // Takes us to 2*N + 3 constraints, or 3*N+1 for N=2

            // Solve for the step
            Eigen::ArrayXd dx = J.colPivHouseholderQr().solve(-r);
            x.array() += dx;
            T = x(x.size()-1);

            auto xtol_threshold = (axtol + relxtol * x.array().cwiseAbs()).eval();
            if ((dx.array() < xtol_threshold).all()) {
                return_code = VLLE_return_code::xtol_satisfied;
                break;
            }

            auto error_threshold = (atol + reltol * r.array().cwiseAbs()).eval();
            if ((r.array().cwiseAbs() < error_threshold).all()) {
                return_code = VLLE_return_code::functol_satisfied;
                break;
            }

            // If the solution has stopped improving, stop. The change in x is equal to dx in infinite precision, but
            // not when finite precision is involved, use the minimum non-denormal float as the determination of whether
            // the values are done changing
            if (((x.array() - dx.array()).cwiseAbs() < std::numeric_limits<double>::min()).all()) {
                return_code = VLLE_return_code::xtol_satisfied;
                break;
            }
            if (iter == maxiter - 1) {
                return_code = VLLE_return_code::maxiter_met;
            }
        }
        double Tfinal = T;
        Eigen::ArrayXd rhovecVfinal = rhovecV, rhovecL1final = rhovecL1, rhovecL2final = rhovecL2;
        return std::make_tuple(return_code, Tfinal, rhovecVfinal, rhovecL1final, rhovecL2final);
    }

    /**
    Derived from https://stackoverflow.com/a/17931809
    */
    template<typename Iterable>
    inline auto get_self_intersections(Iterable& x, Iterable& y) {
        Eigen::Array22d A;
        std::vector<SelfIntersectionSolution> solns;
        for (auto j = 0U; j < x.size() - 1; ++j) {
            auto p0 = (Eigen::Array2d() << x[j], y[j]).finished();
            auto p1 = (Eigen::Array2d() << x[j + 1], y[j + 1]).finished();
            A.col(0) = p1 - p0;
            for (auto k = j + 1; k < x.size() - 1; ++k) {
                auto q0 = (Eigen::Array2d() << x[k], y[k]).finished();
                auto q1 = (Eigen::Array2d() << x[k + 1], y[k + 1]).finished();
                A.col(1) = q0 - q1;
                Eigen::Array2d params = A.matrix().colPivHouseholderQr().solve((q0 - p0).matrix());
                if ((params > 0).binaryExpr((params < 1), [](auto x, auto y) {return x & y; }).all()) { // Both of the params are in (0,1)
                    auto soln = p0 + params[0] * (p1 - p0);
                    solns.emplace_back(SelfIntersectionSolution{ static_cast<std::size_t>(j), static_cast<std::size_t>(k), params[0], params[1], soln[0], soln[1] });
                }
            }
        }
        return solns;
    }

    template<typename Iterable>
    inline auto get_cross_intersections(Iterable& x1, Iterable& y1, Iterable& x2, Iterable& y2) {
        Eigen::Array22d A;
        std::vector<SelfIntersectionSolution> solns;
        for (auto j = 0U; j < x1.size() - 1; ++j) {
            auto p0 = (Eigen::Array2d() << x1[j], y1[j]).finished();
            auto p1 = (Eigen::Array2d() << x1[j + 1], y1[j + 1]).finished();
            A.col(0) = p1 - p0;
            for (auto k = 0U; k < x2.size() - 1; ++k) {
                auto q0 = (Eigen::Array2d() << x2[k], y2[k]).finished();
                auto q1 = (Eigen::Array2d() << x2[k + 1], y2[k + 1]).finished();
                A.col(1) = q0 - q1;
                Eigen::Array2d params = A.matrix().colPivHouseholderQr().solve((q0 - p0).matrix());
                if ((params > 0).binaryExpr((params < 1), [](auto x, auto y) {return x & y; }).all()) { // Both of the params are in (0,1)
                    auto soln = p0 + params[0] * (p1 - p0);
                    solns.emplace_back(SelfIntersectionSolution{ static_cast<std::size_t>(j), static_cast<std::size_t>(k), params[0], params[1], soln[0], soln[1] });
                }
            }
        }
        return solns;
    }

    
    
    inline auto find_VLLE_gen_binary(const AbstractModel& model, const std::vector<nlohmann::json>& traces, const std::string& key, const std::optional<VLLEFinderOptions> options = std::nullopt) {
        std::vector<double> x, y;
        auto opt = options.value_or(VLLEFinderOptions{});

        Eigen::ArrayXd rhoL1(2), rhoL2(2), rhoV(2);
        std::string xkey = (key == "T") ? "T / K" : "pL / Pa";
        std::string ykey = (key == "T") ? "pL / Pa" : "T / K";
        
        // A convenience function to weight the values
        auto avg_values = [](const nlohmann::json&j1, const nlohmann::json &j2, const std::string& key, const double w) -> Eigen::ArrayXd{
            auto v1 = j1.at(key).template get<std::valarray<double>>();
            auto v2 = j2.at(key).template get<std::valarray<double>>();
            std::valarray<double> vs = v1*w + v2*(1 - w);
            return Eigen::Map<Eigen::ArrayXd>(&(vs[0]), vs.size());
        };

        if (traces.empty()) {
            throw InvalidArgument("The traces variable is empty");
        }
        else if (traces.size() == 1) {
            // Build the arrays of values to find the self-intersection
            for (auto& el : traces[0]) {
                auto rhoV_ = el.at("rhoV / mol/m^3").get<std::valarray<double>>();
                auto y_ = el.at(ykey).get<double>();
                x.push_back(rhoV_[0] / rhoV_.sum()); // Mole fractions in the vapor phase
                y.push_back(y_);
            }
            auto intersections = get_self_intersections(x, y);
            //auto& trace = traces[0];

            auto process_intersection = [&](auto& trace, auto& i) {
                rhoL1 = avg_values(trace[i.j], trace[i.j + 1], "rhoL / mol/m^3", i.s);
                rhoL2 = avg_values(trace[i.k], trace[i.k + 1], "rhoL / mol/m^3", i.t);
                rhoV = avg_values(trace[i.j], trace[i.j + 1], "rhoV / mol/m^3", i.s);

                if (key == "T"){
                    double T = trace[0].at("T / K"); // All at same temperature
                    
                    // Polish the solution
                    auto [code, rhoVfinal, rhoL1final, rhoL2final] = mix_VLLE_T(model, T, rhoV, rhoL1, rhoL2, 1e-10, 1e-10, 1e-10, 1e-10, opt.max_steps);
                    
                    return nlohmann::json{
                        {"variables", "rhoV, rhoL1, rhoL2"},
                        {"approximate", {rhoV, rhoL1, rhoL2} },
                        {"polished", {rhoVfinal, rhoL1final, rhoL2final} },
                        {"polisher_return_code", static_cast<int>(code)}
                    };
                }
                else if (key == "P"){
                    double p = trace[0].at("pL / Pa"); // all at same pressure
                    
                    // Polish the solution
                    auto [code, Tfinal, rhoVfinal, rhoL1final, rhoL2final] = mix_VLLE_p(model, p, i.y, rhoV, rhoL1, rhoL2, 1e-10, 1e-10, 1e-10, 1e-10, opt.max_steps);
                    
                    return nlohmann::json{
                        {"variables", "rhoV, rhoL1, rhoL2, T"},
                        {"approximate", {rhoV, rhoL1, rhoL2, i.y} },
                        {"polished", {rhoVfinal, rhoL1final, rhoL2final, Tfinal} },
                        {"polisher_return_code", static_cast<int>(code)}
                    };
                }
                else{
                    throw teqp::InvalidArgument("Bad key");
                }
            };
            std::vector<nlohmann::json> solutions;
            
            for (auto& intersection : intersections) {
                try {
                    auto soln = process_intersection(traces[0], intersection);
                    auto rhovecL1 = soln.at("polished")[1].template get<std::valarray<double>>();
                    auto rhovecL2 = soln.at("polished")[2].template get<std::valarray<double>>();
                    auto rhodiff = 100*(rhovecL1.sum() / rhovecL2.sum() - 1);
                    if (std::abs(rhodiff) > opt.rho_trivial_threshold) {
                        // Only keep non-trivial solutions
                        solutions.push_back(soln);
                    }
                }
                catch(...) {
                    // Additional sanity checking goes here...
                    ;
                }
            }
            return solutions;
        }
        else if (traces.size() == 2) {
            
            std::vector<double> x1, y1, x2, y2;
            // Build the arrays of values to find the cross-intersection
            for (auto& el : traces[0]) {
                auto rhoV_ = el.at("rhoV / mol/m^3").get<std::valarray<double>>();
                auto y_ = el.at(ykey).get<double>();
                x1.push_back(rhoV_[0] / rhoV_.sum()); // Mole fractions in the vapor phase
                y1.push_back(y_);
            }
            for (auto& el : traces[1]) {
                auto rhoV_ = el.at("rhoV / mol/m^3").get<std::valarray<double>>();
                auto y_ = el.at(ykey).get<double>();
                x2.push_back(rhoV_[0] / rhoV_.sum()); // Mole fractions in the vapor phase
                y2.push_back(y_);
            }
            auto intersections = get_cross_intersections(x1, y1, x2, y2);

            auto process_intersection = [&](auto& i) {
                rhoL1 = avg_values(traces[0][i.j], traces[0][i.j + 1], "rhoL / mol/m^3", i.s);
                rhoL2 = avg_values(traces[1][i.k], traces[1][i.k + 1], "rhoL / mol/m^3", i.t);
                rhoV = avg_values(traces[0][i.j], traces[0][i.j + 1], "rhoV / mol/m^3", i.s);
                
                if (key == "T"){
                    double T = traces[0][0].at(xkey);
                    
                    // Polish the solution
                    auto [code, rhoVfinal, rhoL1final, rhoL2final] = mix_VLLE_T(model, T, rhoV, rhoL1, rhoL2, 1e-10, 1e-10, 1e-10, 1e-10, opt.max_steps);
                    
                    return nlohmann::json{
                        {"variables", "rhoV, rhoL1, rhoL2"},
                        {"approximate", {rhoV, rhoL1, rhoL2} },
                        {"polished", {rhoVfinal, rhoL1final, rhoL2final} },
                        {"polisher_return_code", static_cast<int>(code)}
                    };
                }
                else if (key == "P"){
                    double p = traces[0][0].at(xkey);
                    
                    // Polish the solution
                    auto [code, Tfinal, rhoVfinal, rhoL1final, rhoL2final] = mix_VLLE_p(model, p, i.y, rhoV, rhoL1, rhoL2, 1e-10, 1e-10, 1e-10, 1e-10, opt.max_steps);
                    
                    return nlohmann::json{
                        {"variables", "rhoV, rhoL1, rhoL2, T"},
                        {"approximate", {rhoV, rhoL1, rhoL2, i.y} },
                        {"polished", {rhoVfinal, rhoL1final, rhoL2final, Tfinal} },
                        {"polisher_return_code", static_cast<int>(code)}
                    };
                }
                else{
                    throw teqp::InvalidArgument("Bad key");
                }
            };
            std::vector<nlohmann::json> solutions;
            
            for (auto& intersection : intersections) {
                try {
                    auto soln = process_intersection(intersection);
                    auto rhovecL1 = soln.at("polished")[1].template get<std::valarray<double>>();
                    auto rhovecL2 = soln.at("polished")[2].template get<std::valarray<double>>();
                    auto rhodiff = 100*(rhovecL1.sum() / rhovecL2.sum() - 1);
                    if (std::abs(rhodiff) > opt.rho_trivial_threshold) {
                        // Only keep non-trivial solutions
                        solutions.push_back(soln);
                    }
                }
                catch(...) {
                    // Additional sanity checking goes here...
                    ;
                }
            }
            return solutions;
        }
        else {
            throw InvalidArgument("No cross intersection between traces implemented yet");
        }
    }

    /**
    * \brief Given an isothermal VLE trace(s) for a binary mixture, obtain the VLLE solution
    * \param model The Model to be used for the thermodynamics
    * \param traces The nlohmann::json formatted information from the traces, perhaps obtained from trace_VLE_isotherm_binary
    */
    inline auto find_VLLE_T_binary(const AbstractModel& model, const std::vector<nlohmann::json>& traces, const std::optional<VLLEFinderOptions>& options = std::nullopt){
        return find_VLLE_gen_binary(model, traces, "T", options);
    }

    /**
    * \brief Given an isobaric VLE trace(s) for a binary mixture, obtain the VLLE solution
    * \param model The Model to be used for the thermodynamics
    * \param traces The nlohmann::json formatted information from the traces, perhaps obtained from trace_VLE_isobar_binary
    */
    inline auto find_VLLE_p_binary(const AbstractModel& model, const std::vector<nlohmann::json>& traces, const std::optional<VLLEFinderOptions>& options = std::nullopt){
        return find_VLLE_gen_binary(model, traces, "P", options);
    }

    inline auto get_drhovecdT_VLLE_binary(const AbstractModel& model, double T, const EArrayd &rhovecV, const EArrayd& rhovecL1, const EArrayd& rhovecL2){
        
        auto dot = [](const EArrayd&a, const EArrayd &b){ return a.cwiseProduct(b).sum(); };
        
        Eigen::MatrixXd LHS(2, 2);
        Eigen::MatrixXd RHS(2, 1);
        Eigen::MatrixXd PSIV = model.build_Psi_Hessian_autodiff(T, rhovecV);
        Eigen::MatrixXd PSIL1 = model.build_Psi_Hessian_autodiff(T, rhovecL1);
        Eigen::MatrixXd PSIL2 = model.build_Psi_Hessian_autodiff(T, rhovecL2);
        double dpdTV = model.get_dpdT_constrhovec(T, rhovecV);
        double dpdTL1 = model.get_dpdT_constrhovec(T, rhovecL1);
        double dpdTL2 = model.get_dpdT_constrhovec(T, rhovecL2);
        
        // here mu is not the entire chemical potential, rather it is just the residual part and
        // the density-dependent part from the ideal-gas
        double RV = model.R(rhovecV/rhovecV.sum());
        double RL1 = model.R(rhovecL1/rhovecL1.sum());
        double RL2 = model.R(rhovecL2/rhovecL2.sum());
        EArrayd dmudTV = model.build_d2PsirdTdrhoi_autodiff(T, rhovecV) + RV*log(rhovecV);
        EArrayd dmudTL1 = model.build_d2PsirdTdrhoi_autodiff(T, rhovecL1) + RL1*log(rhovecL1);
        EArrayd dmudTL2 = model.build_d2PsirdTdrhoi_autodiff(T, rhovecL2) + RL2*log(rhovecL2);
        
        LHS.row(0) = PSIV*(rhovecL1-rhovecV).matrix();
        LHS.row(1) = PSIV*(rhovecL2-rhovecV).matrix();
        RHS(0) = dot(dmudTL1-dmudTV, rhovecL1) - (dpdTL1-dpdTV);
        RHS(1) = dot(dmudTL2-dmudTV, rhovecL2) - (dpdTL2-dpdTV);
        
        Eigen::ArrayXd drhovecVdT = LHS.colPivHouseholderQr().solve(RHS);
        Eigen::VectorXd AV = PSIV*drhovecVdT.matrix();
        Eigen::ArrayXd drhovecL1dT = PSIL1.colPivHouseholderQr().solve((AV.array() - (dmudTL1-dmudTV)).matrix());
        Eigen::ArrayXd drhovecL2dT = PSIL2.colPivHouseholderQr().solve((AV.array() - (dmudTL2-dmudTV)).matrix());
        
        return std::make_tuple(drhovecVdT, drhovecL1dT, drhovecL2dT);
    };

    /**
    \brief Given an initial VLLE solution, trace the VLLE curve. We know the VLLE curve is a function of only one state variable by Gibbs' rule
     */
    inline auto trace_VLLE_binary(const teqp::VLLE::AbstractModel& model, const double Tinit, const EArrayd& rhovecVinit, const EArrayd& rhovecL1init, const EArrayd& rhovecL2init, const std::optional<VLLETracerOptions>& options_ = std::nullopt){
        auto options = options_.value_or(VLLETracerOptions());
        
        // Typedefs for the types for odeint for simple Euler and RK45 integrators
        using state_type = std::vector<double>;
        using namespace boost::numeric::odeint;
        
        typedef runge_kutta_cash_karp54< state_type > error_stepper_type;
        typedef controlled_runge_kutta< error_stepper_type > controlled_stepper_type;
        
        auto xprime = [&](const state_type& x, state_type& dxdt, const double T)
        {
            // Unpack the inputs
            const auto rhovecV_ = Eigen::Map<const Eigen::ArrayXd>(&(x[0]), 2),
                       rhovecL1_ = Eigen::Map<const Eigen::ArrayXd>(&(x[0]) + 2, 2),
                       rhovecL2_ = Eigen::Map<const Eigen::ArrayXd>(&(x[0]) + 4, 2);
            
            auto [drhovecVdT, drhovecL1dT, drhovecL2dT] = VLLE::get_drhovecdT_VLLE_binary(model, T, rhovecV_, rhovecL1_, rhovecL2_);
            Eigen::Map<Eigen::ArrayXd>(&(dxdt[0]), 2) = drhovecVdT;
            Eigen::Map<Eigen::ArrayXd>(&(dxdt[0]) + 2, 2) = drhovecL1dT;
            Eigen::Map<Eigen::ArrayXd>(&(dxdt[0]) + 4, 2) = drhovecL2dT;
        };
        
        // Define the tolerances
        double abs_err = options.abs_err, rel_err = options.rel_err, a_x = 1.0, a_dxdt = 1.0;
        controlled_stepper_type controlled_stepper(default_error_checker< double, range_algebra, default_operations >(abs_err, rel_err, a_x, a_dxdt));
        
        // Copy variables into the stepping array
        double T = Tinit, dT = options.init_dT;
        state_type x0(3*2);
        Eigen::Map<Eigen::ArrayXd>(&(x0[0]), 2) = rhovecVinit;
        Eigen::Map<Eigen::ArrayXd>(&(x0[0]) + 2, 2) = rhovecL1init;
        Eigen::Map<Eigen::ArrayXd>(&(x0[0]) + 4, 2) = rhovecL2init;
        
        nlohmann::json data_collector = nlohmann::json::array();
        for (auto iter = 0; iter < options.max_step_count; ++iter) {
            int retry_count = 0;
            
            auto res = controlled_step_result::fail;
            try {
                res = controlled_stepper.try_step(xprime, x0, T, dT);
            }
            catch (const std::exception &e) {
                if (options.verbosity > 0) {
                    std::cout << e.what() << std::endl;
                }
                break;
            }
            
            if (res != controlled_step_result::success) {
                // Try again, with a smaller step size
                iter--;
                retry_count++;
                continue;
            }
            else {
                retry_count = 0;
            }
            // Reduce step size if greater than the specified max step size
            dT = std::min(dT, options.max_dT);
            
            const auto rhovecV = Eigen::Map<const Eigen::ArrayXd>(&(x0[0]), 2),
                       rhovecL1 = Eigen::Map<const Eigen::ArrayXd>(&(x0[0]) + 2, 2),
                       rhovecL2 = Eigen::Map<const Eigen::ArrayXd>(&(x0[0]) + 4, 2);
            
            // Polish if requested
            if (options.polish){
                auto [code, rhovecVnew, rhovecL1new, rhovecL2new] = teqp::VLLE::mix_VLLE_T(model, T, rhovecV, rhovecL1, rhovecL2, 1e-10, 1e-10, 1e-10, 1e-10, options.max_polish_steps);
                Eigen::Map<Eigen::ArrayXd>(&(x0[0]), 2) = rhovecVnew;
                Eigen::Map<Eigen::ArrayXd>(&(x0[0]) + 2, 2) = rhovecL1new;
                Eigen::Map<Eigen::ArrayXd>(&(x0[0]) + 4, 2) = rhovecL2new;
            }
            
            auto critV = model.get_criticality_conditions(T, rhovecV);
            auto critL1 = model.get_criticality_conditions(T, rhovecL1);
            auto critL2 = model.get_criticality_conditions(T, rhovecL2);
            double rhomolarV = rhovecV.sum();
            auto molefracV = rhovecV/rhomolarV;
            auto pV = rhomolarV*model.get_R(molefracV)*T*(1.0 + model.get_Ar01(T, rhomolarV, molefracV));
            if (!std::isfinite(pV)){
                if (options.verbosity > 0) {
                    std::cout << "Calculated pressure is not finite" << std::endl;
                }
                break;
            }
            if(options.init_dT > 0 && T > options.T_limit){
                if (options.verbosity > 0) {
                    std::cout << "Exceeded maximum temperature of " << options.T_limit << " K" << std::endl;
                }
                break;
            }
            if(options.init_dT < 0 && T < options.T_limit){
                if (options.verbosity > 0) {
                    std::cout << "Exceeded minimum temperature of " << options.T_limit << " K" << std::endl;
                }
                break;
            }
            
            if (options.verbosity > 100){
                std::cout << "[T,x0L1,x0L2,x0V]: " << T << "," << rhovecL1[0]/rhovecL1.sum() << "," << rhovecL2[0]/rhovecL2.sum() << "," << rhovecV[0]/rhovecV.sum() << std::endl;
                std::cout << "[crits]: " << critV << "," << critL1 << "," << critL2 << std::endl;
            }
            
            if (options.terminate_composition){
                auto c0 = (Eigen::ArrayXd(3) << rhovecL1[0]/rhovecL1.sum(), rhovecL2[0]/rhovecL2.sum(), rhovecV[0]/rhovecV.sum()).finished();
                auto diffs = (Eigen::ArrayXd(3) << c0[0]-c0[1], c0[0]-c0[2], c0[1]-c0[2]).finished();
                if ((diffs.cwiseAbs() < options.terminate_composition_tol).any()){
                    break;
                }
            }
            if (retry_count > options.max_step_retries){
                if (options.verbosity > 0) {
                    std::cout << "Max retries of step exceeded." << std::endl;
                }
                break;
            }
            
            nlohmann::json entry{
                {"T / K", T},
                {"rhoL1 / mol/m^3", rhovecL1},
                {"rhoL2 / mol/m^3", rhovecL2},
                {"rhoV / mol/m^3", rhovecV},
                {"critV", critV},
                {"critL1", critL1},
                {"critL2", critL2},
                {"pV / Pa", pV}
            };
            data_collector.push_back(entry);
        }
        return data_collector;
    }

}
}
