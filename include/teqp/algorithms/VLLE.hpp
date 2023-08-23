#pragma once

#include "teqp/derivs.hpp"
#include "teqp/exceptions.hpp"
#include "teqp/algorithms/VLLE_types.hpp"
#include "teqp/cpp/teqpcpp.hpp"

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
    
    auto mix_VLLE_T(const AbstractModel& model, double T, const EArrayd& rhovecVinit, const EArrayd& rhovecL1init, const EArrayd& rhovecL2init, double atol, double reltol, double axtol, double relxtol, int maxiter) {

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

    /**
    Derived from https://stackoverflow.com/a/17931809
    */
    template<typename Iterable>
    auto get_self_intersections(Iterable& x, Iterable& y) {
        Eigen::Array22d A;
        std::vector<SelfIntersectionSolution> solns;
        for (auto j = 0; j < x.size() - 1; ++j) {
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

    /**
    * \brief Given an isothermal VLE trace for a binary mixture, obtain the VLLE solution
    * \param model The Model to be used for the thermodynamics
    * \param traces The nlohmann::json formatted information from the traces, perhaps obtained from trace_VLE_isotherm_binary
    */
    
    auto find_VLLE_T_binary(const AbstractModel& model, const std::vector<nlohmann::json>& traces, const std::optional<VLLEFinderOptions> options = std::nullopt) {
        std::vector<double> x, y;
        auto opt = options.value_or(VLLEFinderOptions{});

        Eigen::ArrayXd rhoL1(2), rhoL2(2), rhoV(2);

        if (traces.empty()) {
            throw InvalidArgument("The traces variable is empty");
        }
        else if (traces.size() == 1) {
            // Build the arrays of values to find the self-intersection
            for (auto& el : traces[0]) {
                auto rhoV = el.at("rhoV / mol/m^3").get<std::valarray<double>>();
                auto p = el.at("pL / Pa").get<double>();
                x.push_back(rhoV[0] / rhoV.sum()); // Mole fractions in the vapor phase
                y.push_back(p);
            }
            auto intersections = get_self_intersections(x, y);
            //auto& trace = traces[0];

            auto process_intersection = [&](auto& trace, auto& i) {
                auto rhoL1_j = traces[0][i.j].at("rhoL / mol/m^3").template get<std::valarray<double>>();
                auto rhoL1_jp1 = traces[0][i.j + 1].at("rhoL / mol/m^3").template get<std::valarray<double>>();
                std::valarray<double> rhoL1_ = rhoL1_j * i.s + rhoL1_jp1 * (1 - i.s);
                Eigen::Map<Eigen::ArrayXd>(&(rhoL1[0]), rhoL1.size()) = Eigen::Map<Eigen::ArrayXd>(&(rhoL1_[0]), rhoL1_.size());

                auto rhoL2_k = traces[0][i.k].at("rhoL / mol/m^3").template get<std::valarray<double>>();
                auto rhoL2_kp1 = traces[0][i.k + 1].at("rhoL / mol/m^3").template get<std::valarray<double>>();
                std::valarray<double> rhoL2_ = rhoL2_k * i.t + rhoL2_kp1 * (1 - i.t);
                Eigen::Map<Eigen::ArrayXd>(&(rhoL2[0]), rhoL2.size()) = Eigen::Map<Eigen::ArrayXd>(&(rhoL2_[0]), rhoL2_.size());

                auto rhoV_j = traces[0][i.j].at("rhoV / mol/m^3").template get<std::valarray<double>>();
                auto rhoV_jp1 = traces[0][i.j + 1].at("rhoV / mol/m^3").template get<std::valarray<double>>();
                std::valarray<double> rhoV_ = rhoV_j * i.s + rhoV_jp1 * (1 - i.s);
                Eigen::Map<Eigen::ArrayXd>(&(rhoV[0]), rhoV.size()) = Eigen::Map<Eigen::ArrayXd>(&(rhoV_[0]), rhoV_.size());

                double T = traces[0][0].at("T / K");

                // Polish the solution
                auto [code, rhoVfinal, rhoL1final, rhoL2final] = mix_VLLE_T(model, T, rhoV, rhoL1, rhoL2, 1e-10, 1e-10, 1e-10, 1e-10, opt.max_steps);

                //double xL1 = rhoL1[0] / rhoL1.sum(), xL2 = rhoL2[0] / rhoL2.sum(), xV = rhoV[0] / rhoV.sum();
                //double xL1f = rhoL1final[0] / rhoL1final.sum(),
                //       xL2f = rhoL2final[0] / rhoL2final.sum(),
                //       xVf = rhoVfinal[0] / rhoVfinal.sum();

                return nlohmann::json{
                    {"approximate", {rhoV, rhoL1, rhoL2} },
                    {"polished", {rhoVfinal, rhoL1final, rhoL2final} },
                    {"polisher_return_code", static_cast<int>(code)}
                };
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
        else {
            throw InvalidArgument("No cross intersection between traces implemented yet");
        }
    }
}
}
