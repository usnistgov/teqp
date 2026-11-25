#pragma once

#include <fstream>
#include <iomanip>
#include <optional>
#include <sstream>

#include "nlohmann/json.hpp"

#include <Eigen/Dense>
#include "teqp/algorithms/rootfinding.hpp"
#include "teqp/algorithms/critical_pure.hpp"
#include "teqp/algorithms/critical_tracing_types.hpp"
#include "teqp/exceptions.hpp"

// Imports from boost
#include <boost/numeric/odeint/stepper/controlled_runge_kutta.hpp>
#include <boost/numeric/odeint/stepper/runge_kutta_cash_karp54.hpp>
#include <boost/numeric/odeint/stepper/euler.hpp>

namespace teqp {

template<typename Model, typename Scalar = double, typename VecType = Eigen::ArrayXd>
struct CriticalTracing {
    /***
    * \brief Simple wrapper to sort the eigenvalues(and associated eigenvectors) in increasing order
    * \param H The matrix, in this case, the Hessian matrix of Psi w.r.t.the molar concentrations
    * \param values The eigenvalues
    * \returns vectors The eigenvectors, as columns
    *
    * See https://stackoverflow.com/a/56327853
    *
    * \note The input Matrix is symmetric, thus the SelfAdjointEigenSolver can be used, and returned eigenvalues
    * will be real and sorted already with corresponding eigenvectors as columns
    */
    static auto sorted_eigen(const Eigen::MatrixXd& H) {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(H);
        return std::make_tuple(es.eigenvalues(), es.eigenvectors());
    }

    /**
    * \brief Eigenvalues and eigenvectors of the symmetric 2x2 matrix \f$[[a, b], [b, d]]\f$ from one Jacobi rotation
    * \returns The eigenvalues in increasing order and the eigenvectors as columns
    *
    * The tangent of the rotation angle is \f$t = {\rm sgn}(z)/(|z| + \sqrt{1 + z^2})\f$ with \f$z = (d - a)/(2b)\f$
    * and \f${\rm sgn}(0) = 1\f$; for \f$b = 0\f$ there is no rotation.
    * The small component of an eigenvector is then a product of t and a number of order one, and it has the relative
    * accuracy of the matrix entries. This matters for the Hessian of \f$\Psi\f$ at a small concentration of one component:
    * its diagonal entry \f$RT/\rho_i\f$ is large, the component of \f$\vec v_0\f$ along that component is of the order
    * of \f$\rho_i\f$, and the derivatives of the criticality conditions contain it divided by \f$\rho_i\f$.
    */
    static auto symmetric_eigen_2x2(const double a, const double b, const double d) {
        double t = 0;
        if (b != 0) {
            const double z = (d - a) / (2 * b);
            t = ((z < 0) ? -1.0 : 1.0) / (std::abs(z) + std::hypot(1.0, z));
        }
        const double c = 1 / std::hypot(1.0, t), s = t * c;
        Eigen::Vector2d values(a - t * b, d + t * b);
        Eigen::Matrix2d vectors;
        vectors << c, s,
                  -s, c;
        if (values[0] > values[1]) {
            std::swap(values[0], values[1]);
            vectors.col(0).swap(vectors.col(1));
        }
        return std::make_pair(values, vectors);
    }

    static auto eigen_problem(const AbstractModel& model, const Scalar T, const VecType& rhovec, const std::optional<VecType>& alignment_v0 = std::nullopt) {

        EigenData ed;

        auto N = rhovec.size();
        Eigen::ArrayX<bool> mask = (rhovec != 0).eval();

        // Build the Hessian for the residual part;
#if defined(USE_AUTODIFF)
        auto H = model.build_Psir_Hessian_autodiff(T, rhovec);
#else
        using id = IsochoricDerivatives<decltype(model)>;
        auto H = id::build_Psir_Hessian_mcx(model, T, rhovec);
#endif
        // ... and add ideal-gas terms to H
        for (auto i = 0; i < N; ++i) {
            if (mask[i]) {
                H(i, i) += model.R(rhovec/rhovec.sum()) * T / rhovec[i];
            }
        }

        Eigen::Index nonzero_count = mask.count();
        auto zero_count = N - nonzero_count;

        if (zero_count == 0) {
            // Not an infinitely dilute mixture, nothing special
            std::tie(ed.eigenvalues, ed.eigenvectorscols) = sorted_eigen(H);

            // Align with the eigenvector of the component with the smallest density, and make that one positive
            Eigen::Index ind;
            rhovec.minCoeff(&ind);
            if (ed.eigenvectorscols.col(ind).minCoeff() < 0) {
                ed.eigenvectorscols *= -1.0;
            }

            if (N == 2) {
                // The closed form for two components, each eigenvector with the sign of the aligned one of the
                // iterative solver. The closed form reads the entry of H that the solver reads. Where the two
                // eigenvalues are equal to rounding, the matrix does not determine its eigenvectors, and the
                // closed form and the solver can return different pairs.
                auto [values, vectors] = symmetric_eigen_2x2(H(0, 0), H(1, 0), H(1, 1));
                for (auto k = 0; k < 2; ++k) {
                    if (vectors.col(k).dot(ed.eigenvectorscols.col(k)) < 0) {
                        vectors.col(k) *= -1.0;
                    }
                }
                ed.eigenvalues = values;
                ed.eigenvectorscols = vectors;
            }
        }
        else if (zero_count == 1) {
            // Extract Hessian matrix without entries where rho is exactly zero
            std::vector<int> indicesToKeep;

            int badindex = -1;
            for (auto i = 0; i < N; ++i) {
                if (mask[i]) {
                    indicesToKeep.push_back(i);
                }
                else {
                    badindex = i;
                }
            }
            Eigen::MatrixXd Hprime = H(indicesToKeep, indicesToKeep);

            auto [eigenvalues, eigenvectors] = sorted_eigen(Hprime);

            // Inject values into the U^T and v0 vectors
            //
            // Make a padded matrix for U (with eigenvectors as rows)
            Eigen::MatrixXd U = H; U.setZero();

            // Fill in the associated elements corresponding to eigenvectors 
            for (auto i = 0; i < N - nonzero_count; ++i) {
                U(i, indicesToKeep) = eigenvectors.col(i); // Put in the row, leaving a hole for the zero value
            }

            // The last row has a 1 in the column corresponding to the pure fluid entry
            // We insist that there must be only one non-zero entry
            U.row(U.rows() - 1)(badindex) = 1.0;

            ed.eigenvalues = eigenvalues;
            ed.eigenvectorscols = U.transpose();
        }
        else {
            throw std::invalid_argument("More than one non-zero concentration value found; not currently supported");
        }
        if (alignment_v0 && alignment_v0.value().size() > 0 && ed.eigenvectorscols.col(0).matrix().dot(alignment_v0.value().matrix()) < 0) {
            ed.eigenvectorscols.col(0) *= -1;
        }
        
        ed.v0 = ed.eigenvectorscols.col(0);
        ed.v1 = ed.eigenvectorscols.col(1);
        return ed;
    }

    struct psi1derivs {
        Eigen::ArrayXd psir, psi0, tot;
        EigenData ei;
    };

    static auto get_minimum_eigenvalue_Psi_Hessian(const AbstractModel& model, const Scalar T, const VecType& rhovec) {
        return eigen_problem(model, T, rhovec).eigenvalues[0];
    }

    static auto get_derivs(const AbstractModel& model, const double T, const VecType& rhovec, const std::optional<VecType>& alignment_v0 = std::nullopt) {
        auto molefrac = rhovec / rhovec.sum();
        auto R = model.R(molefrac);

        // Solve the complete eigenvalue problem
        auto ei = eigen_problem(model, T, rhovec, alignment_v0);

        // Ideal-gas contributions of psi0 w.r.t. sigma_1, in the same form as the residual part
        Eigen::ArrayXd psi0_derivs(5); psi0_derivs.setZero();
        psi0_derivs[0] = -1; // Placeholder, not needed
        psi0_derivs[1] = -1; // Placeholder, not needed
        for (auto i = 0; i < rhovec.size(); ++i) {
            if (rhovec[i] != 0) {
                psi0_derivs[2] += R * T * pow(ei.v0[i], 2) / rhovec[i];  // second derivative
                psi0_derivs[3] += -R * T * pow(ei.v0[i], 3) / pow(rhovec[i], 2); // third derivative
                psi0_derivs[4] += 2 * R * T * pow(ei.v0[i], 4) / pow(rhovec[i], 3); // fourth derivative
            }
        }

#if defined(USE_AUTODIFF)
        auto psir_derivs = model.get_Psir_sigma_derivs(T, rhovec, ei.v0);
//        // Calculate the first through fourth derivative of Psi^r w.r.t. sigma_1
//        ArrayXdual4th v0(ei.v0.size()); for (auto i = 0; i < ei.v0.size(); ++i) { v0[i] = ei.v0[i]; }
//        ArrayXdual4th rhovecad(rhovec.size());  for (auto i = 0; i < rhovec.size(); ++i) { rhovecad[i] = rhovec[i]; }
//        dual4th varsigma{ 0.0 };
//        auto wrapper = [&rhovecad, &v0, &T, &model](const auto& sigma_1) {
//            auto rhovecused = (rhovecad + sigma_1 * v0).eval();
//            auto rhotot = rhovecused.sum();
//            auto molefrac = (rhovecused / rhotot).eval();
//            return eval(model.alphar(T, rhotot, molefrac) * model.R(molefrac) * T * rhotot);
//        };
//        auto psir_derivs_ = derivatives(wrapper, wrt(varsigma), at(varsigma));
//        Eigen::ArrayXd psir_derivs; psir_derivs.resize(5);
//        for (auto i = 0; i < 5; ++i) { psir_derivs[i] = psir_derivs_[i]; }

#else
        using namespace mcx;
        // Calculate the first through fourth derivative of Psi^r w.r.t. sigma_1
        Eigen::Vector<MultiComplex<double>, Eigen::Dynamic> v0(ei.v0.size()); for (auto i = 0; i < ei.v0.size(); ++i) { v0[i] = ei.v0[i]; }
        Eigen::Vector<MultiComplex<double>, Eigen::Dynamic> rhovecmcx(rhovec.size());  for (auto i = 0; i < rhovec.size(); ++i) { rhovecmcx[i] = rhovec[i]; }
        using fcn_t = std::function<MultiComplex<double>(const MultiComplex<double>&)>;
        fcn_t wrapper = [&rhovecmcx, &v0, &T, &model](const MultiComplex<double>& sigma_1) {
            Eigen::Vector<MultiComplex<double>, Eigen::Dynamic> rhovecused = rhovecmcx + sigma_1 * v0;
            auto rhotot = rhovecused.sum();
            auto molefrac = (rhovecused / rhotot).eval();
            return model.alphar(T, rhotot, molefrac) * model.R(molefrac) * T * rhotot;
        };
        auto psir_derivs_ = diff_mcx1(wrapper, 0.0, 4, true);
        Eigen::ArrayXd psir_derivs; psir_derivs.resize(5);
        for (auto i = 0; i < 5; ++i) { psir_derivs[i] = psir_derivs_[i]; }
#endif

        // As a sanity check, the minimum eigenvalue of the Hessian constructed based on the molar concentrations
        // must match the second derivative of psi_tot w.r.t. sigma_1. This is not always satisfied for derivatives
        // with Cauchy method
        //if (abs(np.min(ei.eigvals) - psitot_derivs[2]) > 1e-3){
        //    print(np.min(ei.eigvals), psitot_derivs[2], rhovec)
        //}

        psi1derivs psi1;
        psi1.psir = psir_derivs;
        psi1.psi0 = psi0_derivs;
        psi1.tot = psi0_derivs + psir_derivs;
        psi1.ei = ei;
        return psi1;
    }

    template <typename Iterable>
    static bool all(const Iterable& foo) {
        return std::all_of(std::begin(foo), std::end(foo), [](const auto x) { return x; });
    }
    template <typename Iterable>
    static bool any(const Iterable& foo) {
        return std::any_of(std::begin(foo), std::end(foo), [](const auto x) { return x; });
    }

private:
    /// Mixed derivatives of a function f along two directions e0, e1 in concentration space and along temperature
    struct MixedDerivs {
        double f00, f01, f000, f001, f011, f0000, f0001, f00T, f000T, f01T;
        MixedDerivs operator+(const MixedDerivs& o) const {
            return { f00 + o.f00, f01 + o.f01, f000 + o.f000, f001 + o.f001, f011 + o.f011, f0000 + o.f0000, f0001 + o.f0001, f00T + o.f00T, f000T + o.f000T, f01T + o.f01T };
        }
    };

    /**
     * \brief The mixed derivatives of \f$\Psi^{\rm r}\f$ other than \f$f_{011}\f$ and \f$f_{01T}\f$, each taken directly
     *
     * One derivative of order 4 along \f$\vec e_0\f$ (get_Psir_sigma_derivs) gives \f$f_{00}\f$, \f$f_{000}\f$ and \f$f_{0000}\f$. Nested dual
     * numbers (get_Psir_mixed_dir_derivs) with the levels along \f$(\vec e_0, \vec e_0, \vec e_0, T)\f$ give \f$f_{000T}\f$ and
     * \f$f_{00T}\f$, and along \f$(\vec e_0, \vec e_0, \vec e_0, \vec e_1)\f$ \f$f_{0001}\f$, \f$f_{001}\f$ and \f$f_{01}\f$.
     * \f$f_{011}\f$ and \f$f_{01T}\f$ are returned as zero; mixed_derivs_f011_f01T gives them.
     */
    static MixedDerivs mixed_derivs(const AbstractModel& model, const double T, const VecType& rhovec, const Eigen::ArrayXd& e0, const Eigen::ArrayXd& e1) {
        MixedDerivs m{};
        const Eigen::ArrayXd s = model.get_Psir_sigma_derivs(T, rhovec, e0);
        m.f00 = s[2]; m.f000 = s[3]; m.f0000 = s[4];
        // Entry 15 is the derivative along all four levels, entry 11 along levels 1, 2 and 4, entry 9 along levels 1 and 4
        Eigen::ArrayXXd V(4, 2);
        Eigen::ArrayXd Tdir = Eigen::ArrayXd::Zero(4);
        V.row(0) = e0.transpose(); V.row(1) = e0.transpose(); V.row(2) = e0.transpose(); V.row(3).setZero(); Tdir[3] = 1.0;
        const Eigen::ArrayXd A = model.get_Psir_mixed_dir_derivs(T, rhovec, Tdir, V);
        m.f000T = A[15]; m.f00T = A[11];
        V.row(3) = e1.transpose(); Tdir[3] = 0.0;
        const Eigen::ArrayXd B = model.get_Psir_mixed_dir_derivs(T, rhovec, Tdir, V);
        m.f0001 = B[15]; m.f001 = B[11]; m.f01 = B[9];
        return m;
    }

    /**
     * \brief The mixed derivatives \f$f_{011}\f$ and \f$f_{01T}\f$ of \f$\Psi^{\rm r}\f$, from nested dual numbers with the levels along
     * \f$(\vec e_0, \vec e_1, \vec e_1)\f$ and \f$(\vec e_0, \vec e_1, T)\f$
     */
    static std::pair<double, double> mixed_derivs_f011_f01T(const AbstractModel& model, const double T, const VecType& rhovec, const Eigen::ArrayXd& e0, const Eigen::ArrayXd& e1) {
        Eigen::ArrayXXd V(3, 2);
        Eigen::ArrayXd Tdir = Eigen::ArrayXd::Zero(3);
        V.row(0) = e0.transpose(); V.row(1) = e1.transpose(); V.row(2) = e1.transpose();
        const double f011 = model.get_Psir_mixed_dir_derivs(T, rhovec, Tdir, V)[7];
        V.row(2).setZero(); Tdir[2] = 1.0;
        const double f01T = model.get_Psir_mixed_dir_derivs(T, rhovec, Tdir, V)[7];
        return { f011, f01T };
    }

    /**
     * \brief The mixed derivatives of the ideal-gas part \f$\Psi^0 = RT\sum_i\rho_i(\ln\rho_i - 1)\f$, with R the value at the state
     *
     * The derivative tensors of \f$\Psi^0\f$ are diagonal: \f$\Psi^0_{ii} = RT/\rho_i\f$, \f$\Psi^0_{iii} = -RT/\rho_i^2\f$, \f$\Psi^0_{iiii} = 2RT/\rho_i^3\f$,
     * \f$\Psi^0_{iiT} = R/\rho_i\f$, \f$\Psi^0_{iiiT} = -R/\rho_i^2\f$. They are contracted with the directions term by term.
     * They are not taken with dual numbers along the eigenvectors: near infinite dilution the terms of order
     * \f$1/\rho_i^3\f$ of the minor component would be subtracted from each other.
     *
     * Each term is a product of the ratios \f$e_{0,i}/\rho_i\f$ and \f$e_{1,i}/\rho_i\f$ and of components of the directions,
     * so no power of \f$1/\rho_i\f$ is formed alone. At a small concentration of component q the component
     * \f$v_{0,q}\f$ of the eigenvector is of the order of \f$\rho_q\f$ and \f$v_{0,q}/\rho_q\f$ stays of order one.
     */
    static MixedDerivs ideal_mixed_derivs(const double R, const double T, const VecType& rhovec, const Eigen::ArrayXd& e0, const Eigen::ArrayXd& e1) {
        const Eigen::ArrayXd u0 = e0 / rhovec, u1 = e1 / rhovec;
        MixedDerivs m;
        m.f00 = R * T * (u0 * e0).sum();
        m.f01 = R * T * (u0 * e1).sum();
        m.f000 = -R * T * (u0 * u0 * e0).sum();
        m.f001 = -R * T * (u0 * u0 * e1).sum();
        m.f011 = -R * T * (u0 * u1 * e1).sum();
        m.f0000 = 2 * R * T * (u0 * u0 * u0 * e0).sum();
        m.f0001 = 2 * R * T * (u0 * u0 * u0 * e1).sum();
        m.f00T = R * (u0 * e0).sum();
        m.f000T = -R * (u0 * u0 * e0).sum();
        m.f01T = R * (u0 * e1).sum();
        return m;
    }

    /// The temperature and the molar concentrations of a state, for an error message
    static std::string describe_state(const double T, const VecType& rhovec) {
        std::stringstream ss;
        ss << std::setprecision(17) << "T = " << T << " K, rhovec = [";
        for (auto i = 0; i < rhovec.size(); ++i) {
            ss << ((i > 0) ? ", " : "") << rhovec[i];
        }
        ss << "] mol/m^3";
        return ss.str();
    }

public:
    /**
     * \brief The criticality conditions \f$C = [\lambda_0, \partial^3\Psi/\partial\sigma_1^3]\f$ of a binary mixture and their exact Jacobian
     *
     * The second condition is a cubic form in the eigenvector
     * \f$\vec v_0\f$, which turns with the state; its derivatives carry the rotation terms with
     * \f$1/(\lambda_0 - \lambda_1)\f$ from first-order perturbation of the eigenvector.
     * The ideal-gas part of \f$\Psi\f$ is built with the number \f$R(\vec x)\f$ at the state, so where
     * R depends on composition the conditions also depend on the concentrations through R;
     * the Jacobian carries that term, \f$(\partial C/\partial R)\,\nabla_{\vec\rho}R\f$.
     *
     * \param alignment_v0 If given, \f$\vec v_0\f$ is oriented to have a positive projection on it. The sign of \f$\vec v_0\f$ sets the sign of the second condition.
     * \throws std::invalid_argument if the mixture is not binary, if a molar concentration is negative, or if both
     * molar concentrations are zero (from eigen_problem). The result is not finite, and nothing is thrown, where the
     * temperature or a molar concentration is not finite, where the model returns a derivative that is not finite, and
     * where the two eigenvalues of the Hessian are equal (the rotation terms divide by their difference).
     */
    static CriticalJacobianData get_criticality_conditions_Jacobian(const AbstractModel& model, const Scalar T, const VecType& rhovec, const std::optional<VecType>& alignment_v0 = std::nullopt) {
        if (rhovec.size() != 2) {
            throw std::invalid_argument("get_criticality_conditions_Jacobian is implemented for binary mixtures");
        }
        if ((rhovec < 0).any()) {
            throw std::invalid_argument("get_criticality_conditions_Jacobian needs molar concentrations that are not negative; " + describe_state(T, rhovec));
        }
        CriticalJacobianData out;
        out.ei = eigen_problem(model, T, rhovec, alignment_v0);
        const Eigen::Index nzero = (rhovec == 0).count();
        if (nzero == 0) {
            const Eigen::ArrayXd v0 = out.ei.v0, v1 = out.ei.v1;
            const double R = model.R(rhovec / rhovec.sum());
            const auto mi = ideal_mixed_derivs(R, T, rhovec, v0, v1);
            auto mr = mixed_derivs(model, T, rhovec, v0, v1);
            std::tie(mr.f011, mr.f01T) = mixed_derivs_f011_f01T(model, T, rhovec, v0, v1);
            const auto m = mr + mi;
            const double gap = out.ei.eigenvalues[0] - out.ei.eigenvalues[1];

            // f011 and the gap grow as 1/rho_i at a small concentration of component i; their ratio does not
            out.C << out.ei.eigenvalues[0], m.f000;
            const Eigen::Vector2d dC_dsigma1(m.f000, m.f0000 + 3 * m.f001 * (m.f001 / gap));
            const Eigen::Vector2d dC_dsigma2(m.f001, m.f0001 + 3 * m.f001 * (m.f011 / gap));
            const Eigen::Vector2d dC_dT(m.f00T, m.f000T + 3 * m.f001 * (m.f01T / gap));
            out.J.col(0) = dC_dT;
            out.J.block(0, 1, 2, 2) = dC_dsigma1 * v0.matrix().transpose() + dC_dsigma2 * v1.matrix().transpose();

            // Dependence through R: d(Hessian)/dR = diag(T/rho_i), d(Psi_000)/dR = -T sum_i v0_i^3/rho_i^2 at fixed v0.
            // These are T/R times the ideal-gas parts of the temperature derivatives f00T, f01T and f000T.
            const Eigen::ArrayXd gradR = model.build_R_gradient_autodiff(rhovec);
            const Eigen::Vector2d dC_dR(T / R * mi.f00T, T / R * (mi.f000T + 3 * m.f001 * (mi.f01T / gap)));
            out.J.block(0, 1, 2, 2) += dC_dR * gradR.matrix().transpose();
        }
        else {
            // Component q is absent (eigen_problem has thrown if both concentrations are zero). As rho_q -> 0, lambda_1 ~ RT/rho_q diverges, v0 -> e_p, and
            // lambda_0 = H_pp - rho_q H_pq^2/(RT) + O(rho_q^2), v0 = e_p - (rho_q H_pq/(RT)) e_q + O(rho_q^2).
            // The conditions and their derivatives then have the finite limits below, in which the
            // derivatives of Psi^r are plain component derivatives.
            out.at_infinite_dilution = true;
            const Eigen::Index q = (rhovec[0] == 0) ? 0 : 1, p = 1 - q;
            Eigen::ArrayXd ep = Eigen::ArrayXd::Zero(2), eq = Eigen::ArrayXd::Zero(2);
            ep[p] = 1.0; eq[q] = 1.0;
            const double R = model.R(rhovec / rhovec.sum()), RT = R * T, rhop = rhovec[p];
            const auto m = mixed_derivs(model, T, rhovec, ep, eq); // derivatives of Psi^r
            const double psir_pp = m.f00, psir_pq = m.f01;

            Eigen::Array2d C(RT / rhop + psir_pp, -RT / (rhop * rhop) + m.f000);
            Eigen::Vector2d dC_dT(R / rhop + m.f00T, -R / (rhop * rhop) + m.f000T);
            Eigen::Vector2d dC_drhop(C[1], 2 * RT / (rhop * rhop * rhop) + m.f0000);
            Eigen::Vector2d dC_drhoq(m.f001 - psir_pq * psir_pq / RT,
                                     m.f0001 - 3 * m.f001 * psir_pq / RT + psir_pq * psir_pq * psir_pq / (RT * RT));
            // Dependence through R, in the same limit: dC/dR = [T/rho_p, -T/rho_p^2]
            const Eigen::ArrayXd gradR = model.build_R_gradient_autodiff(rhovec);
            const Eigen::Vector2d dC_dR(T / rhop, -T / (rhop * rhop));
            dC_drhop += dC_dR * gradR[p];
            dC_drhoq += dC_dR * gradR[q];
            // The second condition is odd in v0; eigen_problem may have returned v0 = -e_p
            const double sign = (out.ei.v0[p] < 0) ? -1.0 : 1.0;
            C[1] *= sign; dC_dT[1] *= sign; dC_drhop[1] *= sign; dC_drhoq[1] *= sign;
            out.C = C;
            out.J.col(0) = dC_dT;
            out.J.col(1 + p) = dC_drhop;
            out.J.col(1 + q) = dC_drhoq;
        }
        return out;
    }

    /**
     * \brief The derivative of the molar concentrations with respect to temperature along the critical curve of a binary mixture
     *
     * The tangent of the critical curve is the null vector \f$\vec t\f$ of the exact Jacobian of the
     * criticality conditions (get_criticality_conditions_Jacobian); this function returns \f$(t_1, t_2)/t_0\f$.
     *
     * \throws std::invalid_argument if the mixture is not binary, if a molar concentration is negative, if both molar
     * concentrations are zero, or if the result is not finite. The result is not finite where the temperature or a
     * molar concentration is not finite, where the model returns a derivative that is not finite, where the two
     * eigenvalues of the Hessian are equal, and where the temperature has an extremum along the curve (\f$t_0 = 0\f$).
     */
    static auto get_drhovec_dT_crit(const AbstractModel& model, const Scalar& T, const VecType& rhovec) {
        const auto cj = get_criticality_conditions_Jacobian(model, T, rhovec);
        const Eigen::Vector3d r0 = cj.J.row(0).transpose(), r1 = cj.J.row(1).transpose();
        const Eigen::Vector3d t = r0.cross(r1);
        Eigen::MatrixXd drhovec_dT(2, 1);
        drhovec_dT << t[1] / t[0], t[2] / t[0];
        if (!drhovec_dT.allFinite()) {
            throw std::invalid_argument("The tangent of the critical curve is not finite; " + describe_state(T, rhovec));
        }
        return drhovec_dT;
    }

    static auto get_criticality_conditions(const AbstractModel& model, const Scalar T, const VecType& rhovec) {
        auto derivs = get_derivs(model, T, rhovec, Eigen::ArrayXd());
        return (Eigen::ArrayXd(2) << derivs.tot[2], derivs.tot[3]).finished();
    }

    /**
    * \brief The EigenVectorPath function of Algorithm 2 from Deiters and Bell: https://dx.doi.org/10.1021/acs.iecr.0c03667
    * 
    * There is a typo in the paper: the last step should be rho + 2*drho*dir
    */
    static auto EigenVectorPath(const AbstractModel& model, const Scalar T, const VecType& rhovec, const VecType &u1, Scalar drho) {
        VecType dir = u1/6.0;
        VecType rhoshift = rhovec + drho*u1;
        auto e1 = eigen_problem(model, T, rhoshift, u1);
        VecType u1new = e1.v0;

        dir += u1/3.0;
        rhoshift = rhovec + drho*u1new;
        auto e2 = eigen_problem(model, T, rhoshift, u1);
        u1new = e2.v0;

        dir += u1/3.0;
        rhoshift = rhovec + 2*drho*u1new;
        auto e3 = eigen_problem(model, T, rhoshift, u1);
        u1new = e3.v0;

        dir += u1new/6.0;
        rhoshift = rhovec + 2*drho*dir;

        return rhoshift;
    }

    /**
    * \brief The LocalStability function of Algorithm 2 from Deiters and Bell: https://dx.doi.org/10.1021/acs.iecr.0c03667
    */
    static auto is_locally_stable(const AbstractModel& model, const Scalar T, const VecType& rhovec, const Scalar stability_rel_drho) {
        // At the critical point
        auto ezero = eigen_problem(model, T, rhovec);
        Scalar lambda1 = ezero.eigenvalues(0);

        Scalar drho = stability_rel_drho * rhovec.sum();
        // Towards positive density shift
        VecType rhovecplus = EigenVectorPath(model, T, rhovec, ezero.v0, drho);
        auto eplus = eigen_problem(model, T, rhovecplus, ezero.v0);
        // Towards negative density shift
        VecType rhovecminus = EigenVectorPath(model, T, rhovec, ezero.v0, -drho);
        auto eminus = eigen_problem(model, T, rhovecminus, ezero.v0);

        bool unstable = (eplus.eigenvalues(0) < lambda1 || eminus.eigenvalues(0) < lambda1);
        return !unstable;
    }

    /// Polish a critical point while keeping the overall composition constant and iterating for temperature and overall density
    static auto critical_polish_fixedmolefrac(const AbstractModel& model, const Scalar T, const VecType& rhovec, const Scalar z0) {
        auto polish_x_resid = [&model, &z0](const auto& x) {
            auto T = x[0];
            Eigen::ArrayXd rhovec(2); rhovec << z0*x[1], (1-z0)*x[1];
            //auto z0new = rhovec[0] / rhovec.sum();
            auto derivs = get_derivs(model, T, rhovec, {});
            // First two are residuals on critical point
            return (Eigen::ArrayXd(2) << derivs.tot[2], derivs.tot[3]).finished();
        };
        Eigen::ArrayXd x0(2); x0 << T, rhovec[0]+rhovec[1];
        auto r0 = polish_x_resid(x0);
        auto x = NewtonRaphson(polish_x_resid, x0, 1e-10);
        auto r = polish_x_resid(x);
        Eigen::ArrayXd change = x0 - x;
        if (!std::isfinite(x[0]) || !std::isfinite(x[1])) {
            throw std::invalid_argument("Something not finite; aborting polishing");
        }
        Eigen::ArrayXd rhovecsoln(2); rhovecsoln << x[1]*z0, x[1] * (1 - z0);
        return std::make_tuple(x[0], rhovecsoln);
    }
    static auto critical_polish_fixedrho(const AbstractModel& model, const Scalar T, const VecType& rhovec, const int i) {
        Scalar rhoval = rhovec[i];
        auto polish_x_resid = [&model, &i, &rhoval](const auto& x) {
            auto T = x[0];
            Eigen::ArrayXd rhovec(2); rhovec << x[1], x[2];
            //auto z0new = rhovec[0] / rhovec.sum();
            auto derivs = get_derivs(model, T, rhovec);
            // First two are residuals on critical point, third is residual on the molar concentration to be held constant
            return (Eigen::ArrayXd(3) << derivs.tot[2], derivs.tot[3], rhovec[i] - rhoval).finished();
        };
        Eigen::ArrayXd x0(3); x0 << T, rhovec[0], rhovec[1];
        auto r0 = polish_x_resid(x0);
        auto x = NewtonRaphson(polish_x_resid, x0, 1e-10);
        auto r = polish_x_resid(x);
        Eigen::ArrayXd change = x0 - x;
        if (!std::isfinite(T) || !std::isfinite(x[1]) || !std::isfinite(x[2])) {
            throw std::invalid_argument("Something not finite; aborting polishing");
        }
        Eigen::ArrayXd rho = x.tail(x.size() - 1);
        return std::make_tuple(x[0], rho);
    }
    static auto critical_polish_fixedT(const AbstractModel& model, const Scalar T, const VecType& rhovec) {
        auto polish_T_resid = [&model, &T](const auto& x) {
            auto derivs = get_derivs(model, T, x);
            return (Eigen::ArrayXd(2) << derivs.tot[2], derivs.tot[3]).finished();
        };
        Eigen::ArrayXd x0 = rhovec;
        auto r0 = polish_T_resid(x0);
        auto x = NewtonRaphson(polish_T_resid, x0, 1e-10);
        auto r = polish_T_resid(x);
        Eigen::ArrayXd change = x0 - x;
        if (!std::isfinite(T) || !std::isfinite(x[1])) {
            throw std::invalid_argument("Something not finite; aborting polishing");
        }
        return x;
    }

    static auto trace_critical_arclength_binary(const AbstractModel& model, const Scalar& T0, const VecType& rhovec0, const std::optional<std::string>& filename_ = std::nullopt, const std::optional<TCABOptions> &options_ = std::nullopt) -> nlohmann::json {
        std::string filename = filename_.value_or("");
        TCABOptions options = options_.value_or(TCABOptions{});

        VecType last_drhodt;

        // Typedefs for the types for odeint for simple Euler and RK45 integrators
        using state_type = std::vector<double>; 
        using namespace boost::numeric::odeint;
        euler<state_type> eul; // Class for simple Euler integration
        typedef runge_kutta_cash_karp54< state_type > error_stepper_type;
        typedef controlled_runge_kutta< error_stepper_type > controlled_stepper_type;

        auto dot = [](const auto& v1, const auto& v2) { return (v1 * v2).sum(); };
        auto norm = [](const auto& v) { return sqrt((v * v).sum()); };

        auto JSONdata = nlohmann::json::array();
        std::ofstream ofs = (filename.empty()) ? std::ofstream() : std::ofstream(filename);
        
        double c = options.init_c; 

        // The function for the derivative in the form of odeint
        // x is [T, rhovec]
        auto xprime = [&](const state_type& x, state_type& dxdt, const double /* t */)
        {
            // Unpack the inputs
            const double T = x[0];
            const auto rhovec = Eigen::Map<const Eigen::ArrayXd>(&(x[0]) + 1, x.size() - 1);
            if (options.terminate_negative_density && rhovec.minCoeff() < 0) {
                throw std::invalid_argument("Density is negative");
            }
            
            auto drhodT = get_drhovec_dT_crit(model, T, rhovec).array().eval();
            auto dTdt = 1.0 / norm(drhodT);
            Eigen::ArrayXd drhodt = c * (drhodT * dTdt).eval();

            dxdt[0] = c*dTdt;

            auto drhodtview = Eigen::Map<Eigen::ArrayXd>(&(dxdt[0]) + 1, dxdt.size() - 1); 
            drhodtview = drhodt; // Copy values into the view

            if (last_drhodt.size() > 0) {
                auto rhodot = dot(drhodt, last_drhodt);
                if (rhodot < 0) {
                    Eigen::Map<Eigen::ArrayXd>(&(dxdt[0]), dxdt.size()) *= -1;
                }
            }
        };
        auto get_dxdt = [&](const state_type& x) {
            auto dxdt = state_type(x.size());
            xprime(x, dxdt, -1);
            return dxdt;
        };
        // Pull out the drhodt from dxdt, just a convenience function
        auto extract_drhodt = [](const state_type& dxdt) -> Eigen::ArrayXd {
            return Eigen::Map<const Eigen::ArrayXd>(&(dxdt[0]) + 1, dxdt.size() - 1);
        };
        // Pull out the dTdt from dxdt, just a convenience function
        auto extract_dTdt = [](const state_type& dxdt) -> double {
            return dxdt[0];
        };

        // Define the tolerances
        double abs_err = options.abs_err, rel_err = options.rel_err, a_x = 1.0, a_dxdt = 1.0;
        controlled_stepper_type controlled_stepper(default_error_checker< double, range_algebra, default_operations >(abs_err, rel_err, a_x, a_dxdt));

        double t = 0, dt = options.init_dt;

        // Build the initial state array, with T followed by rhovec
        std::vector<double> x0(rhovec0.size() + 1); 
        x0[0] = T0;
        Eigen::Map<Eigen::ArrayXd>(&(x0[0]) + 1, x0.size() - 1) = rhovec0;

        // Make variables T and rhovec references to the contents of x0 vector
        // The views are mutable (danger!)
        double& T = x0[0];
        auto rhovec = Eigen::Map<Eigen::ArrayXd>(&(x0[0]) + 1, x0.size() - 1);

        auto store_drhodt = [&](const state_type& x0) {
            last_drhodt = extract_drhodt(get_dxdt(x0));
        };

        auto store_point = [&]() {

            // Calculate some other parameters, for debugging, or scientific interest
            auto rhotot = rhovec.sum();
            double p = rhotot * model.R(rhovec / rhovec.sum()) * T + model.get_pr(T, rhovec);
            auto conditions = get_criticality_conditions(model, T, rhovec);
            double splus = model.get_splus(T, rhovec);
            auto dxdt = x0;
            xprime(x0, dxdt, -1.0);

            // Store the data in a JSON structure
            nlohmann::json point = {
                {"t", t},
                {"T / K", T},
                {"rho0 / mol/m^3", static_cast<double>(rhovec[0])},
                {"rho1 / mol/m^3", static_cast<double>(rhovec[1])},
                {"c", c},
                {"s^+", splus},
                {"p / Pa", p},
                {"dT/dt", dxdt[0]},
                {"drho0/dt", dxdt[1]},
                {"drho1/dt", dxdt[2]},
                {"lambda1", conditions[0]},
                {"dirderiv(lambda1)/dalpha", conditions[1]},
            };
            if (options.calc_stability) {
                point["locally stable"] = is_locally_stable(model, T, rhovec, options.stability_rel_drho);
            }
            JSONdata.push_back(point);
        };

        // Line writer
        auto write_line = [&]() {
            std::stringstream out;
            auto rhotot = rhovec.sum();
            double z0 = rhovec[0] / rhotot;
            auto conditions = get_criticality_conditions(model, T, rhovec);
            out << z0 << "," << rhovec[0] << "," << rhovec[1] << "," << T << "," << rhotot * model.R(rhovec / rhovec.sum()) * T + model.get_pr(T, rhovec) << "," << c << "," << dt << "," << conditions(0) << "," << conditions(1) << std::endl;
            std::string sout(out.str());
            std::cout << sout;
            if (ofs.is_open()) {
                ofs << sout;
            }
        };
        
        int counter_T_converged = 0, retry_count = 0;
        ofs << "z0 / mole frac.,rho0 / mol/m^3,rho1 / mol/m^3,T / K,p / Pa,c,dt,condition(1),condition(2)" << std::endl;
        
        // Determine the initial direction of integration
        {
            const auto step = (rhovec + extract_drhodt(get_dxdt(x0))*dt).eval();
            Eigen::ArrayX<bool> negativestepvals = (step < 0).eval();
            // Flip the sign if the first step would yield any negative concentrations
            if (negativestepvals.any()) {
                c *= -1;
            }
        }
        //store_drhodt(x0);
        if (!filename.empty()) {
            write_line();
        }

        for (auto iter = 0; iter < options.max_step_count; ++iter) {
            
            // Calculate the derivatives at the beginning of the step
            auto dxdt_start_step = get_dxdt(x0);
            auto x_start_step = x0;

            if (iter == 0 && retry_count == 0) { 
                store_point(); }
            
            if (options.integration_order == 5) {
                auto res = controlled_step_result::fail;
                try {
                    res = controlled_stepper.try_step(xprime, x0, t, dt);
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
                    /*auto dxdt = x0;
                    xprime(x0, dxdt, -1.0);
                    auto drhodt = Eigen::Map<Eigen::ArrayXd>(&(dxdt[0]) + 1, dxdt.size() - 1);
                    auto rhodotproduct = dot(drhodt, last_drhodt);
                    int rr = 0;*/
                }
                // Reduce step size if greater than the specified max step size
                dt = std::min(dt, options.max_dt);
            }
            else if (options.integration_order == 1) {
                try {
                    eul.do_step(xprime, x0, t, dt);
                    t += dt;
                }
                catch (const std::exception &e) {
                    std::cout << e.what() << std::endl;
                    break;
                }
            }
            else {
                throw std::invalid_argument("integration order is invalid:" + std::to_string(options.integration_order));
            }

            auto z0 = rhovec[0] / rhovec.sum();
            if (z0 < 0 || z0 > 1) {
                if (options.verbosity > 10) {
                    std::cout << "Termination because z0 of " + std::to_string(z0) + " is outside [0, 1]" << std::endl;
                }
                break;
            }

            // The polishers to be tried, in order, to polish the critical point
            using PolisherType = std::function<std::tuple<double, VecType>(const AbstractModel&, const Scalar, const VecType&)>;
            std::vector<PolisherType> polishers = {
                [&](const AbstractModel& model, const Scalar T, const VecType& rhovec) {
                    auto [Tnew, rhovecnew] = critical_polish_fixedmolefrac(model, T, rhovec, z0);
                    return std::make_tuple(Tnew, rhovecnew);
                },
                [&](const AbstractModel& model, const Scalar T, const VecType& rhovec) {
                    auto rhovecnew = critical_polish_fixedT(model, T, rhovec);
                    return std::make_tuple(T, rhovecnew);
                },
                [&](const AbstractModel& model, const Scalar T, const VecType& rhovec) {
                    int i = 0;
                    auto [Tnew, rhovecnew] = critical_polish_fixedrho(model, T, rhovec, i);
                    return std::make_tuple(Tnew, rhovecnew);
                },
                [&](const AbstractModel& model, const Scalar T, const VecType& rhovec) {
                    int i = 1;
                    auto [Tnew, rhovecnew] = critical_polish_fixedrho(model, T, rhovec, i);
                    return std::make_tuple(Tnew, rhovecnew);
                },
            };
            
            if (options.polish) {
                bool polish_ok = false;
                for (auto &polisher : polishers){
                    try {
                        auto [Tnew, rhovecnew] = polisher(model, T, rhovec);
                        if (std::abs(T - Tnew) > options.polish_reltol_T*T) {
                            throw IterationFailure("Polishing changed the temperature more than " + std::to_string(options.polish_reltol_T*100) + " %");
                        }
                        if (((rhovec-rhovecnew).cwiseAbs() > options.polish_reltol_rho*rhovec).any()){
                            throw IterationFailure("Polishing changed a molar concentration by more than "+std::to_string(options.polish_reltol_rho*100)+" %");
                        }
                        polish_ok = true;
                        T = Tnew;
                        rhovec = rhovecnew;
                        break;
                    }
                    catch (std::exception& e) {
                        if (options.verbosity > 10) {
                            std::cout << "Tracing problem: " << e.what() << std::endl;
                            std::cout << "Starting:: T:" << T << "; rhovec: " << rhovec << std::endl;
                            auto conditions = get_criticality_conditions(model, T, rhovec);
                            std::cout << "Starting crit. cond.: " << conditions << std::endl;
                            try {
                                auto [Tnew, rhovecnew] = polisher(model, T, rhovec);
                                std::cout << "Ending:: T:" << Tnew << "; rhovec: " << rhovecnew << std::endl;
                            } catch (...) {}
                        }
                    }
                }
                if (!polish_ok){
                    if (options.polish_exception_on_fail){
                        throw IterationFailure("Polishing was not successful");
                    }
                    else{
                        if (options.verbosity > 10){
                            std::cout << "Termination because polishing failed" << std::endl;
                        }
                        break;
                    }
                }
            }

            // Store the derivative vector from the beginning of the step, before
            // the actual step is taken.  We don't want to use the values at the end
            // because otherwise simple Euler will never consider the possible change in direction
            //
            // Also, we only do this after two completed steps because sometimes the infinite
            // dilution derivatives seem to be not quite right. There is still a risk that the first
            // step will try to turn around...
            if (iter >= options.skip_dircheck_count) {
                store_drhodt(x_start_step); }

            auto actualstep = (Eigen::Map<Eigen::ArrayXd>(&(x0[0]), x0.size()) - Eigen::Map<Eigen::ArrayXd>(&(x_start_step[0]), x_start_step.size())).eval();
            
            // Check if T has stopped changing
            if (std::abs(actualstep[0]) < options.T_tol) {
                counter_T_converged++;
            }
            else {
                counter_T_converged = 0;
            }

            auto rhotot = rhovec.sum();
            z0 = rhovec[0] / rhotot;
            if (z0 < 0 || z0 > 1) {
                if (options.verbosity > 10) {
                    std::cout << "Termination because z0 of " + std::to_string(z0) + " is outside [0, 1]" << std::endl;
                }
                break;
            }

            if (!filename.empty()) { write_line(); }
            store_point();

            if (counter_T_converged > options.small_T_count) {
                if (options.verbosity > 10){
                    std::cout << "Termination because maximum number of small steps were taken" << std::endl;
                }
                break;
            }
        }
        // If the last step crosses a zero concentration, see if it corresponds to a pure fluid
        // and if so, iterate to find the pure fluid endpoint
        if (options.pure_endpoint_polish) {
            // Simple Euler step t
            auto dxdt = get_dxdt(x0);
            auto drhodt = extract_drhodt(dxdt);
            const auto step = (rhovec + drhodt*dt).eval();
            if ((step * rhovec > 0).any()) {
                // Calculate step sizes to take concentrations to zero
                auto step_sizes = ((-rhovec) / drhodt).eval();
                Eigen::Index ipure;
                rhovec.maxCoeff(&ipure);
                // Find new step size based on the pure we are heading towards: the step that takes
                // the OTHER concentration, the one that vanishes there, to zero
                auto new_step_size = step_sizes(1 - ipure);
                // Take an Euler step of the new size
                const auto new_rhovec = (rhovec + drhodt*new_step_size).eval();
                const double new_T = T + extract_dTdt(dxdt)*new_step_size;
                
                // Solve for pure fluid critical point
                // 
                // A bit trickier here because we need to hack the pure fluid model to put the pure fluid 
                // composition in an index other than the first if ipure != 0 because we don't want 
                // to require that a pure fluid model is also provided since zero mole fractions
                // should be fine
                nlohmann::json flags = { {"alternative_pure_index", ipure}, {"alternative_length", 2} };
                auto [TT, rhorho] = solve_pure_critical(model, new_T, new_rhovec.sum(), flags);

                // Replace the values in T and rhovec
                T = TT;
                rhovec[ipure] = rhorho;
                rhovec[1 - ipure] = 0;

                // And store the polished values
                if (!filename.empty()) { write_line(); }
                store_point();
            }
        }
        //auto N = JSONdata.size();
        return JSONdata;
    }

    /**
    * \brief Calculate dp/dT along the critical locus at given T, rhovec
    * \f[
    *     \rmderivsub{Y}{T}{\CRL} = \deriv{Y}{T}{\vec\rho}  + \sum_j \deriv{Y}{\rho_j}{T,\rho_{k\neq j}} \rmderivsub{\rho_j}{T}{\CRL}
    * \f]
    * where the derivatives on the RHS without the subscript CRL are homogeneous derivatives taken at the mixture statepoint, and are NOT along the critical curve.
    */
    static auto get_dp_dT_crit(const AbstractModel& model, const Scalar& T, const VecType& rhovec) {
        auto dpdrhovec = model.get_dpdrhovec_constT(T, rhovec);
        Scalar v = model.get_dpdT_constrhovec(T, rhovec) + (dpdrhovec.array()*get_drhovec_dT_crit(model, T, rhovec).array()).sum();
        return v;
    }
    
#define CRIT_FUNCTIONS_TO_WRAP \
    X(get_dp_dT_crit) \
    X(trace_critical_arclength_binary) \
    X(critical_polish_fixedmolefrac)  \
    X(get_drhovec_dT_crit) \
    X(get_criticality_conditions_Jacobian) \
    X(get_derivs) \
    X(eigen_problem)

#define X(f) template <typename TemplatedModel, typename ...Params, \
typename = typename std::enable_if<not std::is_base_of<teqp::cppinterface::AbstractModel, TemplatedModel>::value>::type> \
static auto f(const TemplatedModel& model, Params&&... params){ \
    auto view = teqp::cppinterface::adapter::make_cview(model); \
    const AbstractModel& am = *view.get(); \
    return f(am, std::forward<Params>(params)...); \
}
    CRIT_FUNCTIONS_TO_WRAP
#undef X
#undef CRIT_FUNCTIONS_TO_WRAP
    
//template <typename TemplatedModel, typename ...Params,
//typename = typename std::enable_if<not std::is_base_of<teqp::cppinterface::AbstractModel, TemplatedModel>::value>::type>
//static auto critical_polish_fixedmolefrac(const TemplatedModel& model, Params&&... params){
//    auto view = teqp::cppinterface::adapter::make_cview(model);
//    const AbstractModel& am = *view.get();
//    return critical_polish_fixedmolefrac(am, std::forward<Params>(params)...);
//}

}; // namespace Critical

}; // namespace teqp
