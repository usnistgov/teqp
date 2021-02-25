#include <Eigen/Dense>
#include <tuple>
#include <valarray>
#include <iostream>

#include "teqp/core.hpp"

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
auto sorted_eigen(const Eigen::MatrixXd& H) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(H);
    return std::make_tuple(es.eigenvalues(), es.eigenvectors());
}

struct EigenData {
    Eigen::ArrayXd v0, v1, eigenvalues;
    Eigen::MatrixXd eigenvectorscols;
};

template<typename Model, typename TType, typename RhoType>
auto eigen_problem(const Model& model, const TType T, const RhoType& rhovec) {

    EigenData ed;

    auto N = rhovec.size();
    auto mask = (rhovec != 0);

    // Build the Hessian for the residual part;
    auto H = build_Psir_Hessian(model, T, rhovec);
    // ... and add ideal-gas terms to H
    for (auto i = 0; i < N; ++i) {
        if (mask[i]) {
            H(i, i) += model.R*T/rhovec[i];
        }
    }

    int nonzero_count = 0;
    for (auto& m : mask) {
        nonzero_count += (m == 1);
    }
    auto zero_count = N - nonzero_count;

    if (zero_count == 0) {
        // Not an infinitely dilute mixture, nothing special
        std::tie(ed.eigenvalues, ed.eigenvectorscols) = sorted_eigen(H);
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
        std::cout << H << std::endl;
        std::cout << Hprime << std::endl;
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
    ed.v0 = ed.eigenvectorscols.col(0);
    ed.v1 = ed.eigenvectorscols.col(1);
    return ed;
}

struct psi1derivs{
    std::valarray<double> psir, psi0, tot;
    EigenData ei;
};

template<typename Model, typename TType, typename RhoType>
auto get_derivs(const Model& model, const TType T, const RhoType& rhovec) {
    auto R = model.R;

    // Solve the complete eigenvalue problem
    auto ei = eigen_problem(model, T, rhovec);

    // Ideal-gas contributions of psi0 w.r.t. sigma_1, in the same form as the residual part
    std::valarray<double> psi0_derivs(0.0, 5);
    psi0_derivs[0] = -1; // Placeholder, not needed
    psi0_derivs[1] = -1; // Placeholder, not needed
    for (auto i = 0; i < rhovec.size(); ++i) {
        if (rhovec[i] != 0){
            psi0_derivs[2] += R*T*pow(ei.v0[i],2)/rhovec[i];  // second derivative
            psi0_derivs[3] += -R*T*pow(ei.v0[i],3)/pow(rhovec[i],2); // third derivative
            psi0_derivs[4] += 2*R*T*pow(ei.v0[i],4)/pow(rhovec[i],3); // fourth derivative
        }
    }

    // Calculate the first through fourth derivative of Psi^r w.r.t. sigma_1
    std::valarray<MultiComplex<double>> v0(ei.v0.size()); for(auto i = 0; i < ei.v0.size(); ++i){ v0[i] = ei.v0[i]; }
    std::valarray<MultiComplex<double>> rhovecmcx(rhovec.size());  for (auto i = 0; i < rhovec.size(); ++i) { rhovecmcx[i] = rhovec[i]; }
    using fcn_t = std::function<MultiComplex<double>(const MultiComplex<double>&)>;
    fcn_t wrapper = [&rhovecmcx, &v0, &T, &model](const MultiComplex<double>& sigma_1) {
        auto rhovecused = rhovecmcx + sigma_1*v0;
        return get_Psir(model, T, rhovecused);
    };
    auto psir_derivs_ = diff_mcx1(wrapper, 0.0, 4, true);
    auto psir_derivs = std::valarray<double>(&psir_derivs_[0], psir_derivs_.size());

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
bool all(const Iterable &foo) {
    return std::all_of(std::begin(foo), std::end(foo), [](const auto x){ return x;});
}
template <typename Iterable>
bool any(const Iterable& foo) {
    return std::any_of(std::begin(foo), std::end(foo), [](const auto x) { return x;});
}

template<typename Model, typename TType, typename RhoType>
auto get_drhovec_dT_crit(const Model& model, const TType T, const RhoType& rhovec) {

    // The derivatives of total Psi w.r.t.sigma_1(mcx for residual, analytic for ideal)
    // Returns a tuple, with residual, ideal, total dicts with of number of derivatives, value of derivative
    auto all_derivs = get_derivs(model, T, rhovec);
    auto derivs = all_derivs.tot;

    // The temperature derivative of total Psi w.r.t.T from a centered finite difference in T
    auto dT = 1e-7;
    auto plusT = get_derivs(model, T + dT, rhovec).tot;
    auto minusT = get_derivs(model, T - dT, rhovec).tot;
    auto derivT = (plusT - minusT)/(2.0*dT);

    // Solve the eigenvalue problem for the given T & rho
    auto ei = all_derivs.ei;

    auto sigma2 = 2e-5*rhovec.sum(); // This is the perturbation along the second eigenvector

    auto v1 = std::valarray<double>(&ei.v1[0], ei.v1.size());
    auto rhovec_plus = rhovec + v1*sigma2;
    auto rhovec_minus = rhovec - v1*sigma2;
    std::string stepping_desc = "";
    auto deriv_sigma2 = 0.0*all_derivs.tot;
    if (all(rhovec_minus > 0) && all(rhovec_plus > 0)){
        // Conventional centered derivative
        auto plus_sigma2 = get_derivs(model, T, rhovec_plus);
        auto minus_sigma2 = get_derivs(model, T, rhovec_minus);
        deriv_sigma2 = (plus_sigma2.tot - minus_sigma2.tot)/(2.0 * sigma2);
        stepping_desc = "conventional centered";
    }
    else if (any(rhovec_minus < 0)) {
        // Forward derivative in the direction of v1
        auto plus_sigma2 = get_derivs(model, T, rhovec_plus);
        auto rhovec_2plus = rhovec + 2*v1*sigma2;
        auto plus2_sigma2 = get_derivs(model, T, rhovec_2plus);
        deriv_sigma2 = (-3*derivs + 4*plus_sigma2.tot - plus2_sigma2.tot)/(2.0 * sigma2);
        stepping_desc = "forward";
    }
    else if (any(rhovec_minus > 0)) {
        // Negative derivative in the direction of v1
        auto minus_sigma2 = get_derivs(model, T, rhovec_minus);
        auto rhovec_2minus = rhovec - 2*v1*sigma2;
        auto minus2_sigma2 = get_derivs(model, T, rhovec_2minus);
        deriv_sigma2 = (-3*derivs + 4*minus_sigma2.tot - minus2_sigma2.tot) / (-2.0 * sigma2);
        stepping_desc = "backwards";
    }
    else {
        throw std::invalid_argument("This is not possible I think.");
    }

    // The columns of b are from Eq. 31 and Eq. 33
    Eigen::MatrixXd b(2,2);
    b << derivs[3], derivs[4],             // row is d^3\Psi/d\sigma_1^3, d^4\Psi/d\sigma_1^4
         deriv_sigma2[2], deriv_sigma2[3]; // row is d/d\sigma_2(d^3\Psi/d\sigma_1^3), d/d\sigma_2(d^3\Psi/d\sigma_1^3)

    auto LHS = (ei.eigenvectorscols*b).transpose();
    Eigen::MatrixXd RHS(2,1); RHS << -derivT[2], -derivT[3];
    Eigen::MatrixXd drhovec_dT = LHS.colPivHouseholderQr().solve(RHS);

    //            if debug:
    //        print('Conventional Psi^r derivs w.r.t. sigma_1:', all_derivs.psir)
    //            print('Conventional Psi derivs w.r.t. sigma_1:', all_derivs.tot)
    //            print('Derivs w.r.t. T', derivT)
    //            print('sigma_2', sigma2)
    //            print('Finite Psi derivs w.r.t. sigma_2:', deriv_sigma2)
    //            print('stepping', stepping)
    //            print("U (rows as eigenvalues)", ei.U_T.T)
    //            print("LHS", LHS)
    //            print("RHS", RHS)
    //            print("drhovec_dT", drhovec_dT)

    return std::valarray<double>(&drhovec_dT(0), drhovec_dT.size());
}

void trace() { 
    // Argon + Xenon
    std::valarray<double> Tc_K = { 150.687, 289.733 };
    std::valarray<double> pc_Pa = { 4863000.0, 5842000.0 };
    vdWEOS<double> vdW(Tc_K, pc_Pa);
    auto Zc = 3.0/8.0;
    auto rhoc0 = pc_Pa[0]/(vdW.R*Tc_K[0]) / Zc; 
    auto T = Tc_K[0];
    const auto dT = 1;
    std::valarray<double> rhovec = { rhoc0, 0.0 };
    for (auto iter = 0; iter < 1000; ++iter){
        auto drhovecdT = get_drhovec_dT_crit(vdW, T, rhovec);
        rhovec += drhovecdT*dT;
        T += dT;
        int rr = 0;
        auto z0 = rhovec[0] / rhovec.sum();
        std::cout << z0 <<  " ," << rhovec[0] << "," << T << std::endl;
        if (z0 < 0) {
            break;
        }
    }
    int rr = 0;
    
}
int main() {   
    trace();
    return EXIT_SUCCESS;
}