#pragma once
#include "teqp/constants.hpp"

namespace teqp {
namespace association{

enum class association_classes {not_set, a1A, a2B, a3B, a4C, not_associating};

inline auto get_association_classes(const std::string& s) {
    if (s == "1A") { return association_classes::a1A; }
    else if (s == "2B") { return association_classes::a2B; }
    else if (s == "2B") { return association_classes::a2B; }
    else if (s == "3B") { return association_classes::a3B; }
    else if (s == "4C") { return association_classes::a4C; }
    else {
        throw std::invalid_argument("bad association flag: " + s);
    }
}

enum class radial_dists { CS, KG };

inline auto get_radial_dist(const std::string& s) {
    if (s == "CS") { return radial_dists::CS; }
    else if (s == "KG") { return radial_dists::KG; }
    else {
        throw std::invalid_argument("bad radial_dist flag: " + s);
    }
}

enum class Delta_rules {not_set, CR1, Dufal};

inline auto get_Delta_rule(const std::string& s) {
    if (s == "CR1") { return Delta_rules::CR1; }
    else if (s == "Dufal") { return Delta_rules::Dufal; }
    else {
        throw std::invalid_argument("bad Delta_rule flag: " + s);
    }
}

struct CanonicalData{
    Eigen::ArrayXd b_m3mol, ///< The covolume b, in m^3/mol, one per component
        beta, ///< The volume factor, dimensionless, one per component
        epsilon_Jmol; ///< The association energy of each molecule, in J/mol, one per component
    radial_dists radial_dist;
};

struct DufalData{
    // Parameters coming from the non-associating part, one per component
    Eigen::ArrayXd sigma_m, epsilon_Jmol, lambda_r;
    Eigen::ArrayXXd kmat; ///< Matrix of k_{ij} values
    
    // Parameters from the associating part, one per component
    Eigen::ArrayXd epsilon_HB_Jmol, K_HB_m3;
    
    Eigen::ArrayXd SIGMA3ij_m3, EPSILONOVERKij_K, LAMBDA_Rij, EPSILONOVERK_HBij_K, KHBij_m3;
    
    void apply_mixing_rules(){
        std::size_t N = sigma_m.size();
        SIGMA3ij_m3.resize(N,N);
        EPSILONOVERKij_K.resize(N,N);
        LAMBDA_Rij.resize(N,N);
        EPSILONOVERK_HBij_K.resize(N,N);
        KHBij_m3.resize(N,N);
        
        for (auto i = 0U; i < N; ++i){
            for (auto j = 0U; j < N; ++j){
                SIGMA3ij_m3(i,j) = POW3((sigma_m[i] + sigma_m[j])/2.0);
                EPSILONOVERKij_K(i, j) = (1-kmat(i,j))*sqrt(POW3(sigma_m[i]*sigma_m[j]))/SIGMA3ij_m3(i,j)*sqrt(epsilon_Jmol[i]*epsilon_Jmol[j])/constants::R_CODATA2017;
                LAMBDA_Rij(i, j) = 3 + sqrt((lambda_r[i]-3)*(lambda_r[j]-3));
                EPSILONOVERK_HBij_K(i, j) = sqrt(epsilon_HB_Jmol[i]*epsilon_HB_Jmol[j])/constants::R_CODATA2017;
                KHBij_m3(i,j) = POW3((cbrt(K_HB_m3[i]) + cbrt(K_HB_m3[j]))/2.0); // Published erratum in Dufal: https://doi.org/10.1080/00268976.2017.1402604
            }
        }
    }
};


}
}
