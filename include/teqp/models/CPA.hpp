#pragma once

#include <optional>
#include <variant>

#include "nlohmann/json.hpp"
#include <Eigen/Dense>
#include "teqp/types.hpp"
#include "teqp/exceptions.hpp"
#include "teqp/models/association/association.hpp"
#include "teqp/models/association/association_types.hpp"

namespace teqp {

namespace CPA {

template<typename X> auto POW2(X x) { return x * x; };
template<typename X> auto POW3(X x) { return x * POW2(x); };

using radial_dist = association::radial_dists;
using association::association_classes;
using association::get_radial_dist;
using association::get_association_classes;

/// Function that calculates the association binding strength between site A of molecule i and site B on molecule j
template<typename BType, typename TType, typename RhoType, typename VecType>
inline auto get_DeltaAB_pure(radial_dist dist, double epsABi, double betaABi, BType b_cubic, TType RT, RhoType rhomolar, const VecType& /*molefrac*/) {

    using eta_type = std::common_type_t<decltype(rhomolar), decltype(b_cubic)>;
    eta_type eta;
    eta_type g_vm_ref;

    // Calculate the contact value of the radial distribution function g(v)
    switch (dist) {
        case radial_dist::CS: {
            // Carnahan - Starling EOS, given by Kontogeorgis et al., Ind.Eng.Chem.Res. 2006, 45, 4855 - 4868, Eq. 4a and 4b:
            eta = (rhomolar / 4.0) * b_cubic;
            g_vm_ref = (2.0 - eta) / (2.0 * POW3(1.0 - eta));
            break;
        }
        case radial_dist::KG: {
            // Function proposed by  Kontogeorgis, G.M.; Yakoumis, I.V.; Meijer, H.; Hendriks, E.M.; Moorwood, T., Fluid Phase Equilib. 1999, 158 - 160, 201.
            eta = (rhomolar / 4.0) * b_cubic;
            g_vm_ref = 1.0 / (1.0 - 1.9 * eta);
            break;
        }
//        case radial_dist::OT: { // This is identical to KG
//            g_vm_ref = 1.0 / (1.0 - 0.475 * rhomolar * b_cubic);
//            break;
//        }
        default: {
            throw std::invalid_argument("Bad radial_dist");
        }
    }

    // Calculate the association strength between site Ai and Bi for a pure compent
    auto DeltaAiBj = forceeval(g_vm_ref*(exp(epsABi/RT) - 1.0)*b_cubic* betaABi);

    return DeltaAiBj;
};

/// Routine that calculates the fractions of sites Ai not bound to other active sites for pure fluids
/// Some association schemes are explicitly solvable for self-associating compounds, see Huang and Radosz, Ind. Eng. Chem. Res., 29 (11), 1990
/// So far implemented association schemes : 1A, 2B, 3B, 4C (see Kontogeorgis et al., Ind. Eng. Chem. Res. 2006, 45, 4855 - 4868)
/// 

template<typename BType, typename TType, typename RhoType, typename VecType>
inline auto XA_calc_pure(int N_sites, association_classes scheme, radial_dist dist, double epsABi, double betaABi, const BType b_cubic, const TType RT, const RhoType rhomolar, const VecType& molefrac) {

    // Matrix XA(A, j) that contains all of the fractions of sites A not bonded to other active sites for each molecule i
    // Start values for the iteration(set all sites to non - bonded, = 1)
    using result_type = std::common_type_t<decltype(RT), decltype(rhomolar), decltype(molefrac[0])>;
    Eigen::Array<result_type, Eigen::Dynamic, Eigen::Dynamic> XA;  // A maximum of 4 association sites(A, B, C, D)
    XA.resize(N_sites, 1);
    XA.setOnes();

    // Get the association strength between the associating sites
    auto DeltaAiBj = get_DeltaAB_pure(dist, epsABi, betaABi, b_cubic, RT, rhomolar, molefrac);

    if (scheme == association_classes::a1A) { // Acids
        // Only one association site "A"  (OH - group with C = O - group)
        XA(0, 0) = forceeval((-1.0 + sqrt(1.0 + 4.0 * rhomolar * DeltaAiBj)) / (2.0 * rhomolar * DeltaAiBj));
    }
    else if (scheme == association_classes::a2B) { // Alcohols
        // Two association sites "A" and "B"
        XA(0, 0) = forceeval((-1.0 + sqrt(1.0 + 4.0 * rhomolar * DeltaAiBj)) / (2.0 * rhomolar * DeltaAiBj));
        XA(1, 0) = XA(0, 0);   // XB = XA;
    }
    else if (scheme == association_classes::a3B) { // Glycols
        // Three association sites "A", "B", "C"
        XA(0, 0) = forceeval((-(1.0 - rhomolar * DeltaAiBj) + sqrt(POW2(1.0 + rhomolar * DeltaAiBj) + 4.0 * rhomolar * DeltaAiBj)) / (4.0 * rhomolar * DeltaAiBj));
        XA(1, 0) = XA(0, 0);           // XB = XA
        XA(2, 0) = 2.0*XA(0, 0) - 1.0; // XC = 2XA - 1
    }
    else if (scheme == association_classes::a4C) { // Water
        // Four association sites "A", "B", "C", "D"
        XA(0, 0) = forceeval((-1.0 + sqrt(1.0 + 8.0 * rhomolar * DeltaAiBj)) / (4.0 * rhomolar * DeltaAiBj));
        XA(1, 0) = XA(0, 0);   // XB = XA
        XA(2, 0) = XA(0, 0);   // XC = XA
        XA(3, 0) = XA(0, 0);   // XD = XA
    }
    else if (scheme == association_classes::not_associating) { // non - associating compound
        XA(0, 0) = 1;
        XA(1, 0) = 1;
        XA(2, 0) = 1;
        XA(3, 0) = 1;
    }
    else {
        throw std::invalid_argument("Bad scheme");
    }
    return XA;
};

enum class cubic_flag {not_set, PR, SRK};
inline auto get_cubic_flag(const std::string& s) {
    if (s == "PR") { return cubic_flag::PR; }
    else if (s == "SRK") { return cubic_flag::SRK; }
    else {
        throw std::invalid_argument("bad cubic flag:" + s);
    }
}


class CPACubic {
private:
    const std::valarray<double> a0, bi, c1, Tc;
    double delta_1, delta_2;
    const double R_gas;
    const std::optional<std::vector<std::vector<double>>> kmat;
public:
    CPACubic(cubic_flag flag, const std::valarray<double> &a0, const std::valarray<double> &bi, const std::valarray<double> &c1, const std::valarray<double> &Tc, const double R_gas, const std::optional<std::vector<std::vector<double>>> & kmat = std::nullopt) : a0(a0), bi(bi), c1(c1), Tc(Tc), R_gas(R_gas), kmat(kmat) {
        switch (flag) {
        case cubic_flag::PR:
        { delta_1 = 1 + sqrt(2.0); delta_2 = 1 - sqrt(2.0); break; }
        case cubic_flag::SRK:
        { delta_1 = 1; delta_2 = 0; break; }
        default:
            throw std::invalid_argument("Bad cubic flag");
        }
    };
    
    std::size_t size() const {return a0.size(); }

    template<typename VecType>
    auto R(const VecType& /*molefrac*/) const { return R_gas; }

    template<typename TType>
    auto get_ai(TType T, int i) const {
        return forceeval(a0[i] * POW2(1.0 + c1[i]*(1.0 - sqrt(T / Tc[i]))));
    }

    template<typename TType, typename VecType>
    auto get_ab(const TType T, const VecType& molefrac) const {
        using return_type = std::common_type_t<decltype(T), decltype(molefrac[0])>;
        return_type asummer = 0.0, bsummer = 0.0;
        for (auto i = 0U; i < molefrac.size(); ++i) {
            bsummer += molefrac[i] * bi[i];
            auto ai = get_ai(T, i);
            for (auto j = 0U; j < molefrac.size(); ++j) {
                auto aj = get_ai(T, j);
                double kij = (kmat) ? kmat.value()[i][j] : 0.0;
                auto a_ij = (1.0 - kij) * sqrt(ai * aj);
                asummer += molefrac[i] * molefrac[j] * a_ij;
            }
        }
        return std::make_tuple(asummer, bsummer);
    }

    template<typename TType, typename RhoType, typename VecType>
    auto alphar(const TType T, const RhoType rhomolar, const VecType& molefrac) const {
        auto [a_cubic, b_cubic] = get_ab(T, molefrac);
        return forceeval(
            -log(1.0 - b_cubic * rhomolar) // repulsive part
            -a_cubic/R_gas/T*log((delta_1*b_cubic*rhomolar + 1.0) / (delta_2*b_cubic*rhomolar + 1.0)) / b_cubic / (delta_1 - delta_2) // attractive part
        );
    }
};

/** Implement the association approach of Huang & Radosz for pure fluids
 */
class CPAAssociation {
private:
    const std::vector<association_classes> classes;
    const radial_dist dist;
    const std::valarray<double> epsABi, betaABi, bi;
    const std::vector<int> N_sites;
    const double R_gas;

    auto get_N_sites(const std::vector<association_classes> &the_classes) {
        std::vector<int> N_sites_out;
        auto get_N = [](auto cl) {
            switch (cl) {
            case association_classes::a1A: return 1;
            case association_classes::a2B: return 2;
            case association_classes::a3B: return 3;
            case association_classes::a4C: return 4;
            default: throw std::invalid_argument("Bad association class");
            }
        };
        for (auto cl : the_classes) {  
            N_sites_out.push_back(get_N(cl));
        }
        return N_sites_out;
    }

public:
    CPAAssociation(const std::vector<association_classes>& classes, const radial_dist dist, const std::valarray<double> &epsABi, const std::valarray<double> &betaABi, const std::valarray<double> &bi, double R_gas)
        : classes(classes), dist(dist), epsABi(epsABi), betaABi(betaABi), bi(bi), N_sites(get_N_sites(classes)), R_gas(R_gas) {};
    
    nlohmann::json get_assoc_calcs(double T, double rhomolar, const Eigen::ArrayXd& mole_fractions) const{
        
        auto fromArrayX = [](const Eigen::ArrayXd &x){std::valarray<double>n(x.size()); for (auto i = 0U; i < n.size(); ++i){ n[i] = x[i];} return n;};
        
        // Calculate b of the mixture
        auto b_cubic = (Eigen::Map<const Eigen::ArrayXd>(&bi[0], bi.size())*mole_fractions).sum();
        
        // Calculate the fraction of sites not bonded with other active sites
        auto RT = forceeval(R_gas * T); // R times T
        auto XA = XA_calc_pure(N_sites[0], classes[0], dist, epsABi[0], betaABi[0], b_cubic, RT, rhomolar, mole_fractions);
        
        return {
            {"b_mix", b_cubic},
            {"X_A", fromArrayX(XA)},
            {"note", "X_A is the fraction of non-bonded sites for each site type"}
        };
    }

    template<typename TType, typename RhoType, typename VecType>
    auto alphar(const TType& T, const RhoType& rhomolar, const VecType& molefrac) const {
        // Calculate b of the mixture
        auto b_cubic = (Eigen::Map<const Eigen::ArrayXd>(&bi[0], bi.size())*molefrac).sum();

        // Calculate the fraction of sites not bonded with other active sites
        auto RT = forceeval(R_gas * T); // R times T
        auto XA = XA_calc_pure(N_sites[0], classes[0], dist, epsABi[0], betaABi[0], b_cubic, RT, rhomolar, molefrac);

        using return_type = std::common_type_t<decltype(T), decltype(rhomolar), decltype(molefrac[0])>;
        return_type alpha_r_asso = 0.0;
        auto i = 0;
        for (auto xi : molefrac){ // loop over all components
            auto XAi = XA.col(i);
            alpha_r_asso += forceeval(xi * (log(XAi) - XAi / 2).sum());
            alpha_r_asso += xi*static_cast<double>(N_sites[i])/2;
            i++;
        }
        return forceeval(alpha_r_asso);
    }
};

template <typename Cubic, typename Assoc>
class CPAEOS {
public:
    const Cubic cubic;
    const Assoc assoc;

    template<class VecType>
    auto R(const VecType& molefrac) const {
        return cubic.R(molefrac);
    }

    CPAEOS(Cubic &&cubic, Assoc &&assoc) : cubic(cubic), assoc(assoc) {
    }

    /// Residual dimensionless Helmholtz energy from the SRK or PR core and contribution due to association
    /// alphar = a/(R*T) where a and R are both molar quantities
    template<typename TType, typename RhoType, typename VecType>
    auto alphar(const TType& T, const RhoType& rhomolar, const VecType& molefrac) const {
        if (static_cast<std::size_t>(molefrac.size()) != cubic.size()){
            throw teqp::InvalidArgument("Mole fraction size is not correct; should be " + std::to_string(cubic.size()));
        }

        // Calculate the contribution to alphar from the conventional cubic EOS
        auto alpha_r_cubic = cubic.alphar(T, rhomolar, molefrac);

        // Calculate the contribution to alphar from association
        auto alpha_r_assoc = assoc.alphar(T, rhomolar, molefrac);

        return forceeval(alpha_r_cubic + alpha_r_assoc);
    }
};

struct AssociationVariantWrapper{
    using vartype = std::variant<CPAAssociation, association::Association>;
    const vartype holder;
    
    AssociationVariantWrapper(const vartype& holder) : holder(holder) {};
    
    template<typename TType, typename RhoType, typename MoleFracsType>
    auto alphar(const TType& T, const RhoType& rhomolar, const MoleFracsType& molefracs) const{
        return std::visit([&](auto& h){ return h.alphar(T, rhomolar, molefracs); }, holder);
    }
    
    auto get_assoc_calcs(double T, double rhomolar, const Eigen::ArrayXd& mole_fractions) const {
        return std::visit([&](auto& h){ return h.get_assoc_calcs(T, rhomolar, mole_fractions); }, holder);
    }
        
};

/// A factory function to return an instantiated CPA instance given
/// the JSON representation of the model
inline auto CPAfactory(const nlohmann::json &j){
    auto build_cubic = [](const auto& j) {
        auto N = j["pures"].size();
        std::valarray<double> a0i(N), bi(N), c1(N), Tc(N);
        std::vector<std::vector<double>> kmat;
        if (j.contains("kmat")){
            kmat = j.at("kmat");
            std::string kmaterr = "The kmat is the wrong size. It should be square with dimension " + std::to_string(N);
            if (kmat.size() != N){
                throw teqp::InvalidArgument(kmaterr);
            }
            else{
                for (auto& krow: kmat){
                    if(krow.size() != N){
                        throw teqp::InvalidArgument(kmaterr);
                    }
                }
            }
        }
        else{
            kmat.resize(N); for (auto i = 0U; i < N; ++i){ kmat[i].resize(N); for (auto k = 0U; k < N; ++k){kmat[i][k] = 0.0;} }
        }
        
        std::size_t i = 0;
        for (auto p : j["pures"]) {
            a0i[i] = p["a0i / Pa m^6/mol^2"];
            bi[i] = p["bi / m^3/mol"];
            c1[i] = p["c1"];
            Tc[i] = p["Tc / K"];
            i++;
        }
        return CPACubic(get_cubic_flag(j["cubic"]), a0i, bi, c1, Tc, j["R_gas / J/mol/K"], kmat);
    };
    
	auto build_assoc_pure = [](const auto& j) -> AssociationVariantWrapper{
        auto N = j["pures"].size();
        if (N == 1 && j.at("pures").at(0).contains("class") ){
            // This is the backwards compatible approach
            // with the old style of defining the association class {1,2B...}
            std::vector<association_classes> classes;
            auto dist = get_radial_dist(j.at("radial_dist"));
            std::valarray<double> epsABi(N), betaABi(N), bi(N);
            std::size_t i = 0;
            for (auto p : j.at("pures")) {
                epsABi[i] = p.at("epsABi / J/mol");
                betaABi[i] = p.at("betaABi");
                bi[i] = p.at("bi / m^3/mol");
                classes.push_back(get_association_classes(p.at("class")));
                i++;
            }
            return AssociationVariantWrapper{CPAAssociation(classes, dist, epsABi, betaABi, bi, j["R_gas / J/mol/K"])};
        }
        else{
            // This branch uses the new code
            Eigen::ArrayXd b_m3mol(N), beta(N), epsilon_Jmol(N);
            association::AssociationOptions opt;
            opt.radial_dist = get_radial_dist(j.at("radial_dist"));
            if (j.contains("options")){
                opt = j.at("options"); // Pulls in the options that are POD types
            }
            // Copy over the self-association mask
            if (j.contains("/options/self_association_mask"_json_pointer)){
                opt.self_association_mask = j.at("/options/self_association_mask"_json_pointer).template get<std::vector<bool>>();
            }
            
            std::vector<std::vector<std::string>> molecule_sites;
            std::size_t i = 0;
            std::set<std::string> unique_site_types;
            for (auto p : j["pures"]) {
                epsilon_Jmol[i] = p.at("epsABi / J/mol");
                beta[i] = p.at("betaABi");
                b_m3mol[i] = p.at("bi / m^3/mol");
                molecule_sites.push_back(p.at("sites"));
                for (auto & s : molecule_sites.back()){
                    unique_site_types.insert(s);
                }
                i++;
            }
            if (j.contains("options") && j.at("options").contains("interaction_partners")){
                opt.interaction_partners = j.at("options").at("interaction_partners");
                for (auto [k,partners] : opt.interaction_partners){
                    if (unique_site_types.count(k) == 0){
                        throw teqp::InvalidArgument("Site is invalid in interaction_partners: " + k);
                    }
                    for (auto& partner : partners){
                        if (unique_site_types.count(partner) == 0){
                            throw teqp::InvalidArgument("Partner " + partner + " is invalid for site " + k);
                        }
                    }
                }
            }
            else{
                // Every site type is assumed to interact with every other site type, except for itself
                for (auto& site1 : unique_site_types){
                    std::vector<std::string> partners;
                    for (auto& site2: unique_site_types){
                        if (site1 != site2){
                            partners.push_back(site2);
                        }
                    }
                    opt.interaction_partners[site1] = partners;
                }
            }
            
            return AssociationVariantWrapper{association::Association(b_m3mol, beta, epsilon_Jmol, molecule_sites, opt)};
        }
    };
	return CPAEOS(build_cubic(j), build_assoc_pure( j));
}

}; /* namespace CPA */

}; // namespace teqp
