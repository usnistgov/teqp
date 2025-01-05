/**
General routines for the calculation of association
 
The implementation follows the approach of Langenbach for the index compression,
 
 Many helpful hints from Andres Riedemann
 
*/

#pragma once

#include <map>
#include <set>
#include <variant>

#include "teqp/constants.hpp"
#include "teqp/types.hpp"
#include "teqp/exceptions.hpp"

#include <Eigen/Dense>
#include "teqp/math/pow_templates.hpp"
#include "teqp/models/association/association_types.hpp"
#include "teqp/json_tools.hpp"

namespace teqp{
namespace association{

struct AssociationOptions{
    std::map<std::string, std::vector<std::string>> interaction_partners;
    std::vector<std::string> site_order;
    association::radial_dists radial_dist;
    association::Delta_rules Delta_rule = association::Delta_rules::CR1;
    std::vector<bool> self_association_mask;
    bool allow_explicit_fractions=true;
    double alpha = 0.5;
    double rtol = 1e-12, atol = 1e-12;
    int max_iters = 100;
};
inline void from_json(const nlohmann::json& j, AssociationOptions& o) {
    if (j.contains("alpha")){ j.at("alpha").get_to(o.alpha); }
    if (j.contains("rtol")){ j.at("rtol").get_to(o.rtol); }
    if (j.contains("atol")){ j.at("atol").get_to(o.atol); }
    if (j.contains("max_iters")){ j.at("max_iters").get_to(o.max_iters); }
}


namespace DufalMatrices{
    extern const std::unordered_map<int, Eigen::MatrixXd> bcoeffs;
}

/***
 A general class for calculating association fractions and association energy for mixtures
 
 A mixture is formed of multiple components, each component has an index i.
 For each component, it may multiple unique sites, and each unique site has a multiplicity associated with it.
 
 For instance, for 4C water, there are two "e" sites and two "H" sites in the nomenclature of Clapeyron.
 Thus the "e" site has multiplicity of 2 and the "H" site has multiplicity of 2
 */
class Association{
public:
    using CompSite = std::tuple<std::size_t, std::string>;
    using CompCIndex = std::tuple<std::size_t, std::size_t>;
    struct IndexMapper{
        std::map<std::size_t, CompSite> to_CompSite;
        std::map<CompSite, std::size_t> to_siteid;
        std::map<CompCIndex, std::size_t> CompCIndex_to_siteid;
        Eigen::ArrayXi counts; ///< An array of counts of each siteid for the entire mixture
        Eigen::ArrayXi N_sites; ///< How many total sites are on each molecule
        Eigen::ArrayXi N_unique_sites; ///< How many unique types of sites are on each molecule
        Eigen::ArrayXi comp_index; ///< The pure component indices associated with each siteid
        std::vector<std::vector<std::string>> molecule_sites;
    };
private:
    IndexMapper make_mapper(const std::vector<std::vector<std::string>>& molecule_sites, const AssociationOptions& options_in) const {
        IndexMapper ind;
        ind.counts.resize(1000);
        ind.comp_index.resize(1000);
        ind.N_sites.resize(molecule_sites.size());
        ind.N_unique_sites.resize(molecule_sites.size());
        ind.molecule_sites = molecule_sites;
                
        // Build the maps from siteid->(component index, site name) and (component index, site name)->siteid
        std::size_t icomp = 0;
        std::size_t siteid = 0;
        for (auto& molecule: molecule_sites){
            /// Count up how many times each site is present in the list of sites
            std::map<std::string, int> site_counts;
            for (auto& site: molecule){
                ++site_counts[site];
            }
            auto unique_sites_on_molecule = std::set(molecule.begin(), molecule.end());
            if (!options_in.site_order.empty()){
                // TODO: enforce sites to appear in the order matching the specification
                // TODO: this would be required to for instance check the D matrix of Langenbach and Enders
            }
            std::size_t Cindex = 0;
            for (auto& site: unique_sites_on_molecule){
                CompSite cs{icomp, site};
                ind.to_CompSite[siteid] = cs;
                ind.to_siteid[cs] = siteid;
                ind.CompCIndex_to_siteid[CompCIndex{icomp, Cindex}] = siteid;
                ind.counts[siteid] = site_counts[site];
                ind.comp_index[siteid] = static_cast<int>(icomp);
                Cindex++;
                siteid++;
            }
            ind.N_sites[icomp] = static_cast<int>(molecule.size());
            ind.N_unique_sites[icomp] = static_cast<int>(unique_sites_on_molecule.size());
            icomp++;
        }
        
        ind.counts.conservativeResize(siteid);
        ind.comp_index.conservativeResize(siteid);
        return ind;
    }
    
    /***
    Construct the counting matrix \f$ D_{IJ} \f$ as given by Langenbach and Enders
    */
    auto make_D(const IndexMapper& ind, const AssociationOptions& options_in ) const{
        
        auto get_DIJ = [&ind, &options_in](std::size_t I, std::size_t J) -> int {
            /** Return the value of an entry in the D_{IJ} matrix
            
            For a given unique site, look at all other sites on all other molecules
            */
            auto [ph1, typei] = ind.to_CompSite.at(I);
            auto [ph2, typej] = ind.to_CompSite.at(J);
            
            // If self-association is disabled for this site, then return zero for the D matrix
            if (!options_in.self_association_mask.empty() && ph1 == ph2 && !options_in.self_association_mask[ph1]){
                return 0;
            }
            auto contains = [](auto& container, const auto& val){ return std::find(container.begin(), container.end(), val) != container.end(); };
            /// If interaction parameters are not provided, assume conservatively that all sites can interact with all other sites
            if (options_in.interaction_partners.empty() || (contains(options_in.interaction_partners.at(typei), typej))){
                return ind.counts[J];
            }
            return 0;
        };
        if (!options_in.self_association_mask.empty() && options_in.self_association_mask.size() != static_cast<std::size_t>(ind.N_sites.size())){
            throw teqp::InvalidArgument("self_association_mask is of the wrong size");
        }
        int Ngroups = static_cast<int>(ind.to_siteid.size());
        Eigen::ArrayXXi Dmat(Ngroups, Ngroups);
        // I and J are the numerical indices of the sites
        for (int I = 0; I < Ngroups; ++I){
            for (int J = 0; J < Ngroups; ++J){
                Dmat(I, J) = get_DIJ(I, J);
            }
        }
        return Dmat;
    }
    static auto toEig(const nlohmann::json& j, const std::string& k) -> Eigen::ArrayXd{
        std::vector<double> vec = j.at(k);
        return Eigen::Map<Eigen::ArrayXd>(&vec[0], vec.size());
    };
    static auto get_association_options(const nlohmann::json&j){
        AssociationOptions opt;
        if (j.contains("options")){
            opt = j.at("options").get<AssociationOptions>();
            // Copy over the self-association mask
            if (j.contains("/options/self_association_mask"_json_pointer)){
                opt.self_association_mask = j.at("/options/self_association_mask"_json_pointer).template get<std::vector<bool>>();
            }
        }
        return opt;
    }
public:
    const AssociationOptions options;
    const IndexMapper mapper;
    const Eigen::ArrayXXi D;
    const Delta_rules m_Delta_rule;
    const std::variant<CanonicalData, DufalData> datasidecar;
    
    Association(const Eigen::ArrayXd& b_m3mol, const Eigen::ArrayXd& beta, const Eigen::ArrayXd& epsilon_Jmol, const std::vector<std::vector<std::string>>& molecule_sites, const AssociationOptions& options) : options(options), mapper(make_mapper(molecule_sites, options)), D(make_D(mapper, options)), m_Delta_rule(options.Delta_rule), datasidecar(CanonicalData{b_m3mol, beta, epsilon_Jmol, options.radial_dist}){
        if (options.Delta_rule != Delta_rules::CR1){
            throw std::invalid_argument("Delta rule is invalid; options are: {CR1}");
        }
    }
    Association(const decltype(datasidecar)& data, const std::vector<std::vector<std::string>>& molecule_sites, const AssociationOptions& options) : options(options), mapper(make_mapper(molecule_sites, options)), D(make_D(mapper, options)), m_Delta_rule(options.Delta_rule), datasidecar(data) {
    }
    static Association factory(const nlohmann::json& j){
        
        // Collect the set of unique site types among all the molecules
        std::set<std::string> unique_site_types;
        for (auto molsite : j.at("molecule_sites")) {
            for (auto & s : molsite){
                unique_site_types.insert(s);
            }
        }
                
        auto get_interaction_partners = [&](const nlohmann::json& j){
            std::map<std::string, std::vector<std::string>> interaction_partners;
            
            if (j.contains("options") && j.at("options").contains("interaction_partners")){
                interaction_partners = j.at("options").at("interaction_partners");
                for (auto [k,partners] : interaction_partners){
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
                    interaction_partners[site1] = partners;
                }
            }
            return interaction_partners;
        };
        
        if (j.contains("Delta_rule")){
            std::string Delta_rule = j.at("Delta_rule");
            if (Delta_rule == "CR1"){
                CanonicalData data;
                data.b_m3mol = toEig(j, "b / m^3/mol");
                data.beta = toEig(j, "beta");
                data.epsilon_Jmol = toEig(j, "epsilon / J/mol");
                auto options =  get_association_options(j);
                options.Delta_rule = Delta_rules::CR1;
                data.radial_dist = options.radial_dist;
                options.interaction_partners = get_interaction_partners(j);
                return {data, j.at("molecule_sites"), options};
            }
            else if(Delta_rule == "Dufal"){
                DufalData data;
                
                // Parameters for the dispersive part
                data.sigma_m = toEig(j, "sigma / m");
                if (j.contains("epsilon / J/mol")){
                    data.epsilon_Jmol = toEig(j, "epsilon / J/mol");
                }
                else if (j.contains("epsilon/kB / K")){
                    data.epsilon_Jmol = toEig(j, "epsilon/kB / K")*constants::R_CODATA2017;
                }
                else{
                    throw teqp::InvalidArgument("One of the epsilon variables must be provided");
                }
                data.lambda_r = toEig(j, "lambda_r");
                data.kmat = build_square_matrix(j.at("kmat"));
                // Parameters for the associating part
                data.epsilon_HB_Jmol = toEig(j, "epsilon_HB / J/mol");
                data.K_HB_m3 = toEig(j, "K_HB / m^3");
                data.apply_mixing_rules();
                
                auto options =  get_association_options(j);
                options.Delta_rule = Delta_rules::Dufal;
                options.interaction_partners = get_interaction_partners(j);
                return {data, j.at("molecule_sites"), options};
            }
        }
        throw std::invalid_argument("The Delta_rule has not been specified");
    }
    
    /**
        Build the Delta matrix, where entries are given by
     \f[
        \Delta_{IJ} = ...
     \f]
     */
    template<typename TType, typename RhoType, typename MoleFracsType>
    auto get_Delta(const TType& T, const RhoType& rhomolar, const MoleFracsType& molefracs) const {
        
        using resulttype = std::common_type_t<decltype(T), decltype(rhomolar), decltype(molefracs[0])>; // Type promotion, without the const-ness
        using Mat = Eigen::Array<resulttype, Eigen::Dynamic, Eigen::Dynamic>;
        auto Ngroups = mapper.to_CompSite.size();
        
        // Calculate the radial_dist if it is meaningful
        using eta_t = std::common_type_t<decltype(rhomolar), decltype(molefracs[0])>; // Type promotion, without the const-ness
        std::optional<eta_t> g;
        if (m_Delta_rule == Delta_rules::CR1){
            const CanonicalData& d = std::get<CanonicalData>(datasidecar);
            auto bmix = (molefracs*d.b_m3mol).sum();
            auto eta = bmix*rhomolar/4.0;
            switch(d.radial_dist){
                case radial_dists::CS:
                    g = (2.0-eta)/(2.0*(1.0-eta)*(1.0-eta)*(1.0-eta)); break;
                case radial_dists::KG:
                    g = 1.0 / (1.0 - 1.9*eta); break;
                default:
                    throw std::invalid_argument("Bad radial distribution");
            }
        }
        
        /// A helper function for the I integral representation of Dufal (http://dx.doi.org/10.1080/00268976.2015.1029027)
        auto get_I_Dufal = [](const auto& Tstar, const auto& rhostar, const auto& lambda_r){
            
            using result_t = std::decay_t<std::common_type_t<decltype(Tstar), decltype(rhostar), decltype(lambda_r)>>;
            result_t summer = 0.0;
            
            std::decay_t<decltype(rhostar)> rhostar_to_i = 1.0;
            for (auto i = 0U; i <= 10; ++i){
                std::decay_t<decltype(Tstar)> Tstar_to_j = 1.0;
                for (auto j = 0U; i + j <= 10; ++j){
                    double aij = 0.0, lambdar_to_k = 1.0;
                    for (auto k = 0; k <= 6; ++k){
                        double bijk = DufalMatrices::bcoeffs.at(k)(i,j); 
                        aij += bijk*lambdar_to_k;
                        lambdar_to_k *= lambda_r;
                    }
                    summer += aij*rhostar_to_i*Tstar_to_j;
                    Tstar_to_j *= Tstar;
                }
                rhostar_to_i *= rhostar;
            }
            return summer;
        };
        
        using rhostar_t = std::common_type_t<decltype(rhomolar), decltype(molefracs[0])>; // Type promotion, without the const-ness
        std::optional<rhostar_t> rhostar;
        if (m_Delta_rule == Delta_rules::Dufal){
            const DufalData& d = std::get<DufalData>(datasidecar);
            // Calculate the one-fluid vdW1 sigma from Eq. 40 of Dufal
            std::decay_t<decltype(molefracs[0])> sigma3_vdW1 = 0.0;
            auto N = molefracs.size();
            for (auto i = 0U; i < N; ++i){
                for (auto j = 0U; j < N; ++j){
                    double sigma3_ij_m3 = POW3((d.sigma_m[i] + d.sigma_m[j])/2.0);
                    sigma3_vdW1 += molefracs[i]*molefracs[j]*sigma3_ij_m3;
                }
            }
            rhostar = rhomolar*N_A*sigma3_vdW1;
        }
        
        Mat Delta = Mat::Zero(Ngroups, Ngroups);
        for (auto I = 0U; I < Ngroups; ++I){
            auto i = std::get<0>(mapper.to_CompSite.at(I));
            for (auto J = 0U; J < Ngroups; ++J){
                auto j = std::get<0>(mapper.to_CompSite.at(J));
                
                using namespace teqp::constants;
                
                if (m_Delta_rule == Delta_rules::CR1){
                    // The CR1 rule is used to calculate the cross contributions
                    const CanonicalData& d = std::get<CanonicalData>(datasidecar);
                    auto b_ij = (d.b_m3mol[i] + d.b_m3mol[j])/2.0;
                    auto epsilon_ij_Jmol = (d.epsilon_Jmol[i] + d.epsilon_Jmol[j])/2.0;
                    auto beta_ij = sqrt(d.beta[i]*d.beta[j]);
                    Delta(I, J) = g.value()*b_ij*beta_ij*(exp(epsilon_ij_Jmol/(R_CODATA2017*T))-1.0)/N_A; // m^3
                }
                else if (m_Delta_rule == Delta_rules::Dufal){
                    const DufalData& d = std::get<DufalData>(datasidecar);
                    auto Tstar = forceeval(T/d.EPSILONOVERKij_K(i,j));
                    auto _I = get_I_Dufal(Tstar, rhostar.value(), d.LAMBDA_Rij(i, j));
                    auto F_Meyer = exp(d.EPSILONOVERK_HBij_K(i, j)/T)-1.0;
                    Delta(I, J) = F_Meyer*d.KHBij_m3(i,j)*_I;
                }
                else{
                    throw std::invalid_argument("Don't know what to do with this Delta rule");
                }
            }
        }
        return Delta;
    }
    
    template<typename TType, typename RhoType, typename MoleFracsType, typename XType>
    auto successive_substitution(const TType& T, const RhoType& rhomolar, const MoleFracsType& molefracs, const XType& X_init) const {
        
        if (X_init.size() != static_cast<long>(mapper.to_siteid.size())){
            throw teqp::InvalidArgument("Wrong size of X_init; should be "+ std::to_string(mapper.to_siteid.size()));
        }
        
        using resulttype = std::common_type_t<decltype(T), decltype(rhomolar), decltype(molefracs[0])>; // Type promotion, without the const-ness
        using Mat = Eigen::Array<resulttype, Eigen::Dynamic, Eigen::Dynamic>;
        
        const Mat Delta = get_Delta(T, rhomolar, molefracs);
//        const Mat DD = Delta.array()*D.cast<resulttype>().array(); // coefficient-wise product, with D upcast from int to type of Delta matrix
        
        auto Ngroups = mapper.to_CompSite.size();
        Eigen::RowVectorX<typename MoleFracsType::Scalar> xj(Ngroups); // Mole fractions of the component containing each siteid
        for (auto I = 0U; I< Ngroups; ++I){
            xj(I) = molefracs(std::get<0>(mapper.to_CompSite.at(I)));
        }
        
        using rDDXtype = std::decay_t<std::common_type_t<typename decltype(Delta)::Scalar, decltype(rhomolar), decltype(molefracs[0])>>; // Type promotion, without the const-ness
        Eigen::ArrayX<std::decay_t<rDDXtype>> X = X_init.template cast<rDDXtype>(), Xnew;
        
        Eigen::MatrixX<rDDXtype> rDDX = rhomolar*N_A*(Delta.array()*D.cast<resulttype>().array()).matrix();
        for (auto j = 0; j < rDDX.rows(); ++j){
            rDDX.row(j).array() = rDDX.row(j).array()*xj.array().template cast<rDDXtype>();
        }
//        rDDX.rowwise() *= xj;
        
        // Use explicit solutions in the case that there is a pure
        // fluid with two kinds of sites, and no self-self interactions
        // between sites
        if (options.allow_explicit_fractions && molefracs.size() == 1 && mapper.counts.size() == 2 && (rDDX.matrix().diagonal().unaryExpr([](const auto&x){return getbaseval(x); }).array() == 0.0).all()){
            auto Delta_ = Delta(0, 1);
            auto kappa_A = rhomolar*N_A*static_cast<double>(mapper.counts[0])*Delta_;
            auto kappa_B = rhomolar*N_A*static_cast<double>(mapper.counts[1])*Delta_;
            // See the derivation in the docs in the association page; see also https://github.com/ClapeyronThermo/Clapeyron.jl/blob/494a75e8a2093a4b48ca54b872ff77428a780bb6/src/models/SAFT/association.jl#L463
            auto X_A1 = (kappa_A-kappa_B-sqrt(kappa_A*kappa_A-2.0*kappa_A*kappa_B + 2.0*kappa_A + kappa_B*kappa_B + 2.0*kappa_B+1.0)-1.0)/(2.0*kappa_A);
            auto X_A2 = (kappa_A-kappa_B+sqrt(kappa_A*kappa_A-2.0*kappa_A*kappa_B + 2.0*kappa_A + kappa_B*kappa_B + 2.0*kappa_B+1.0)-1.0)/(2.0*kappa_A);
            // Keep the positive solution, likely to be X_A2
            if (getbaseval(X_A1) < 0 && getbaseval(X_A2) > 0){
                X(0) = X_A2;
            }
            else if (getbaseval(X_A1) > 0 && getbaseval(X_A2) < 0){
                X(0) = X_A1;
            }
            auto X_B = 1.0/(1.0+kappa_A*X(0)); // From the law of mass-action
            X(1) = X_B;
            return X;
        }
        
        for (auto counter = 0; counter < options.max_iters; ++counter){
            // calculate the new array of non-bonded site fractions X
            Xnew = options.alpha*X + (1.0-options.alpha)/(1.0+(rDDX*X.matrix()).array());
            // These unaryExpr extract the numerical value from an Eigen array of generic type, allowing for comparison.
            // Otherwise for instance it is imposible to compare two complex numbers (if you are using complex step derivatives)
            auto diff = (Xnew-X).eval().cwiseAbs().unaryExpr([](const auto&x){return getbaseval(x); }).eval();
            auto tol = (options.rtol*Xnew + options.atol).unaryExpr([](const auto&x){return getbaseval(x); }).eval();
            if ((diff < tol).all()){
                break;
            }
            X = Xnew;
        }
        return X;
    }
    
    /**
     \brief Calculate the contribution \f$\alpha = a/(RT)\f$, where the Helmholtz energy \f$a\f$ is on a molar basis, making \f$\alpha\f$ dimensionless.
     */
    template<typename TType, typename RhoType, typename MoleFracsType>
    auto alphar(const TType& T, const RhoType& rhomolar, const MoleFracsType& molefracs) const {
        if (molefracs.size() != mapper.N_sites.size()){
            throw teqp::InvalidArgument("Wrong size of molefracs; should be "+ std::to_string(mapper.N_sites.size()));
        }
        
        // Do the sucessive substitution to obtain the non-bonded fractions for each unique site
        Eigen::ArrayXd X_init = Eigen::ArrayXd::Ones(mapper.to_siteid.size());
        auto X_A = successive_substitution(T, rhomolar, molefracs, X_init);
        
        // Calculate the contribution alpha based on the "naive" summation like in Clapeyron
        using resulttype = std::common_type_t<decltype(T), decltype(rhomolar), decltype(molefracs[0])>; // Type promotion, without the const-ness
        resulttype alpha_r_asso = 0.0;
        for (auto icomp = 0; icomp < molefracs.size(); ++icomp){ // loop over all components
            resulttype summer = 0.0;
            for (auto jsite = 0; jsite < mapper.N_unique_sites[icomp]; ++jsite){
                // Get the siteid given the (component index, site index on the component)
                const auto I = mapper.CompCIndex_to_siteid.at({icomp, jsite});
                
                // each contribution (per unique site on a molecule) is multiplied by its multiplicity within the molecule
                summer += (log(X_A(I))- X_A(I)/2.0 + 0.5)*static_cast<double>(mapper.counts(I));
            }
            alpha_r_asso += molefracs[icomp]*summer;
        }
        return alpha_r_asso;
        
//        // And do the summation to calculate the contribution to alpha=a/(RT)
//        using V = std::common_type_t<double, decltype(X_A[0]), int>;
//        auto xj = molefracs(mapper.comp_index); // mole fractions of the components attached at each site
//        auto alpha_r_asso = (xj.template cast<V>()*(log(X_A) - X_A/2.0 + 0.5).template cast<V>()*mapper.counts.cast<double>()).sum();
//        return forceeval(alpha_r_asso);
    }
    
    /**
    \brief Get things from the association calculations for debug purposes
     */
    nlohmann::json get_assoc_calcs(double T, double rhomolar, const Eigen::ArrayXd& mole_fractions) const{
        
        using Mat = Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic>;
        const Mat Delta = get_Delta(T, rhomolar, mole_fractions);
    
        Eigen::ArrayXd XAinit = Eigen::ArrayXd::Ones(mapper.to_siteid.size());
        auto XA = successive_substitution(T, rhomolar, mole_fractions, XAinit);
        
        auto fromArrayXd = [](const Eigen::ArrayXd &x){std::valarray<double>n(x.size()); for (auto i = 0U; i < n.size(); ++i){ n[i] = x[i];} return n;};
        auto fromArrayXXd = [](const Eigen::ArrayXXd &x){
            std::size_t N = x.rows();
            std::vector<std::vector<double>> n; n.resize(N);
            for (auto i = 0U; i < N; ++i){
                n[i].resize(N);
                for (auto j = 0U; j < N; ++j){
                    n[i][j] = x(i,j);
                }
            }
            return n;
        };
        auto fromArrayXXi = [](const Eigen::ArrayXXi &x){
            std::size_t N = x.rows();
            std::vector<std::vector<int>> n; n.resize(N);
            for (auto i = 0U; i < N; ++i){
                n[i].resize(N);
                for (auto j = 0U; j < N; ++j){
                    n[i][j] = x(i,j);
                }
            }
            return n;
        };
        return {
            {"to_CompSite", mapper.to_CompSite},
            {"to_siteid", mapper.to_siteid},
            {"counts", mapper.counts},
            {"D", fromArrayXXi(D.array())},
            {"Delta", fromArrayXXd(Delta.array())},
            {"X_A", fromArrayXd(XA.array())},
            {"self_association_mask", options.self_association_mask},
            {"note", "X_A is the fraction of non-bonded sites for each siteid"}
        };
    }
    
};

}
}
