/**
 This header is derived from the code in https://github.com/usnistgov/COSMOSAC, and the methodology is described in the paper:
 
 Ian H. Bell, Erik Mickoleit, Chieh-Ming Hsieh, Shiang-Tai Lin, Jadran Vrabec, Cornelia Breitkopf, and Andreas Jäger
 Journal of Chemical Theory and Computation 2020 16 (4), 2635-2646
 DOI: 10.1021/acs.jctc.9b01016
 */

namespace teqp::activity::activity_models::COSMOSAC{

/**
 A single sigma profile.  In this data structure (for consistency with the 2005 Virginia Tech
 database of COSMO-SAC parameters), the first column is the electron density in [e/A^2], and the second column
 is the probability of finding a segment with this electron density, multiplied by the segment area, in A^2
 */
class SigmaProfile {
public:
    Eigen::ArrayXd m_sigma, m_psigmaA;
    /// Default constructor
    SigmaProfile() {};
    /// Copy-by-reference constructor
    SigmaProfile(const Eigen::ArrayXd &sigma, const Eigen::ArrayXd &psigmaA) : m_sigma(sigma), m_psigmaA(psigmaA) {};
    /// Copy-by-reference constructor with std::vector
    SigmaProfile(const std::vector<double> &sigma, const std::vector<double> &psigmaA) : m_sigma(Eigen::Map<const Eigen::ArrayXd>(&(sigma[0]), sigma.size())), m_psigmaA(Eigen::Map<const Eigen::ArrayXd>(&(psigmaA[0]), psigmaA.size())) {};
    /// Move constructor
    SigmaProfile(Eigen::ArrayXd &&sigma, Eigen::ArrayXd &&psigmaA) : m_sigma(sigma), m_psigmaA(psigmaA) {};
    const Eigen::ArrayXd &psigmaA() const { return m_psigmaA; }
    const Eigen::ArrayXd &sigma() const { return m_sigma; }
    const Eigen::ArrayXd psigma(double A_i) const { return m_psigmaA/A_i; }
};

// A holder class for the sigma profiles for a given fluid
struct FluidSigmaProfiles {
    SigmaProfile nhb, ///< The profile for the non-hydrogen-bonding segments
    oh, ///< The profile for the OH-bonding segments
    ot; ///< The profile for the "other" segments
};

struct CombinatorialConstants {
    double q0 = 79.53, // [A^2]
        r0 = 66.69, // [A^3]
        z_coordination = 10.0;
    std::string to_string() {
        return "q0: " + std::to_string(q0) + " A^2 \nr0: " + std::to_string(r0) + " A^3\nz_coordination: " + std::to_string(z_coordination);
    }
};



struct COSMO3Constants {
    double
    AEFFPRIME = 7.25, // [A^2]
    c_OH_OH = 4013.78, // [kcal A^4/(mol e^2)]
    c_OT_OT = 932.31, // [kcal A^4 /(mol e^2)]
    c_OH_OT = 3016.43, // [kcal A^4 /(mol e^2)]
    A_ES = 6525.69, // [kcal A^4 /(mol e^2)]
    B_ES = 1.4859e8, // [kcal A^4 K^2/(mol e^2)]
    N_A = 6.022140758e23, // [mol^{-1}]
    k_B = 1.38064903e-23, // [J K^{-1}]
    R = k_B*N_A/4184, // [kcal/(mol*K)]; Universal gas constant of new redefinition of 2018, https://doi.org/10.1088/1681-7575/aa99bc
    Gamma_rel_tol = 1e-8; // relative tolerance for Gamma in iterative loop
    bool fast_Gamma = true;
    std::string to_string() {
        return "NOT IMPLEMENTED YET";
        //return "c_hb: " + std::to_string(c_hb) + " kcal A^4 /(mol*e^2) \nsigma_hb: " + std::to_string(sigma_hb) + " e/A^2\nalpha_prime: " + std::to_string(alpha_prime) + " kcal A^4 /(mol*e^2)\nAEFFPRIME: " + std::to_string(AEFFPRIME) + " A\nR: " + std::to_string(R) + " kcal/(mol*K)";
    }
};

enum class profile_type { NHB_PROFILE, OH_PROFILE, OT_PROFILE };

class COSMO3 {
private:
    std::vector<double> A_COSMO_A2; ///< The area per fluid, in \AA^2
    std::vector<double> V_COSMO_A3; ///< The volume per fluid, in \AA^3
    std::vector<FluidSigmaProfiles> profiles; ///< The vector of profiles, one per fluid
    COSMO3Constants m_consts;
    COSMOSAC::CombinatorialConstants m_comb_consts;
    Eigen::Index ileft, w;
public:
    COSMO3(const std::vector<double>& A_COSMO_A2, const std::vector<double>& V_COSMO_A3, const std::vector<FluidSigmaProfiles> &SigmaProfiles, const COSMO3Constants &constants = COSMO3Constants(), const CombinatorialConstants &comb_constants = CombinatorialConstants())
    : A_COSMO_A2(A_COSMO_A2), V_COSMO_A3(V_COSMO_A3), profiles(SigmaProfiles), m_consts(constants), m_comb_consts(comb_constants) {
        Eigen::Index iL, iR;
        std::tie(iL, iR) = get_nonzero_bounds();
        this->ileft = iL; this->w = iR - iL + 1;
    };
    
    /// Determine the left and right bounds for the psigma that is nonzero to accelerate the iterative calculations
    std::tuple<Eigen::Index, Eigen::Index> get_nonzero_bounds(){
        // Determine the range of entries in p(sigma) that are greater than zero, we
        // will only calculate segment activity coefficients for these segments
        Eigen::Index min_ileft = 51, max_iright = 0;
        for (auto i = 0U; i < profiles.size(); ++i) {
            const Eigen::ArrayXd psigma = profiles[i].nhb.psigma(A_COSMO_A2[i]);
            Eigen::Index ileft_ = 0, iright_ = psigma.size();
            for (auto ii = 0; ii < psigma.size(); ++ii) { if (std::abs(psigma(ii)) > 1e-16) { ileft_ = ii; break; } }
            for (auto ii = psigma.size() - 1; ii > ileft_; --ii) { if (std::abs(psigma(ii)) > 1e-16) { iright_ = ii; break; } }
            if (ileft_ < min_ileft) { min_ileft = ileft_; }
            if (iright_ > max_iright) { max_iright = iright_; }
        }
        return std::make_tuple(min_ileft, max_iright);
    }
    
    template<typename MoleFractions>
    auto get_psigma_mix(const MoleFractions &z, profile_type type = profile_type::NHB_PROFILE) const {
        Eigen::ArrayX<std::decay_t<decltype(z[0])>> psigma_mix(51); psigma_mix.fill(0);
        std::decay_t<decltype(z[0])> xA = 0.0;
        for (auto i = 0; i < z.size(); ++i) {
            switch (type) {
                case profile_type::NHB_PROFILE:
                    psigma_mix += z[i] * profiles[i].nhb.psigmaA(); break;
                case profile_type::OH_PROFILE:
                    psigma_mix += z[i] * profiles[i].oh.psigmaA(); break;
                case profile_type::OT_PROFILE:
                    psigma_mix += z[i] * profiles[i].ot.psigmaA(); break;
            }
            xA += z[i] * A_COSMO_A2[i];
        }
        psigma_mix /= xA;
        return psigma_mix;
    }
    
    /// Get access to the parameters that are in use
    COSMO3Constants &get_mutable_COSMO_constants() { return m_consts; }
    
    std::tuple<Eigen::Index, Eigen::Index> get_ileftw() const { return std::make_tuple(ileft, w);} ;
    
    double get_c_hb(profile_type type1, profile_type type2) const{
        
        if (type1 == type2){
            if (type1 == profile_type::OH_PROFILE) {
                return m_consts.c_OH_OH;
            }
            else if (type1 == profile_type::OT_PROFILE) {
                return m_consts.c_OT_OT;
            }
            else {
                return 0.0;
            }
        }
        else if ((type1 == profile_type::OH_PROFILE && type2 == profile_type::OT_PROFILE)
                 || (type1 == profile_type::OT_PROFILE && type2 == profile_type::OH_PROFILE)) {
            return m_consts.c_OH_OT;
        }
        else {
            return 0.0;
        }
    }
    template<typename TType>
    auto get_DELTAW(const TType& T, profile_type type_t, profile_type type_s) const {
        auto delta_sigma = 2*0.025/50;
        Eigen::ArrayXX<TType> DELTAW(51, 51);
        double cc = get_c_hb(type_t, type_s);
        for (auto m = 0; m < 51; ++m) {
            for (auto n = 0; n < 51; ++n) {
                double sigma_m = -0.025 + delta_sigma*m,
                sigma_n = -0.025 + delta_sigma*n,
                c_hb = (sigma_m*sigma_n >= 0) ? 0 : cc;
                auto c_ES = m_consts.A_ES + m_consts.B_ES/(T*T);
                DELTAW(m, n) = c_ES*POW2(sigma_m + sigma_n) - c_hb*POW2(sigma_m-sigma_n);
            }
        }
        return DELTAW;
    }
    template<typename TType>
    auto get_DELTAW_fast(TType T, profile_type type_t, profile_type type_s) const {
        auto delta_sigma = 2 * 0.025 / 50;
        Eigen::ArrayXX<TType> DELTAW(51, 51); DELTAW.setZero();
        double cc = get_c_hb(type_t, type_s);
        for (auto m = ileft; m < ileft+w+1; ++m) {
            for (auto n = ileft; n < ileft+w+1; ++n) {
                double sigma_m = -0.025 + delta_sigma*m,
                sigma_n = -0.025 + delta_sigma*n,
                c_hb = (sigma_m*sigma_n >= 0) ? 0 : cc;
                auto c_ES = m_consts.A_ES + m_consts.B_ES / (T*T);
                DELTAW(m, n) = c_ES*POW2(sigma_m + sigma_n) - c_hb*POW2(sigma_m - sigma_n);
            }
        }
        return DELTAW;
    }
    
    template<typename TType, typename PSigma>
    auto get_AA(const TType& T, PSigma psigmas){
        // Build the massive \Delta W matrix that is 153*153 in size
        Eigen::ArrayXX<TType> DELTAW(51, 51); // Depends on temperature
        double R = m_consts.R;
        std::vector<profile_type> types = { profile_type::NHB_PROFILE, profile_type::OH_PROFILE, profile_type::OT_PROFILE };
        for (auto i = 0; i < 3; ++i) {
            for (auto j = 0; j < 3; ++j) {
                DELTAW.matrix().block(51 * i, 51 * j, 51, 51) = get_DELTAW(T, types[i], types[j]);
            }
        }
        return Eigen::exp(-DELTAW/(R*T)).rowwise()*psigmas.transpose();
    }
    /**
     Obtain the segment activity coefficients \f$\Gamma\f$ for given a set of charge densities
     
     If fast_Gamma is enabled, then only segments that have some contribution are included in the iteration (can be a significant acceleration for nonpolar+nonpolar mixtures)
     
     \param T Temperature, in K
     \param psigmas Charge densities, in the order of NHB, OH, OT.  Length is 153.
     */
    template<typename TType, typename PSigmaType>
    auto get_Gamma(const TType& T, PSigmaType psigmas) const {
        //auto startTime = std::chrono::high_resolution_clock::now();
        
        using TXType = std::decay_t<std::common_type_t<TType, decltype(psigmas[0])>>;
        
        double R = m_consts.R;
        Eigen::ArrayX<TXType> Gamma(153), Gammanew(153); Gamma.setOnes(); Gammanew.setOnes();
        
        // A convenience function to convert double values to string in scientific format
        auto to_scientific = [](double val) { std::ostringstream out; out << std::scientific << val; return out.str(); };
        
        auto max_iter = 500;
        if (!m_consts.fast_Gamma){
            // The slow and simple branch
            
            // Build the massive \Delta W matrix that is 153*153 in size
            Eigen::ArrayXX<TType> DELTAW(153, 153);
            std::vector<profile_type> types = { profile_type::NHB_PROFILE, profile_type::OH_PROFILE, profile_type::OT_PROFILE };
            for (auto i = 0; i < 3; ++i) {
                for (auto j = 0; j < 3; ++j) {
                    DELTAW.matrix().block(51*i, 51*j, 51, 51) = get_DELTAW(T, types[i], types[j]);
                }
            }
            
            auto AA = (Eigen::exp(-DELTAW / (R*T)).template cast<TXType>().rowwise()*psigmas.template cast<TXType>().transpose()).eval();
            for (auto counter = 0; counter <= max_iter; ++counter) {
                Gammanew = 1 / (AA.rowwise()*Gamma.transpose()).rowwise().sum();
                Gamma = (Gamma + Gammanew) / 2;
                double maxdiff = getbaseval(((Gamma - Gammanew) / Gamma).cwiseAbs().real().maxCoeff());
                if (maxdiff < m_consts.Gamma_rel_tol) {
                    break;
                }
                if (counter == max_iter) {
                    throw std::invalid_argument("Could not obtain the desired tolerance of "
                                                + to_scientific(m_consts.Gamma_rel_tol)
                                                + " after "
                                                + std::to_string(max_iter)
                                                + " iterations in get_Gamma; current value is "
                                                + to_scientific(maxdiff));
                }
            }
            return Gamma;
        }
        else {
            // The fast branch!
            // ----------------
            
            // Build the massive AA matrix that is 153*153 in size
            //auto midTime = std::chrono::high_resolution_clock::now();
            std::vector<profile_type> types = { profile_type::NHB_PROFILE, profile_type::OH_PROFILE, profile_type::OT_PROFILE };
            std::vector<Eigen::Index> offsets = {0*51, 1*51, 2*51};
            Eigen::ArrayXX<TXType> AA(153, 153);
            for (auto i = 0; i < 3; ++i) {
                for (auto j = 0; j < 3; ++j) {
                    Eigen::Index rowoffset = offsets[i], coloffset = offsets[j];
                    AA.matrix().block(rowoffset + ileft, coloffset + ileft, w, w) = Eigen::exp(-get_DELTAW_fast(T, types[i], types[j]).block(ileft, ileft, w, w).array() / (R*T)).template cast<TXType>().rowwise()*psigmas.template cast<TXType>().segment(coloffset+ileft,w).transpose();
                }
            }
            //auto midTime2 = std::chrono::high_resolution_clock::now();
            
            for (auto counter = 0; counter <= max_iter; ++counter) {
                for (Eigen::Index offset : {51*0, 51*1, 51*2}){
                    Gammanew.segment(offset + ileft, w) = 1 / (
                                                               AA.matrix().block(offset+ileft,51*0+ileft,w,w).array().rowwise()*Gamma.segment(51*0+ileft, w).transpose()
                                                               + AA.matrix().block(offset+ileft,51*1+ileft,w,w).array().rowwise()*Gamma.segment(51*1+ileft, w).transpose()
                                                               + AA.matrix().block(offset+ileft,51*2+ileft,w,w).array().rowwise()*Gamma.segment(51*2+ileft, w).transpose()
                                                               ).rowwise().sum();
                }
                for (Eigen::Index offset : {51 * 0, 51 * 1, 51 * 2}) {
                    Gamma.segment(offset + ileft, w) = (Gamma.segment(offset + ileft, w) + Gammanew.segment(offset + ileft, w)) / 2;
                }
                double maxdiff = getbaseval(((Gamma - Gammanew) / Gamma).cwiseAbs().real().maxCoeff());
                if (maxdiff < m_consts.Gamma_rel_tol) {
                    break;
                }
                if (!std::isfinite(maxdiff)){
                    throw teqp::InvalidArgument("Gammas are not finite");
                }
                if (counter == max_iter){
                    throw std::invalid_argument("Could not obtain the desired tolerance of "
                                                + to_scientific(m_consts.Gamma_rel_tol)
                                                +" after "
                                                +std::to_string(max_iter)
                                                +" iterations in get_Gamma; current value is "
                                                + to_scientific(maxdiff));
                }
            }
            //auto endTime = std::chrono::high_resolution_clock::now();
            //std::cout << std::chrono::duration<double>(midTime - startTime).count() << " s elapsed (DELTAW)\n";
            //std::cout << std::chrono::duration<double>(midTime2 - midTime).count() << " s elapsed (AA)\n";
            //std::cout << std::chrono::duration<double>(endTime - midTime2).count() << " s elapsed (comps)\n";
            //std::cout << std::chrono::duration<double>(endTime - startTime).count() << " s elapsed (total)\n";
            return Gamma;
        }
    }
    
    /**
     The residual part of ln(γ_i), for the i-th component
     */
    template<typename TType, typename Array>
    auto get_lngamma_resid(std::size_t i, TType T, const Array &lnGamma_mix) const
    {
        double AEFFPRIME = m_consts.AEFFPRIME;
        Eigen::ArrayX<double> psigmas(3*51); // For a pure fluid, p(sigma) does not depend on temperature
        double A_i = A_COSMO_A2[i];
        psigmas << profiles[i].nhb.psigma(A_i), profiles[i].oh.psigma(A_i), profiles[i].ot.psigma(A_i);
        //        double check_sum = psigmas.sum(); //// Should sum to 1.0
        auto lnGammai = get_Gamma(T, psigmas).log().eval();
        return A_i/AEFFPRIME*(psigmas*(lnGamma_mix - lnGammai)).sum();
    }
    
    /**
     This overload is a convenience overload, less computationally
     efficient, but simpler to use and more in keeping with the other
     contributions.  It does not require the lnGamma_mix to be pre-calculated
     and passed into this function
     */
    
    template<typename TType, typename MoleFracs>
    auto get_lngamma_resid(TType T, const MoleFracs& molefracs) const
    {
        using TXType = std::decay_t<std::common_type_t<TType, decltype(molefracs[0])>>;
        
        Eigen::Array<TXType, 153, 1> psigmas;
        psigmas << get_psigma_mix(molefracs, profile_type::NHB_PROFILE), get_psigma_mix(molefracs, profile_type::OH_PROFILE), get_psigma_mix(molefracs, profile_type::OT_PROFILE);
        
        Eigen::ArrayX<TXType> lngamma(molefracs.size());
        //        double check_sum = psigmas.sum(); //// Should sum to 1.0
        Eigen::ArrayX<TXType> lnGamma_mix = get_Gamma(T, psigmas).log();
        for (Eigen::Index i = 0; i < molefracs.size(); ++i) {
            lngamma(i) = get_lngamma_resid(i, T, lnGamma_mix);
        }
        return lngamma;
    }
    template<typename TType, typename MoleFracs>
    auto calc_lngamma_resid(TType T, const MoleFracs& molefracs) const
    {
        return get_lngamma_resid(T, molefracs);
    }
    
    /**
    The combinatorial part of ln(γ_i)
    */
    template<typename TType, typename MoleFractions>
    auto calc_lngamma_comb(const TType& /*T*/, const MoleFractions &x) const {
        double q0 = m_comb_consts.q0,
               r0 = m_comb_consts.r0,
               z_coordination = m_comb_consts.z_coordination;
        std::decay_t<MoleFractions> q = Eigen::Map<const Eigen::ArrayXd>(&(A_COSMO_A2[0]), A_COSMO_A2.size()) / q0,
            r = Eigen::Map<const Eigen::ArrayXd>(&(V_COSMO_A3[0]), V_COSMO_A3.size()) / r0,
            l = z_coordination / 2 * (r - q) - (r - 1),
            phi_over_x = r / contiguous_dotproduct(x, r),
            phi = x * phi_over_x,
            theta_over_x = q / contiguous_dotproduct(x, q),
            theta = x * theta_over_x,
            theta_over_phi = theta_over_x/phi_over_x;
            
        // Eq. 15 from Bell, JCTC, 2020
        return ((log(phi_over_x) + z_coordination / 2 * q * log(theta_over_phi) + l - phi_over_x * contiguous_dotproduct(x, l))).eval();
    }
    
    //    EigenArray get_lngamma_disp(const EigenArray &x) const {
    //        if (x.size() != 2){ throw std::invalid_argument("Multi-component mixtures not supported for dispersive contribution yet"); }
    //        double w = 0.27027; // default value
    //        auto cls0 = m_fluids[0].dispersion_flag, cls1 = m_fluids[1].dispersion_flag;
    //        using d = FluidProfiles::dispersion_classes;
    //        if ((cls0 == d::DISP_WATER && cls1 == d::DISP_ONLY_ACCEPTOR)
    //            |
    //            (cls1 == d::DISP_WATER && cls0 == d::DISP_ONLY_ACCEPTOR)){
    //            w = -0.27027;
    //        }
    //        else if ((cls0 == d::DISP_WATER && cls1 == d::DISP_COOH)
    //            |
    //            (cls1 == d::DISP_WATER && cls0 == d::DISP_COOH)) {
    //            w = -0.27027;
    //        }
    //        else if ((cls0 == d::DISP_COOH && (cls1 == d::DISP_NHB || cls1 == d::DISP_DONOR_ACCEPTOR))
    //            |
    //            (cls1 == d::DISP_COOH && (cls0 == d::DISP_NHB || cls0 == d::DISP_DONOR_ACCEPTOR))) {
    //            w = -0.27027;
    //        }
    //
    //        double ekB0 = m_fluids[0].dispersion_eoverkB, ekB1 = m_fluids[1].dispersion_eoverkB;
    //        double A = w*(0.5*(ekB0+ekB1) - sqrt(ekB0*ekB1));
    //        EigenArray lngamma_dsp(2);
    //        lngamma_dsp(0) = A*x[1]*x[1];
    //        lngamma_dsp(1) = A*x[0]*x[0];
    //        return lngamma_dsp;
    //    }
    //    EigenArray get_lngamma(double T, const EigenArray &x) const override {
    //        return get_lngamma_comb(T, x) + get_lngamma_resid(T, x) +  get_lngamma_disp(x);
    //    }
    
    // Returns ares/RT
    template<typename TType, typename MoleFractions>
    auto operator () (const TType& T, const MoleFractions& molefracs) const {
        return contiguous_dotproduct(molefracs, get_lngamma_resid(T, molefracs));
    }
};

};
