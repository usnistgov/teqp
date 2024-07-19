#pragma once

namespace teqp::saft::PCSAFT{

/**
 The model of Gross & Sadowski, simplified down to the case of pure fluids
 */
class PCSAFTPureGrossSadowski2001{
private:
    Eigen::Array<double, 7, 1> aim, bim;
public:
    const double pi = 3.141592653589793238462643383279502884197;
    const Eigen::Array<double, 7, 6> coeff;
    const double m, sigma_A, eps_k;
    double kappa1, kappa2;
    PCSAFTPureGrossSadowski2001(const nlohmann::json&j) : coeff((Eigen::Array<double, 7, 6>() << 0.9105631445,-0.3084016918,-0.0906148351,0.7240946941,-0.5755498075,0.0976883116  ,
                                                                 0.6361281449,0.1860531159,0.4527842806,2.2382791861,0.6995095521,-0.2557574982    ,
                                                                 2.6861347891,-2.5030047259,0.5962700728,-4.0025849485,3.8925673390,-9.1558561530  ,
                                                                 -26.547362491,21.419793629,-1.7241829131,-21.003576815,-17.215471648,20.642075974 ,
                                                                 97.759208784,-65.255885330,-4.1302112531,26.855641363,192.67226447,-38.804430052  ,
                                                                 -159.59154087,83.318680481,13.776631870,206.55133841,-161.82646165,93.626774077   ,
                                                                 91.297774084,-33.746922930,-8.6728470368,-355.60235612,-165.20769346,-29.666905585).finished()),
    m(j.at("m")), sigma_A(j.at("sigma / A")), eps_k(j.at("epsilon_over_k")) {
        auto mfac1 = (m-1.0)/m;
        auto mfac2 = (m-2.0)/m*mfac1;
        aim = coeff.col(0) + coeff.col(1)*mfac1 + coeff.col(2)*mfac2; // Can do this because m is not depending on composition because a pure fluid
        bim = coeff.col(3) + coeff.col(4)*mfac1 + coeff.col(5)*mfac2; // Can do this because m is not depending on composition because a pure fluid
        kappa1 = (2.0*pi*eps_k*pow(m, 2)*pow(sigma_A, 3));
        kappa2 = (pi*pow(eps_k, 2)*pow(m, 3)*pow(sigma_A, 3));
    }
    
    template<class VecType>
    auto R(const VecType& molefrac) const {
        return get_R_gas<decltype(molefrac[0])>();
    }
    
    template<typename TTYPE, typename RhoType, typename VecType>
    auto alphar(const TTYPE& T, const RhoType& rhomolar, const VecType& /*mole_fractions*/) const {
        
        auto rhoN_A3 = forceeval(rhomolar*N_A/1e30); // [A^3]
        
        auto d = forceeval(sigma_A*(1.0-0.12*exp(-3.0*eps_k/T)));
        Eigen::Array<decltype(d), 4, 1> dpowers; dpowers(0) = 1.0; for (auto i = 1U; i <= 3; ++i){ dpowers(i) = d*dpowers(i-1); }
        auto D = pi/6.0*m*dpowers;
        auto zeta = rhoN_A3*D.template cast<std::common_type_t<TTYPE, RhoType>>();
        
        auto zeta2_to2 = zeta[2]*zeta[2];
        auto zeta2_to3 = zeta2_to2*zeta[2];
        auto zeta3_to2 = zeta[3]*zeta[3];
        auto onemineta = forceeval(1.0-zeta[3]);
        auto onemineta_to2 = onemineta*onemineta;
        auto onemineta_to3 = onemineta*onemineta_to2;
        auto onemineta_to4 = onemineta*onemineta_to3;
        
        std::decay_t<decltype(zeta[0])> alpha_hs = forceeval((3.0*zeta[1]*zeta[2]/onemineta
                         + zeta2_to3/(zeta[3]*onemineta_to2)
                         + (zeta2_to3/zeta3_to2-zeta[0])*log(1.0-zeta[3]))/zeta[0]);
        if (getbaseval(zeta[0]) == 0){
            auto Upsilon = 1.0-zeta[3];
            alpha_hs = forceeval(
                3.0*D[1]/D[0]*zeta[2]/Upsilon
                + D[2]*D[2]*zeta[2]/(D[3]*D[0]*Upsilon*Upsilon)
                - log(Upsilon)
                + (D[2]*D[2]*D[2])/(D[3]*D[3]*D[0])*log(Upsilon)
            );
        }
        
        auto fac_g_hs = d/2.0; // d*d/(2*d)
        auto gii = (1.0/onemineta
                    + fac_g_hs*3.0*zeta[2]/onemineta_to2
                    + (fac_g_hs*fac_g_hs)*2.0*zeta2_to2/onemineta_to3);
        auto alpha_hc = m*alpha_hs - (m-1)*log(gii);
        
        auto eta = zeta[3];
        auto eta2 = eta*eta;
        auto eta3 = eta2*eta;
        auto eta4 = eta2*eta2;
        auto C1 = 1.0+m*(8.0*eta-2.0*eta2)/onemineta_to4+(1.0-m)*(20.0*eta-27.0*eta2+12.0*eta3-2.0*eta4)/onemineta_to2/((2.0-eta)*(2.0-eta));
        
        Eigen::Array<decltype(eta), 7, 1> etapowers; etapowers(0) = 1.0; for (auto i = 1U; i <= 6; ++i){ etapowers(i) = eta*etapowers(i-1); }
        auto I1 = (aim.array().template cast<decltype(eta)>()*etapowers).sum();
        auto I2 = (bim.array().template cast<decltype(eta)>()*etapowers).sum();
        
        auto alpha_disp = -kappa1*rhoN_A3*I1/T - kappa2*rhoN_A3*I2/C1/(T*T);
        
        return forceeval(alpha_hc + alpha_disp);
    }
};

};
