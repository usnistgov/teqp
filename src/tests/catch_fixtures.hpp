#pragma one

#include <Eigen/Dense>

#include <catch2/matchers/catch_matchers_floating_point.hpp>
using Catch::Matchers::WithinRel;

template <typename Model>
class VirialTestFixture{
public:
    const Model& model;
    const Eigen::ArrayXd& molefrac;
    
    VirialTestFixture(const Model& model, const Eigen::ArrayXd& molefrac) : model(model), molefrac(molefrac) {};
    
    /**
    See Eq. 9 of https://doi.org/10.1021/acs.iecr.2c00237?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as
     
     \f[
     B^n = \frac{\Lambda^r_{0(n-1)}}{\rho^{n-1}(n-2)!}
     \f]
     
     \note The calculations of the non-dilute part have to be done in extended precision to have enough precision to calculate high-order virial coefficients
     */
    auto test_virial(auto n, auto T, auto rhotest, auto reltol = 1e-6){
        auto B_n = model->get_dmBnvirdTm(n, 0, T, molefrac);
        CAPTURE(n);
        CAPTURE(B_n);
        
        REQUIRE(std::isfinite(B_n));
        if (n == 2){
            // B
            auto B_n_nondilute_ep = model->get_Ar01ep(T, rhotest, molefrac)/rhotest; // and divided by (n-2)! or 0! = 1
            CAPTURE(B_n_nondilute_ep);
            CHECK_THAT(B_n, WithinRel(B_n_nondilute_ep, reltol));
        }
        if (n == 3){
            // C
            auto B_n_nondilute_ep = model->get_Ar02ep(T, rhotest, molefrac)/(rhotest*rhotest); // and divided by (n-2)! or 1! = 1
            CAPTURE(B_n_nondilute_ep);
            CHECK_THAT(B_n, WithinRel(B_n_nondilute_ep, reltol));
        }
        if (n == 4){
            // D
            auto B_n_nondilute_ep = model->get_Ar03ep(T, rhotest, molefrac)/(rhotest*rhotest*rhotest)/2.0; // and divided by (n-2)! or 2! = 2
            CAPTURE(B_n_nondilute_ep);
//            auto B_n_nondilute = model->get_Ar03(T, rhotest, molefrac)/(rhotest*rhotest*rhotest)/2.0; // and divided by (n-2)! or 2! = 2
//            CAPTURE(B_n_nondilute);
            if (std::abs(B_n_nondilute_ep) > 1e-12){
                CHECK_THAT(B_n, WithinRel(B_n_nondilute_ep, reltol));
            }
            else{
                CHECK_THAT(B_n, WithinAbs(B_n_nondilute_ep, reltol));
            }
        }
    }
};
