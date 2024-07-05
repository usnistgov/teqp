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
     */
    auto test_virial(auto n, auto T, auto rhotest = 1e-8, auto reltol = 1e-6){
        auto B_n = model->get_dmBnvirdTm(n, 0, T, molefrac);
        CAPTURE(n);
        REQUIRE(std::isfinite(B_n));
        if (n == 2){
            // B
            auto B_n_nondilute = model->get_Ar01(T, rhotest, molefrac)/rhotest;
            CHECK_THAT(B_n, WithinRel(B_n_nondilute, reltol));
        }
        if (n == 3){
            // C
            auto B_n_nondilute = model->get_Ar02(T, rhotest, molefrac)/rhotest/rhotest;
            CHECK_THAT(B_n, WithinRel(B_n_nondilute, reltol));
        }
        if (n == 4){
            // D
            auto B_n_nondilute = model->get_Ar03(T, rhotest, molefrac)/(rhotest/rhotest/rhotest)/2.0;
            CHECK_THAT(B_n, WithinRel(B_n_nondilute, reltol));
        }
    }
};
