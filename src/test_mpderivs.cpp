// Imports from boost
#include <boost/multiprecision/cpp_bin_float.hpp>
using namespace boost::multiprecision; 

#include "teqp/models/cubics.hpp"
#include "teqp/algorithms/VLE.hpp"
#include "teqp/derivs.hpp"

int main() {

    // Values taken from http://dx.doi.org/10.6028/jres.121.011
    std::valarray<double> Tc_K = { 154.581}, pc_Pa = { 5042800}, acentric = { 0.022};
    auto modelPR = teqp::canonical_PR(Tc_K, pc_Pa, acentric);

    using my_float = boost::multiprecision::number<boost::multiprecision::cpp_bin_float<200>>; // Overkill: 200 digits of working precision!
    
    // Get the already very accurate values from the superancillary equation
    auto [rhoLdbl, rhoVdbl] = modelPR.superanc_rhoLV(125);
    
    // Now iterate in extended precision with the multicomplex backend to find the VLE solution
    my_float T = 125, rhoL = rhoLdbl, rhoV = rhoVdbl;
    
    teqp::IsothermPureVLEResiduals<decltype(modelPR), my_float, teqp::ADBackends::multicomplex> residual(modelPR, T);
    auto soln = teqp::do_pure_VLE_T<decltype(residual), my_float>(residual, rhoL, rhoV, 10).cast<double>();
    
    std::cout << soln << std::endl;
}
