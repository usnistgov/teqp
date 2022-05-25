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

    using my_float = boost::multiprecision::number<boost::multiprecision::cpp_bin_float<200>>;
    auto [rhoLdbl, rhoVdbl] = modelPR.superanc_rhoLV(125);
    my_float T = 125;
    auto soln = teqp::pure_VLE_T<decltype(modelPR), my_float, teqp::ADBackends::multicomplex>(modelPR, T, static_cast<my_float>(rhoLdbl), static_cast<my_float>(rhoVdbl), 20).cast<double>();
    std::cout << soln << std::endl;

    my_float x = 10567.6789;
    std::function<mcx::MultiComplex<my_float>(const mcx::MultiComplex<my_float>&)> ff = [](const auto & x) {return 1.0 - x;};
    auto o = mcx::diff_mcx1<my_float>(ff, x, 3, true);
    std::cout << o[1] << std::endl; 

    std::cout << std::is_arithmetic_v<my_float> << std::endl;
}