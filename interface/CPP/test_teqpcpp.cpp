/**
* This is a minimal example of the use of the C++ interface around teqp
* 
* The point of the interface is to reduce compilation time, as compilation 
* of this file should be MUCH faster than compilation with the full set of 
* algorithms and models because the template instantiations are all included in the 
* library that this file is linked against and the compiler does not need to resolve
* all the templates every time this file is compiled
*/


#include "teqpcpp.hpp"

int main() {
    teqp::AllowedModels model = teqp::build_multifluid_model({ "Methane", "Ethane" }, "../mycp");
    auto z = (Eigen::ArrayXd(2) << 0.5, 0.5).finished();
    double Ar01 = teqp::cppinterface::get_Arxy(model, 0, 1, 300, 3, z);
}