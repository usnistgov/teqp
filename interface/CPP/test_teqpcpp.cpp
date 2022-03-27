#include "teqpcpp.hpp"

using namespace teqp::cppinterface;

int main(){
	teqp::AllowedModels model = teqp::build_multifluid_model({ "Methane","Ethane" }, "../mycp");
	auto z = (Eigen::ArrayXd(2) << 0.5, 0.5).finished();
	
	double Ar01 = get_Arxy(model, 0, 1, 300, 3, z); 

	double Tc1 = 371;
	auto cr = trace_critical_arclength_binary(model, Tc1, z);
}