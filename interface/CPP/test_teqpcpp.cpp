#define NO_TYPES_HEADER
#include "teqpcpp.hpp"

using namespace teqp::cppinterface;

int main(){
	teqp::AllowedModels model = teqp::build_multifluid_model({ "Methane","Ethane" }, "../mycp");

	//teqp::AllowedModels model = teqp::vdWEOS1(3, 0.1); 
	
	auto z = (Eigen::ArrayXd(2) << 0.5, 0.5).finished( ); 
	
	double Ar01 = get_Arxy(model, 0, 1, 300, 3, z); 

	double Tc1 = 371.1;
	auto cr = trace_critical_arclength_binary(model, Tc1, z);
}