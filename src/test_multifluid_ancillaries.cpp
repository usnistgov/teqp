
#include "teqp/models/multifluid_ancillaries.hpp"
#include "teqp/models/multifluid.hpp"
#include "teqp/algorithms/VLE.hpp"

int main() {
	auto model = teqp::build_multifluid_model({"n-Propane"}, "../mycp");
	auto j = nlohmann::json::parse(model.get_meta());
	auto jancillaries = j.at("pures")[0].at("ANCILLARIES");
	teqp::MultiFluidVLEAncillaries anc(jancillaries);
	double T = 340;
	auto rhoV = anc.rhoV(T), rhoL = anc.rhoL(T);
	auto rhovec = teqp::pure_VLE_T(model, T, rhoL, rhoV, 10);
	//auto [rhoLnew, rhoVnew] = rhovec;
	int rr = 0;
}