#include "pybind11_wrapper.hpp"

#include "teqp/models/pcsaft.hpp"
#include "teqp/derivs.hpp"

using namespace PCSAFT;

void add_PCSAFT(py::module& m) {

	py::class_<SAFTCoeffs>(m, "SAFTCoeffs")
	.def(py::init<>())
	.def_readwrite("name", &SAFTCoeffs::name)
	.def_readwrite("m", &SAFTCoeffs::m)
	.def_readwrite("sigma_Angstrom", &SAFTCoeffs::sigma_Angstrom)
	.def_readwrite("epsilon_over_k", &SAFTCoeffs::epsilon_over_k)
	.def_readwrite("BibTeXKey", &SAFTCoeffs::BibTeXKey)
	;

	auto wPCSAFT = py::class_<PCSAFTMixture>(m, "PCSAFTEOS")
	.def(py::init<const std::vector<std::string>&, const Eigen::ArrayXXd&>(), py::arg("names"), py::arg_v("kmat", Eigen::ArrayXXd(0,0), "None"))
	.def(py::init<const std::vector<SAFTCoeffs>&>(), py::arg("coeffs"))
	.def("print_info", &PCSAFTMixture::print_info)
	.def("max_rhoN", &PCSAFTMixture::max_rhoN<Eigen::ArrayXd>)
	.def("get_m", &PCSAFTMixture::get_m)
	.def("get_sigma_Angstrom", &PCSAFTMixture::get_sigma_Angstrom)
	.def("get_epsilon_over_k_K", &PCSAFTMixture::get_epsilon_over_k_K)
	;
	add_derivatives<PCSAFTMixture>(m, wPCSAFT);
}