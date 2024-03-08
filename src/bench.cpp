#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark_all.hpp>

#include "teqp/models/vdW.hpp"
#include "teqp/models/pcsaft.hpp"
#include "teqp/models/cubics.hpp"

#include "teqp/derivs.hpp"

using namespace teqp;

TEST_CASE("vdW derivatives", "[vdW]")
{
	double Omega_b = 1.0 / 8, Omega_a = 27.0 / 64;
    double Tcrit = 150.687, pcrit = 4863000.0; // Argon
    double R = get_R_gas<double>(); // Universal gas constant
    double b = Omega_b * R * Tcrit / pcrit;
    double ba = Omega_b / Omega_a / Tcrit / R;
    double a = b / ba;
    auto model = vdWEOS1(a, b);
    double T = 300, rho = 2;
    std::valarray<double> z(2, 1.0);
    using tdx = TDXDerivatives<decltype(model), double, decltype(z)>;

	BENCHMARK("alphar") {
		return model.alphar(T, rho, z);
	};
    BENCHMARK("alphar via get_Ar00") {
        return tdx::get_Ar00(model, T, rho, z);
    };
    BENCHMARK("rho*dalphar/drho w/ autodiff") {
        return tdx::get_Ar01(model, T, rho, z);
    };
    BENCHMARK("rho*dalphar/drho w/ multicomplex") {
        return tdx::get_Ar01<ADBackends::multicomplex>(model, T, rho, z);
    };
    BENCHMARK("rho*dalphar/drho w/ complex step") {
        return tdx::get_Ar01<ADBackends::complex_step>(model, T, rho, z);
    };
    BENCHMARK("rho^2*d^2alphar/drho^2 w/ autodiff") {
        return tdx::get_Ar02(model, T, rho, z);
    };
    BENCHMARK("rho^2*d^2alphar/drho^2 w/ multicomplex") {
        return tdx::get_Ar02<ADBackends::multicomplex>(model, T, rho, z);
    };
    BENCHMARK("(1/T)*dalphar/d(1/T) w/ autodiff") {
        return tdx::get_Ar10(model, T, rho, z);
    };
    BENCHMARK("(1/T)*dalphar/d(1/T) w/ mcx") {
        return tdx::get_Ar10<ADBackends::multicomplex>(model, T, rho, z);
    };
}

TEST_CASE("PCSAFT derivatives", "[PCSAFT]")
{
    using namespace PCSAFT;
    std::vector<std::string> names = { "Methane", "Ethane" };
    auto model = PCSAFTMixture(names);

    double T = 300, rho = 2;
    Eigen::ArrayX<double> z(2); z.fill(1.0/z.size());
    using tdx = TDXDerivatives<decltype(model), double, decltype(z)>;

    BENCHMARK("alphar") {
        return model.alphar(T, rho, z);
    };
    BENCHMARK("alphar via get_Ar00") {
        return tdx::get_Ar00(model, T, rho, z);
    };
    BENCHMARK("rho*dalphar/drho w/ autodiff") {
        return tdx::get_Ar01(model, T, rho, z);
    };
    BENCHMARK("rho*dalphar/drho w/ multicomplex") {
        return tdx::get_Ar01<ADBackends::multicomplex>(model, T, rho, z);
    };
    BENCHMARK("rho*dalphar/drho w/ complex step") {
        return tdx::get_Ar01<ADBackends::complex_step>(model, T, rho, z);
    };
    BENCHMARK("rho^2*d^2alphar/drho^2 w/ autodiff") {
        return tdx::get_Ar02(model, T, rho, z);
    };
    BENCHMARK("rho^2*d^2alphar/drho^2 w/ multicomplex") {
        return tdx::get_Ar02<ADBackends::multicomplex>(model, T, rho, z);
    };
    BENCHMARK("(1/T)*dalphar/d(1/T) w/ autodiff") {
        return tdx::get_Ar10(model, T, rho, z);
    };
    /*BENCHMARK("(1/T)*dalphar/d(1/T) w/ mcx") {
        return tdx::get_Ar10<ADBackends::multicomplex>(model, T, rho, z);
    };*/
}


TEST_CASE("PCSAFT more derivatives", "[PCSAFT]")
{
    using namespace PCSAFT;
    std::vector<std::string> names = { "Methane", "Ethane", "Propane" };
    auto model = PCSAFTMixture(names);

    double T = 300, rho = 2;
    Eigen::ArrayX<double> z(3); z.fill(1.0/3.0);
    using tdx = TDXDerivatives<decltype(model), double, decltype(z)>;
    Eigen::ArrayXd rhovec = rho*z;

    BENCHMARK("alphar") {
        return model.alphar(T, rho, z);
    };
    BENCHMARK("fugacity_coefficients w/ autodiff") {
        return -IsochoricDerivatives<decltype(model)>::get_fugacity_coefficients(model, T, rhovec);
    };
    BENCHMARK("compressibility w/ autodiff") {
        return 1.0+tdx::get_Ar01(model, T, rho, z);
    };
    BENCHMARK("c_vr/R w/ autodiff") {
        return -tdx::get_Ar20(model, T, rho, z);
    };
    BENCHMARK("partial_molar_volumes w/ autodiff") {
        return -IsochoricDerivatives<decltype(model)>::get_partial_molar_volumes(model, T, rhovec);
    };
}




TEST_CASE("Canonical cubic EOS derivatives", "[cubic]")
{
    // Values taken from http://dx.doi.org/10.6028/jres.121.011
    std::valarray<double> Tc_K = { 190.564, 154.581, 150.687 },
        pc_Pa = { 4599200, 5042800, 4863000 },
        acentric = { 0.011, 0.022, -0.002 };
    auto model = canonical_PR(Tc_K, pc_Pa, acentric);

    double T = 300, rho = 2;
    std::valarray<double> z = { 0.5, 0.3, 0.2 };
    using tdx = TDXDerivatives<decltype(model), double, decltype(z)>;

    BENCHMARK("alphar") {
        return model.alphar(T, rho, z);
    };
    BENCHMARK("alphar via get_Ar00") {
        return tdx::get_Ar00(model, T, rho, z);
    };
    BENCHMARK("rho*dalphar/drho w/ autodiff") {
        return tdx::get_Ar01(model, T, rho, z);
    };
    /*BENCHMARK("rho*dalphar/drho w/ multicomplex") {
        return tdx::get_Ar01<ADBackends::multicomplex>(model, T, rho, z);
    };*/
    /*BENCHMARK("rho*dalphar/drho w/ complex step") {
        return tdx::get_Ar01<ADBackends::complex_step>(model, T, rho, z);
    };*/
    BENCHMARK("rho^2*d^2alphar/drho^2 w/ autodiff") {
        return tdx::get_Ar02(model, T, rho, z);
    };
    /*BENCHMARK("rho^2*d^2alphar/drho^2 w/ multicomplex") {
        return tdx::get_Ar02<ADBackends::multicomplex>(model, T, rho, z);
    };*/
    BENCHMARK("(1/T)*dalphar/d(1/T) w/ autodiff") {
        return tdx::get_Ar10(model, T, rho, z);
    };
    /*BENCHMARK("(1/T)*dalphar/d(1/T) w/ mcx") {
        return tdx::get_Ar10<ADBackends::multicomplex>(model, T, rho, z);
    };*/
}
