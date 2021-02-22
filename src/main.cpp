#include <iostream>
#include <algorithm>
#include <numeric>
#include <vector>
#include <valarray>
#include <functional>
#include <complex>
#include <chrono>
#include <optional>

#include "teqp/core.hpp"

void test_vdW() {
	volatile double T = 298.15;
	auto rho = 3.0;
	auto R = get_R_gas<double>();

	double Omega_b = 1.0 / 8, Omega_a = 27.0 / 64;
	double Tcrit = 150.687, pcrit = 4863000.0; // Argon
	double b = Omega_b * R * Tcrit / pcrit;
	double ba = Omega_b / Omega_a / Tcrit / R;
	double a = b / ba;

	auto vdW = vdWEOS1(a, b);

	auto t2 = std::chrono::steady_clock::now();
	volatile auto pp = vdW.p(T, 1 / rho);
	auto t3 = std::chrono::steady_clock::now();
	std::cout << std::chrono::duration<double>(t3 - t2).count() << " from p(T,v)" << std::endl;

	const std::valarray<double> rhovec = { rho, 0.0 };

	auto t21 = std::chrono::steady_clock::now();
	
	auto Psir = vdW.Psir(T, rhovec);
	auto dPsirdrho0 = rhovec[0] * deriv2([&vdW](const auto& T, const auto& rhovec) { return vdW.Psir(T, rhovec); }, T, rhovec);
	auto dPsirdrho1 = rhovec[1] * deriv3([&vdW](const auto& T, const auto& rhovec) { return vdW.Psir(T, rhovec); }, T, rhovec);
	auto pfromderiv = rho * R * T - Psir + dPsirdrho0 + dPsirdrho1;

	auto t31 = std::chrono::steady_clock::now();
	std::cout << std::chrono::duration<double>(t31 - t21).count() << " from isochoric" << std::endl;
	std::cout << pfromderiv / pp - 1 << std::endl;
}

void test_vdwMix() {
	// Argon + Xenon
	std::valarray<double> Tc_K = { 150.687, 289.733 };
	std::valarray<double> pc_Pa = { 4863000.0, 5842000.0 };
	vdWEOS<double> vdW(Tc_K, pc_Pa);

	double T = 298.15;
	auto rho = 3.0;
	auto R = get_R_gas<double>();
	auto rhotot = rho;

	const std::valarray<double> rhovec = { rho, 0.0 };

	auto t2 = std::chrono::steady_clock::now();

	auto Psir = vdW.Psir(T, rhovec);
	auto dPsirdrho0 = rhovec[0] * deriv2([&vdW, rhotot](const auto& T, const auto& rhovec) { return vdW.Psir(T, rhovec); }, T, rhovec);
	auto dPsirdrho1 = rhovec[1] * deriv3([&vdW, rhotot](const auto& T, const auto& rhovec) { return vdW.Psir(T, rhovec); }, T, rhovec);
	auto pfromderiv = rho * R * T - Psir + dPsirdrho0 + dPsirdrho1;

	auto t3 = std::chrono::steady_clock::now();
	std::cout << std::chrono::duration<double>(t3 - t2).count() << " from isochoric (mix) " << std::endl;
}

int main(){
	test_vdW();
	test_vdwMix();
	return EXIT_SUCCESS;
}