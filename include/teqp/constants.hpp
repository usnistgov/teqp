#pragma once

const double N_A = 6.02214076e23; ///< Avogadro's number

///< Gas constant, according to CODATA 2019, in the given number type
template<typename NumType>
const auto get_R_gas() {
	const double k_B = 1.380649e-23; ///< Boltzmann constant
	const double N_A = 6.02214076e23; ///< Avogadro's number
	return forceeval(static_cast<NumType>(N_A*k_B));
};