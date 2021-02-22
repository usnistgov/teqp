#pragma once

///< Gas constant, according to CODATA 2019, in the given number type
template<typename NumType>
const NumType get_R_gas() {
	constexpr NumType k_B = 1.380649e-23; ///< Boltzmann constant
	constexpr NumType N_A = 6.02214076e23; ///< Avogadro's number
	return N_A * k_B;
};

#include "teqp/types.hpp"
#include "teqp/derivs.hpp"
#include "teqp/eos.hpp"