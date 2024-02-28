#pragma once

namespace teqp {

namespace constants{

const double R_CODATA2002 = 8.314472; ///< molar gas constant from CODATA 2002: https://doi.org/10.1103/RevModPhys.77.1
const double R_CODATA2006 = 8.314472; ///< molar gas constant from CODATA 2006: https://doi.org/10.1103/RevModPhys.80.633
const double R_CODATA2010 = 8.3144621; ///< molar gas constant from CODATA 2010: https://doi.org/10.1103/RevModPhys.84.1527
const double R_CODATA2017 = 8.31446261815324; ///< molar gas constant from CODATA 2017: https://doi.org/10.1103/RevModPhys.84.1527

const double N_A = 6.02214076e23; ///< Avogadro's number
const double k_B = 1.380649e-23; ///< Boltzmann constant
const double epsilon_0 = 8.8541878128e-12; ///< Vacuum permittivity (https://en.wikipedia.org/wiki/Vacuum_permittivity), in F/m, or C^2⋅N^−1⋅m^−2
const double PI = 3.141592653589793238462643383279502884197;
const double k_e = 1.0/(4.0*PI*epsilon_0); ///< Coulomb constant, with units of N m^2 / C^2
}
    using constants::N_A; // Bring N_ into the teqp namespace

	///< Gas constant, according to CODATA 2019, in the given number type
	template<typename NumType>
	auto get_R_gas() {
		const NumType k_B = 1.380649e-23; ///< Boltzmann constant
		const NumType N_A_ = 6.02214076e23; ///< Avogadro's number
		return static_cast<NumType>(N_A_ * k_B);
	};

}; // namespace teqp
