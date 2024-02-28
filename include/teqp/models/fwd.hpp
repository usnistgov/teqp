#pragma once

/**
* The name of this file is currently a bit of a misnomer while
* we think about whether it is possible to forward-declare the models
* 
* It seems like perhaps that is not possible.  For now this header 
* just provides the main variant with the model definitions.
*/

#include <valarray>
#include <variant>

#include "teqp/models/vdW.hpp"
#include "teqp/models/cubics.hpp"
#include "teqp/models/CPA.hpp"
#include "teqp/models/pcsaft.hpp"
#include "teqp/models/saftvrmie.hpp"
#include "teqp/models/multifluid.hpp"
#include "teqp/models/multifluid_mutant.hpp"
#include "teqp/models/ECSHuberEly.hpp"
#include "teqp/ideal_eosterms.hpp"
#include "teqp/models/ammonia_water.hpp"
#include "teqp/models/model_potentials/squarewell.hpp"
#include "teqp/models/model_potentials/exp6.hpp"
#include "teqp/models/model_potentials/2center_ljf.hpp"
#include "teqp/models/mie/lennardjones.hpp"
#include "teqp/models/mie/mie.hpp"
#include "teqp/models/GERG/GERG.hpp"
#include "teqp/models/LKP.hpp"

namespace teqp {

	using vad = std::valarray<double>;
    using vecs = std::vector<std::string>;

    // Define the EOS types by interrogating the types returned by the respective
    // factory function or by alias of the class name
    using canonical_cubic_t = decltype(canonical_PR(vad{}, vad{}, vad{}));
    using PCSAFT_t = decltype(PCSAFT::PCSAFTfactory(nlohmann::json{}));
    using SAFTVRMie_t = decltype(SAFTVRMie::SAFTVRMieMixture(vecs{}));
    using CPA_t = decltype(CPA::CPAfactory(nlohmann::json{}));
    using multifluid_t = decltype(multifluidfactory(nlohmann::json{}));
    using multifluidmutant_t = decltype(build_multifluid_mutant(multifluidfactory(nlohmann::json{}), nlohmann::json{}));
    using ammonia_water_TillnerRoth_t = AmmoniaWaterTillnerRoth;
    using SW_EspindolaHeredia2009_t = squarewell::EspindolaHeredia2009;
    using EXP6_Kataoka1992_t = exp6::Kataoka1992;
    using vdWEOS_t = vdWEOS<double>;
    using twocenterLJF_t = decltype(twocenterljf::build_two_center_model_dipole(std::string{}, double{}, double{}));
    using LJ126KolafaNezbeda1994_t = LJ126KolafaNezbeda1994;
    using LJ126Johnson1993_t = LJ126Johnson1993;
    using Mie6Pohl2023_t = Mie::Mie6Pohl2023;
    using QuantumPR_t = QuantumCorrectedPR;

    using idealgas_t = IdealHelmholtz;

	/*
     // The set of these models is exposed in the variant
	using AllowedModels = std::variant<
		vdWEOS1,
        vdWEOS_t,
        canonical_cubic_t,
        PCSAFT_t,
        SAFTVRMie_t,
        CPA_t,
        multifluid_t,
        multifluidmutant_t,
        idealgas_t,
        ammonia_water_TillnerRoth_t,
        SW_EspindolaHeredia2009_t,
        EXP6_Kataoka1992_t,
        twocenterLJF_t,
        LJ126KolafaNezbeda1994_t,
        LJ126Johnson1993_t,
        Mie6Pohl2023_t
	>;
     */
}
