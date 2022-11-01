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
#include "teqp/models/multifluid.hpp"
#include "teqp/models/multifluid_mutant.hpp"

namespace teqp {

	using vad = std::valarray<double>;

    // Define the EOS types by interrogating the types returned by the respective factory function
    using canonical_cubic_t = decltype(canonical_PR(vad{}, vad{}, vad{}));
    using PCSAFT_t = decltype(PCSAFT::PCSAFTfactory(nlohmann::json{}));
    using CPA_t = decltype(CPA::CPAfactory(nlohmann::json{}));
    using multifluid_t = decltype(multifluidfactory(nlohmann::json{}));
    //using multifluidmutant_t = decltype(build_multifluid_mutant(multifluid_t{}, nlohmann::json{})); // need to figure out how to get this to work

	// The set of these models is exposed in the variant
	using AllowedModels = std::variant<
		vdWEOS1,
        canonical_cubic_t,
        PCSAFT_t,
        CPA_t,
        multifluid_t
        //multifluidmutant_t
	>;
}
