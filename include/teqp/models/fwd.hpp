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

namespace teqp {

	using vad = std::valarray<double>;

	// Define the EOS types by interrogating the types returned by the respective factory function
	using cub = decltype(canonical_PR(vad{}, vad{}, vad{}));
	using cpatype = decltype(CPA::CPAfactory(nlohmann::json{}));
	using pcsafttype = decltype(PCSAFT::PCSAFTfactory(nlohmann::json{}));
	using multifluidtype = decltype(multifluidfactory(nlohmann::json{}));

	using AllowedModels = std::variant<vdWEOS1, cub, cpatype, pcsafttype, multifluidtype>;
}