#pragma once

namespace teqp::saft::polar_terms {

enum class multipolar_argument_spec {
    TK_rhoNA3_packingfraction_molefractions,
    TK_rhoNm3_rhostar_molefractions
};

enum class multipolar_rhostar_approach {
    kInvalid,
    use_packing_fraction,
    calculate_Gubbins_rhostar
};

// map multipolar_rhostar_approach values to JSON as strings
NLOHMANN_JSON_SERIALIZE_ENUM( multipolar_rhostar_approach, {
    {multipolar_rhostar_approach::kInvalid, nullptr},
    {multipolar_rhostar_approach::use_packing_fraction, "use_packing_fraction"},
    {multipolar_rhostar_approach::calculate_Gubbins_rhostar, "calculate_Gubbins_rhostar"},
})

template<typename type>
struct MultipolarContributionGubbinsTwuTermsGT{
    type alpha2;
    type alpha3;
    type alpha;
};

}
