#pragma once

namespace teqp{
namespace VLLE{

enum class VLLE_return_code { unset, xtol_satisfied, functol_satisfied, maxiter_met };

struct SelfIntersectionSolution {
    std::size_t
    j, ///< The index on one side of the intersection, j and j+1 bracket the intersection, s is the fraction between j and j+1
    k; ///< The index on one side of the intersection, k and k+1 bracket the intersection, t is the fraction between k and k+1
    double s, ///< The fraction of the way between j and j+1
    t, ///< The fraction of the way between k and k+1
    x, ///< The x coordinate of the estimated intersection
    y; ///< The y coordinate of the estimated intersection
};

struct VLLEFinderOptions {
    int max_steps = 20; ///< The maximum number of steps allowed in polisher
    double rho_trivial_threshold = 1e-16; ///< The relative difference between densities of liquid solutions that indicates a non-trivial solution has been found
};

}
}
