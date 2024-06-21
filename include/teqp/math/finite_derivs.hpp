#pragma once

#include <valarray>
#include <map>

namespace teqp{

/**
* Routines for finite differentiation, useful for testing derivatives obtained by other methods
*
* From:
* Bengt Fornberg, 1988, "Coefficients from Generation of Finite Difference Formulas on Arbitrarily Spaced Grids", MATHEMATICS OF COMPUTATION, v. 51, n. 184, pp. 699-706
*
* Higher derivatives should always be done in extended precision mathematics!
*
* Warning: these routines may give entirely erroneous results for double precision arithmetic,
* especially the higher derivatives
*
* Warning: these routines are optimized for accuracy, not for speed or memory use
*/
template<int Nderiv, int Norder, typename Function, typename Scalar>
auto centered_diff(const Function &f, const Scalar x, const Scalar h) {

	struct DiffCoeffs {
		std::valarray<int> k;
		std::valarray<Scalar> c;
	};
	
	// See https://en.wikipedia.org/wiki/Finite_difference_coefficient#Central_finite_difference
	// Watch out that if you would like to use extended precision, you also need to keep 
	// all the coefficients in extended precision too
	//
	// This is because 1.0/2.0 is different than casting each of 1.0 and 2.0 to extended precision
	// and then taking their ratio
	using r = Scalar;
	static std::map<std::tuple<int, int>, DiffCoeffs> CentralDiffCoeffs = {
		{{1, 2}, {{-1,1},      {-r(1)/r(2), r(1)/r(2)}} },
		{{1, 4}, {{-2,-1,1,2}, {r(1)/ r(12), -r(2)/r(3), r(2)/r(3), -r(1)/r(12)}} },
		{{1, 6}, {{-3,-2,-1,1,2,3}, {-r(1)/r(60), r(3)/r(20), -r(3)/r(4), r(3)/r(4), -r(3)/r(20), r(1)/r(60)}} },

		{{2, 2}, {{-1,0,1}, {1, -2, 1} }},
		{{2, 4}, {{-2,-1,0,1,2}, {r(-1)/r(12), r(4)/r(3), r(-5)/r(2), r(4)/r(3), r(-1)/r(12)}} },
		{{2, 6}, {{-3,-2,-1,0,1,2,3}, {r(1)/r(90), r(-3)/r(20), r(3)/r(2), r(-49)/r(18), r(3)/r(2), r(-3)/r(20), r(1)/r(90)}} },
		
		{{3, 2}, {{-2, -1, 0, 1, 2}, {r(-1)/r(2), r(1), 0, r(-1), r(1)/r(2)}} },
		{{3, 4}, {{-3,-2,-1,0,1,2,3}, {r(1)/r(8), r(-1), r(13)/r(8), 0, r(-13)/r(8), r(1), r(-1)/r(8)}} },
		{{3, 6}, {{-4,-3,-2,-1,0,1,2,3,4}, {r(-7)/r(240), r(3)/r(10), r(-169)/r(120), r(61)/r(30), 0, r(-61)/r(30), r(169)/r(120), r(-3)/r(10), r(7)/r(240)}} },

		{{4, 2}, {{-2,-1,0,1,2}, {1,-4,6,-4,1}} },
		{{4, 4}, {{-3,-2,-1,0,1,2,3}, {-r(1)/r(6), r(2), -r(13)/r(2), r(28)/r(3.0), -r(13)/r(2), r(2), -r(1)/r(6)}} },
		{{4, 6}, {{-4,-3,-2,-1,0,1,2,3,4}, {r(7)/r(240), -r(2)/r(5), r(169)/r(60), -r(122)/r(15), r(91)/r(8), -r(122)/r(15), r(169)/r(60), -r(2)/r(5), r(7)/r(240)}} },
	};

	auto [k, c] = CentralDiffCoeffs[std::make_tuple(Nderiv, Norder)];
	if (c.size() == 0) {
		throw std::invalid_argument("Cannot obtain the necessary finite differentiation coefficients");
	}
	// Sanity check...
	if (c.size() != k.size()) {
		throw std::invalid_argument("Finite differentiation coefficient arrays not the same size");
	}
	Scalar num = 0.0;
	for (auto i = 0U; i < k.size(); ++i) {
		num = num + c[i]*f(x + h*k[i]);
	}
	auto val = num / pow(h, Nderiv);
	return val;
}

template<typename Function, typename Scalarx, typename Scalary>
auto centered_diff_xy(const Function &f, const Scalarx x, const Scalary y, const Scalarx dx, const Scalary dy) {
    return (f(x+dx, y+dy) - f(x+dx, y-dy) - f(x-dx, y+dy) + f(x-dx, y-dy))/(4*dx*dy);
}

template<typename Function, typename Vec, typename Scalar>
auto gradient_forward(const Function &f, const Vec& x, Scalar h) {
    Vec out = 0.0*x, xplus = x;
    for (auto i = 0; i < out.size(); ++i){
        xplus = x;
        xplus[i] += h;
        out[i] = (f(xplus)-f(x))/h;
    }
    return out;
}
    
}; // namespace teqp
