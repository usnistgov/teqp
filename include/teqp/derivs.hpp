#pragma once

template <typename TType, typename ContainerType, typename FuncType>
typename std::enable_if<is_container<ContainerType>::value, typename ContainerType::value_type>::type
caller(const FuncType& f, TType T, const ContainerType& rho) {
	return f(T, rho);
}

template <typename TType, typename ContainerType, typename FuncType>
typename std::enable_if<is_container<ContainerType>::value, typename ContainerType::value_type>::type
deriv1(const FuncType& f, TType T, const ContainerType& rho) {
	double h = 1e-100;
	return f(std::complex<TType>(T, h), rho).imag() / h;
}

template <typename TType, typename ContainerType, typename FuncType>
typename std::enable_if<is_container<ContainerType>::value, typename ContainerType::value_type>::type
deriv2(const FuncType& f, TType T, const ContainerType& rho) {
	double h = 1e-100;
	using comtype = std::complex<ContainerType::value_type>;
	std::valarray<comtype> rhocom(rho.size());
	rhocom[0] = comtype(rho[0], h);
	for (auto i = 1; i < rho.size(); ++i) {
		rhocom[i] = comtype(rho[i], 0.0);
	}
	return f(T, rhocom).imag() / h;
}

template <typename TType, typename ContainerType, typename FuncType>
typename std::enable_if<is_container<ContainerType>::value, typename ContainerType::value_type>::type
deriv3(const FuncType& f, TType T, const ContainerType& rho) {
	double h = 1e-100;
	using comtype = std::complex<ContainerType::value_type>;
	std::valarray<comtype> rhocom(rho.size());
	rhocom[0] = comtype(rho[0], 0.0);
	rhocom[1] = comtype(rho[1], h);
	return f(T, rhocom).imag() / h;
}

template<typename Model, typename TType, typename RhoType>
auto build_Psir_Hessian(const Model& model, const TType T, const RhoType& rho) {
	// Double derivatives in each component's concentration

}