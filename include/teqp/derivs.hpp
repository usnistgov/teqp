#pragma once

template <typename TType, typename ContainerType, typename FuncType>
typename std::enable_if<is_container<ContainerType>::value, typename ContainerType::value_type>::type
caller(const FuncType& f, TType T, const ContainerType& rho) {
    return f(T, rho);
}

/// Given a function, use complex step derivatives to calculate the derivative with respect to the first variable
/// which here is temperature
template <typename TType, typename ContainerType, typename FuncType>
typename std::enable_if<is_container<ContainerType>::value, typename ContainerType::value_type>::type
derivT(const FuncType& f, TType T, const ContainerType& rho) {
    double h = 1e-100;
    return f(std::complex<TType>(T, h), rho).imag() / h;
}

/// Given a function, use complex step derivatives to calculate the derivative with respect to the given composition variable
template <typename TType, typename ContainerType, typename FuncType, typename Integer>
typename std::enable_if<is_container<ContainerType>::value, typename ContainerType::value_type>::type
derivrhoi(const FuncType& f, TType T, const ContainerType& rho, Integer i) {
    double h = 1e-100;
    using comtype = std::complex<ContainerType::value_type>;
    std::valarray<comtype> rhocom(rho.size());
    for (auto j = 0; j < rho.size(); ++j) {
        rhocom[j] = comtype(rho[j], 0.0);
    }
    rhocom[i] = comtype(rho[i], h);
    return f(T, rhocom).imag() / h;
}

template<typename Model, typename TType, typename RhoType>
auto build_Psir_Hessian(const Model& model, const TType T, const RhoType& rho) {
    // Double derivatives in each component's concentration

}