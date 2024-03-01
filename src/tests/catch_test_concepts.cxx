#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/benchmark/catch_benchmark_all.hpp>
using Catch::Approx;

#include <iostream>
#include <concepts>

template<typename T>
concept UsesTauDelta = requires(T t) {
    { t.has_alphar_taudelta } -> std::same_as<std::true_type&>;
};

template<typename T>
concept DoesntUseTauDelta = requires(T t) {
    { t.has_alphar_taudelta } -> std::same_as<std::false_type&>;
};

void get(auto t){
    if constexpr (UsesTauDelta<decltype(t)>){
        std::cout << "Defined and used" << std::endl;
    }
    else if constexpr (DoesntUseTauDelta<decltype(t)>){
        std::cout << "Defined and not used" << std::endl;
    }
    else{
        std::cout << "unknown" << std::endl;
    }
}

struct Adeftrue{
    std::false_type has_alphar_taudelta;
};

struct Bdeffalse{
    std::true_type has_alphar_taudelta;
};
struct C{
};

TEST_CASE("Test some C++20 things", "[C++20]"){
    static_assert(DoesntUseTauDelta<Adeffalse>);
    static_assert(UsesTauDelta<Adeftrue>);
}
