#include <catch2/catch_test_macros.hpp>

#include <iostream>
#include <concepts>

template<typename T, typename target_type>
concept TauDeltaDef = requires(T t) {
    { t.has_alphar_taudelta } -> std::same_as<target_type&>;
};

void get(auto t){
    if constexpr (TauDeltaDef<decltype(t), std::true_type>){
        std::cout << "Defined and used" << std::endl;
    }
    else if constexpr (TauDeltaDef<decltype(t), std::false_type>){
        std::cout << "Defined and not used" << std::endl;
    }
    else{
        std::cout << "unknown" << std::endl;
    }
}

struct Adeftrue{
    std::true_type has_alphar_taudelta;
};
struct Adeffalse{
    std::false_type has_alphar_taudelta;
};
struct C{
};

TEST_CASE("Test some C++20 things", "[C++20]"){
    static_assert(TauDeltaDef<Adeffalse, std::false_type>);
    static_assert(TauDeltaDef<Adeftrue, std::true_type>);
}
