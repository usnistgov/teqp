#include "autodiff/forward/dual.hpp"
#include <iostream>

template<typename T1, typename T2>
auto f(const T1 x, const T2 y){
    return x*sin(y);
};
int main(){
    
    autodiff::dual2nd x = 3.8, y = 9.8;
    auto g = f(x, y);
    std::cout << g.l << std::endl;
}
