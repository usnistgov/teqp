#include <boost/asio/thread_pool.hpp>
#include <boost/asio/post.hpp>

#include "teqp/models/multifluid.hpp"
#include "teqp/derivs.hpp"
#include "teqp/algorithms/VLE.hpp"

#include <iostream>
#include <vector>

int simple() {

    std::vector<int> numbers = { 1,2,3,4,5,6,7,8,-8,9,10,11 };
    std::vector<std::string> outputs(numbers.size());

    // Launch the pool with four threads.
    boost::asio::thread_pool pool(4);

    auto serial = [&numbers, &outputs]() {
        std::size_t i = 0;
        for (auto& num : numbers) {
            outputs[i] = "as str: " + std::to_string(num);
            i++;
        }
    };
    
    auto parallel = [&numbers, &outputs, &pool]() {
        std::size_t i = 0;
        for (auto& num : numbers) {
            auto& o = outputs[i];
            // Submit a lambda object to the pool.
            boost::asio::post(pool, [&num, &o]() {o = "as str: " + std::to_string(num); });
            i++;
        }
        // Wait for all tasks in the pool to complete.
        pool.join();
    };

    auto tic = std::chrono::high_resolution_clock::now();
    serial();
    auto tic2 = std::chrono::high_resolution_clock::now();
    parallel();
    auto tic3 = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration<double>(tic2 - tic).count() << std::endl;
    std::cout << std::chrono::duration<double>(tic3 - tic2).count() << std::endl;

    for (auto& o : outputs) {
        std::cout << o << std::endl;
    }
    return 0;
}

int evaluation() {

    double T = 308.15;
    auto rhovecL = (Eigen::ArrayXd(2) << 0.0, 55174.92375117).finished();
    auto rhovecV = (Eigen::ArrayXd(2) << 0.0, 2.20225704).finished();
    
    using MultiFluid = decltype(teqp::build_multifluid_model(std::vector<std::string>{"", ""}, "", ""));
    std::vector<MultiFluid> models(40, teqp::build_multifluid_model({ "CarbonDioxide", "Water" }, "../teqp/fluiddata"));
    std::vector<std::string> outputs(models.size());

    auto serial = [&]() {
        std::size_t i = 0;
        for (auto& model : models) {
            using TDX = teqp::TDXDerivatives<decltype(model)>;
            outputs[i] = teqp::trace_VLE_isotherm_binary(model, T, rhovecL, rhovecV).dump();
            i++;
        }
    };
    auto parallel = [&]() {
        // Launch the pool with four threads.
        boost::asio::thread_pool pool(40);

        std::size_t i = 0;
        for (auto& model : models) {
            auto &model_ = models[0];
            auto& o = outputs[i];
            // Submit a lambda object to the pool.
            boost::asio::post(pool, [&model_, &o, &T, &rhovecL, &rhovecV]() {
                using TDX = teqp::TDXDerivatives<decltype(model_)>;
                o = teqp::trace_VLE_isotherm_binary(model_, T, rhovecL, rhovecV).dump();
            });
            i++;
        }

        // Wait for all tasks in the pool to complete.
        pool.join();
    };

    auto tic = std::chrono::high_resolution_clock::now();
    serial();
    auto tic2 = std::chrono::high_resolution_clock::now();
    parallel();
    auto tic3 = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration<double>(tic2 - tic).count() << std::endl;
    std::cout << std::chrono::duration<double>(tic3 - tic2).count() << std::endl;
    
    return 0;
}


int main() {
//    simple();
    evaluation();
}
