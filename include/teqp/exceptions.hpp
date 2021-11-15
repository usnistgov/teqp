#pragma once

#include <exception>

namespace teqp {

    class teqpcException : public std::exception {
    public:
        const int code;
        const std::string msg;
        teqpcException(int code, const std::string& msg) : code(code), msg(msg) {};
        const char* what() const noexcept override {
            return msg.c_str();
        }
    };

}; // namespace teqp