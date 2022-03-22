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

    class teqpException : public std::exception {
    protected:
        const int code;
        const std::string msg;
        teqpException(int code, const std::string& msg) : code(code), msg(msg) {};
        const char* what() const noexcept override {
            return msg.c_str();
        }
    };

    class InvalidArgument : public teqpException {
    public:
        InvalidArgument(const std::string& msg) : teqpException(1, msg) {};
    };

}; // namespace teqp