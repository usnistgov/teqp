#pragma once

#include <exception>

class teqpcException : public std::exception {
public:
    const int code;
    const std::string msg;
    teqpcException(int code, const std::string& msg) : code(code), msg(msg) {};
    const char *what() const override {
        return msg.c_str();
    }
};