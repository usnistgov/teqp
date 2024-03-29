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
    public:
        const char* what() const noexcept override {
            return msg.c_str();
        }
    };

    // Exceptions related to arguments
    class InvalidArgument : public teqpException {
    public:
        InvalidArgument(const std::string& msg) : teqpException(1, msg) {};
    };

    // Exceptions related to calculations
    class IterationFailure : public teqpException {
    public:
        IterationFailure(const std::string& msg) : teqpException(100, msg) {};
    };
    using IterationError = IterationFailure;
    class InvalidValue : public teqpException {
    public:
        InvalidValue(const std::string& msg) : teqpException(101, msg) {};
    };

    class NotImplementedError : public teqpException {
    public:
        NotImplementedError(const std::string& msg) : teqpException(200, msg) {};
    };

    /// Validation of a JSON schema failed
    class JSONValidationError : public teqpException {
    private:
        auto errors_to_string(const std::vector<std::string> &errors, const std::string delim = "|/|\\|"){
            std::string o = "";
            if (errors.empty()){ return o; }
            o = errors[0];
            for (auto j = 1U; j < errors.size(); ++j){
                o += delim + errors[j];
            }
            return o;
        }
    public:
        JSONValidationError(const std::vector<std::string>& errors) : teqpException(300, errors_to_string(errors)) {};
    };

}; // namespace teqp
