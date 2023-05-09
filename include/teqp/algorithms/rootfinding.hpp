#pragma once

namespace teqp{

template<typename Callable, typename Inputs>
auto NewtonRaphson(Callable f, const Inputs& args, double tol) {
    // Jacobian matrix
    Eigen::ArrayXd x = args, r0;
    Eigen::MatrixXd J(args.size(), args.size());
    for (int iter = 0; iter < 30; ++iter) {
        r0 = f(x);
        for (auto i = 0; i < args.size(); ++i) {
            auto dri = std::max(1e-6 * x[i], 1e-8);
            auto argsplus = x;
            argsplus[i] += dri;
            J.col(i) = (f(argsplus) - r0) / dri; // Forward diff to avoid negative concentration possibility
        }
        Eigen::ArrayXd v = J.colPivHouseholderQr().solve(-r0.matrix());
        x += v;
        auto err = r0.matrix().norm();
        if (!std::isfinite(err)){
            throw std::invalid_argument("err is now NaN");
        }
        if (err < tol) {
            break;
        }
    }
    return x;
}

}; /* namespace teqp */