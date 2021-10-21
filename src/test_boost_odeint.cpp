#include <iostream>
#include <valarray>

#include <boost/numeric/odeint/stepper/controlled_runge_kutta.hpp>
#include <boost/numeric/odeint/stepper/runge_kutta_cash_karp54.hpp>
#include <boost/numeric/odeint/stepper/euler.hpp>

/* The type of container used to hold the state vector */
typedef std::vector< double > state_type;

void xprime( const state_type &x , state_type &dxdt , const double /* t */ )
{
    const double gam = 0.15;
    dxdt[0] = x[1];
    dxdt[1] = -x[0] - gam*x[1];
}

int main(int /* argc */ , char** /* argv */ )
{
    using namespace boost::numeric::odeint;

    // Typedefs for the types
    typedef runge_kutta_cash_karp54< state_type > error_stepper_type;
    typedef controlled_runge_kutta< error_stepper_type > controlled_stepper_type;

    // Define the tolerances
    double abs_err = 1.0e-10 , rel_err = 1.0e-6 , a_x = 1.0 , a_dxdt = 1.0;
    controlled_stepper_type controlled_stepper( default_error_checker< double , range_algebra , default_operations >( abs_err , rel_err , a_x , a_dxdt ) );

    state_type x0 = {1.0, 0.0}; // Starting point

    // First integrate, adaptively, until you get as close to the end as you can
    double t = 0, dt = 0.001, tmax = 10.0;
    auto write = [&]() { std::cout << t << " " << x0[0] << "," << x0[1] << std::endl;  };
    for (auto i = 0; t < tmax; ++i){
        if (t + dt > tmax) { break; }
        write();
        auto status = controlled_stepper.try_step(xprime, x0, t, dt);
        // The state is mutable, so here is where you could modify the solution
    }
    write();
    double dtfinal = tmax - t;
    auto status = controlled_stepper.try_step(xprime, x0, t, dtfinal);
    write();

    {
        state_type x0 = { 1.0, 0.0 }; // Starting point
        euler<state_type> eul;

        double t = 0, dt = 0.01, tmax = 10.0;
        auto write = [&]() { std::cout << t << " " << x0[0] << "," << x0[1] << std::endl;  };
        for (auto i = 0; t < tmax; ++i) {
            if (t + dt > tmax) { break; }
            write();
            eul.do_step(xprime, x0, t, dt);
            t += dt;
        }
        write();
        double dtfinal = tmax - t;
        eul.do_step(xprime, x0, t, dt);
        write();
    }

}