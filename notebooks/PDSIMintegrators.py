"""
These integrator classes started their life in PDSim (https://pdsim.readthedocs.io/en/latest/) and since have 
been updated and some additional functionality has been added
"""

from __future__ import division, print_function

import abc, math
import numpy as np
import matplotlib.pyplot as plt

class AbstractODEIntegrator(object):
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def get_initial_array(self):
        pass
        
    @abc.abstractmethod
    def pre_step_callback(self):
        pass
        
    @abc.abstractmethod
    def post_deriv_callback(self):
        pass
        
    @abc.abstractmethod
    def post_step_callback(self):
        pass
        
    @abc.abstractmethod
    def derivs(self):
        pass
        
    @abc.abstractmethod
    def do_integration(self):
        pass
    
class AbstractSimpleEulerODEIntegrator(AbstractODEIntegrator):
    __metaclass__ = abc.ABCMeta
        
    def do_integration(self,N,tmin,tmax):
        """
        The simple Euler ODE integrator
        
        Parameters
        ----------
        N : integer
            Number of steps taken (not including the initial step).
        tmin : float
            Starting value of the independent variable.  ``t`` is in the closed range [``tmin``, ``tmax``]
        tmax : float
            Ending value for the independent variable.  ``t`` is in the closed range [``tmin``, ``tmax``] 
            
        Returns
        -------
        abort_flag
            If an abort has been requested (by returning a value other than ``False`` from ``premature_termination``), return value from ``premature_termination``.  ``None`` otherwise
        """
        
        # Step variables
        self.t0 = tmin
        self.h = (tmax-tmin)/(N)
        
        # Get the initial value
        self.xold = self.get_initial_array()
        
        for self.Itheta in range(N):
            
            # Check for termination
            abort = self.premature_termination()
            if abort != False: return abort
            
            # Call the pre-step callback
            self.pre_step_callback()
            
            # Derivatives evaluated at old values of t = t0
            self.f1 = self.derivs(self.t0, self.xold)
            
            # Call post derivative callback after the first derivative evaluation (which might cache values)
            self.post_deriv_callback()
            
            # Calculate the new values
            self.xnew = self.xold + self.h*self.f1
            
            # Everything from this step is finished, now update for the next
            # step coming
            self.t0 += self.h
            self.xold = self.xnew[:]
            
            # Call the post-step callback
            self.post_step_callback()
            
        # Bump up once more
        self.Itheta += 1
        
        # Make sure we end up at the right place
        assert((self.t0 - tmax) < 1e-10)
        
        # No termination was requested
        return False

class AbstractHeunODEIntegrator(AbstractODEIntegrator):
    __metaclass__ = abc.ABCMeta
        
    def do_integration(self, N, tmin, tmax):
        """
        The Heun system of ODE integrator
        
        Parameters
        ----------
        N : integer
            Number of steps taken.  There will be N+1 entries in the state matrices
        tmin : float
            Starting value of the independent variable.  ``t`` is in the closed range [``tmin``, ``tmax``]
        tmax : float
            Ending value for the independent variable.  ``t`` is in the closed range [``tmin``, ``tmax``] 
            
        Returns
        -------
        abort_flag
            If an abort has been requested (by returning a value other than ``False`` from ``premature_termination``), return value from ``premature_termination``.  ``None`` otherwise
        
        """
        
        # Step variables
        self.t0 = tmin
        self.h = (tmax-tmin)/(N)
        
        # Get the initial value
        self.xold = self.get_initial_array()
        
        for self.Itheta in range(N):
            
            # Check for termination
            abort = self.premature_termination()
            if abort != False: return abort
            
            # Call the pre-step callback
            self.pre_step_callback()
            
            # Step 1: derivatives evaluated at old values
            self.f1 = self.derivs(self.t0, self.xold)
            
            # Call post derivative callback after the first derivative evaluation (which might cache values)
            self.post_deriv_callback()
            
            # Predicted values based on extrapolation using initial derivatives
            self.xtemp = self.xold + self.h*self.f1
            
            # Step 2: Evaluated at predicted step
            self.f2 = self.derivs(self.t0 + self.h, self.xtemp)
            
            # Get corrected values
            self.xnew = self.xold + self.h/2.0*(self.f1 + self.f2)
            
            # Everything from this step is finished, now update for the next
            # step coming
            self.t0 += self.h
            self.xold = self.xnew
            
            # Call the post-step callback
            self.post_step_callback()
        
        # Bump up once more
        self.Itheta += 1
        
        # No termination was requested
        return False
            
class AbstractRK45ODEIntegrator(AbstractODEIntegrator):
    __metaclass__ = abc.ABCMeta
        
    def do_integration(self,
                   tmin=0,
                   tmax=2.0*math.pi,
                   hmin=1e-4,
                   atol=1e-3,
                   rtol=1e-10,
                   step_relax=0.9,
                   **kwargs):
        """
        
        This function implements an adaptive Runge-Kutta-Feldberg 4th/5th order
        solver for the system of equations
        
        Parameters
        ----------
        hmin : float
            Minimum step size, something like 1e-5 usually is good.  Don't make this too big or you may not be able to get a stable solution
        tmin : float
            Starting value of the independent variable.  ``t`` is in the closed range [``tmin``, ``tmax``]
        tmax : float
            Ending value for the independent variable.  ``t`` is in the closed range [``tmin``, ``tmax``]
        eps_allowed : float
            Maximum absolute error of any CV per step allowed.  Don't make this parameter too big or you may not be able to get a stable solution.  Also don't make it too small because then you are going to run into truncation error.
        step_relax : float, optional
            The relaxation factor that is used in the step resizing algorithm.  Should be less than 1.0; you can play with this parameter to improve the adaptive resizing, but should not be necessary.
            
        Returns
        -------
        abort_flag
            If an abort has been requested (by returning a value other than ``False`` from ``premature_termination``), return value from ``premature_termination``.  ``None`` otherwise
        
        Notes
        -----
        
        Mathematically the adaptive solver can be expressed as::
        
            k1=h*dy(xn                                                                   ,t)
            k2=h*dy(xn+1.0/4.0*k1                                                        ,t+1.0/4.0*h)
            k3=h*dy(xn+3.0/32.0*k1+9.0/32.0*k2                                           ,t+3.0/8.0*h)
            k4=h*dy(xn+1932.0/2197.0*k1-7200.0/2197.0*k2+7296.0/2197.0*k3                ,t+12.0/13.0*h)
            k5=h*dy(xn+439.0/216.0*k1-8.0*k2+3680.0/513.0*k3-845.0/4104.0*k4             ,t+h)
            k6=h*dy(xn-8.0/27.0*k1+2.0*k2-3544.0/2565.0*k3+1859.0/4104.0*k4-11.0/40.0*k5 ,t+1.0/2.0*h)

        where the function dy(y,t) returns a vector of the ODE expressions.
        The new value is calculated from::
        
            xnplus=xn+gamma1*k1+gamma2*k2+gamma3*k3+gamma4*k4+gamma5*k5+gamma6*k6

        In the adaptive solver, the errors for a given step can be calculated from::

            error=1.0/360.0*k1-128.0/4275.0*k3-2197.0/75240.0*k4+1.0/50.0*k5+2.0/55.0*k6

        If the maximum absolute error is above allowed error, the step size is decreased and the step is 
        tried again until the error is below tolerance.  If the error is better than required, the step
        size is increased to minimize the number of steps required.
        
        Before the step is run, a callback the ``step_callback`` method of this class is called.  In the ``step_callback`` callback function you can do anything you want, but you must return 
        """
        
        # Get the starting array of variables
        self.xold = self.get_initial_array()
        
        # Start at an index of 0
        self.Itheta = 0
        self.t0 = tmin
        self.tmax = tmax
        self.h = hmin
        self.minstepcount = 0
        
        # gamma1=16.0/135.0
        # gamma2=0.0
        # gamma3=6656.0/12825.0
        # gamma4=28561.0/56430.0
        # gamma5=-9.0/50.0
        # gamma6=2.0/55.0
        
        #t is the independent variable here, where t takes on values in the bounded range [tmin,tmax]
        while (self.t0 < self.tmax - 1e-10):
            
            # Check for termination
            abort = self.premature_termination()
            if abort != False: return abort

            self.stepAccepted = False
            
            while not self.stepAccepted:

                # reset the flag
                self.disableAdaptive = False
                
                if self.t0 + self.h > self.tmax:
                    self.disableAdaptive = True
                    self.h = self.tmax - self.t0
            
                # 
                self.pre_step_callback()
                
                # We check stepAccepted again because if the derived class
                # sets the variable stepAccepted, we should not actually do the evaluation
                if not self.stepAccepted:
                    
                    if self.h < hmin and not self.disableAdaptive:
                        # Step is too small, just use the minimum step size
                        self.h = 1.0*hmin
                        self.disableAdaptive = True
                    if self.h == hmin:
                        self.minstepcount += 1
                    else:
                        self.minstepcount = 0
                
                    # Step 1: derivatives evaluated at old values
                    self.f1 = self.derivs(self.t0, self.xold)
                    
                    # Call post derivative callback after the first derivative evaluation (which might cache values)
                    self.post_deriv_callback()
                
                    self.xnew1 = self.xold+self.h*(1.0/5.0)*self.f1
                    
                    self.f2 = self.derivs(self.t0+1.0/5.0*self.h, self.xnew1)
                    self.xnew2 = self.xold+self.h*(+3.0/40.0*self.f1+9.0/40.0*self.f2)

                    self.f3 = self.derivs(self.t0+3.0/10.0*self.h, self.xnew2)
                    self.xnew3 = self.xold+self.h*(3.0/10.0*self.f1-9.0/10.0*self.f2+6.0/5.0*self.f3)

                    self.f4 = self.derivs(self.t0+3.0/5.0*self.h, self.xnew3)
                    self.xnew4 = self.xold+self.h*(-11.0/54.0*self.f1+5.0/2.0*self.f2-70/27.0*self.f3+35.0/27.0*self.f4)
                    
                    self.f5 = self.derivs(self.t0+self.h, self.xnew4)
                    self.xnew5 = self.xold+self.h*(1631.0/55296*self.f1+175.0/512.0*self.f2+575.0/13824.0*self.f3+44275.0/110592.0*self.f4+253.0/4096.0*self.f5)
                    
                    self.f6 = self.derivs(self.t0+7/8*self.h, self.xnew5)
                    
                    # Updated values at the next step using 5-th order
                    self.xnew = self.xold + self.h*(37/378*self.f1 + 250/621*self.f3 + 125/594*self.f4 + 512/1771*self.f6)
                    
                    # Estimation of error
                    error = abs(self.h*(-277/64512*self.f1+6925/370944*self.f3-6925/202752*self.f4-277.0/14336*self.f5+277/7084*self.f6))

                    error_threshold = atol + rtol*abs(self.xnew)

                    # max_error = np.sqrt(np.sum(np.power(error, 2)))

                    # rel_error = error/self.xnew
                    # rel_error[self.xnew == 0] = error[self.xnew == 0]
                    # max_error = np.max(np.abs(rel_error))

                    # print(max_error, error, self.xnew)
                    # print('error @h=',self.h, abs(error), abs(error_threshold))
                    
                    # If the error is too large, make the step size smaller and try
                    # the step again
                    if (any(error > error_threshold)):
                        if not self.disableAdaptive:
                            # Take a smaller step next time, try again on this step
                            # But only if adaptive mode is on
                            downsize_factor = np.min(error_threshold/error)**(0.3)
                            # print('downsize', downsize_factor, error, error_threshold)
                            self.h *= step_relax*downsize_factor
                            self.stepAccepted=False
                        else:
                            # Accept the step regardless of whether the error 
                            # is too large or not
                            self.stepAccepted = True
                    else:
                        self.stepAccepted = True
                else:
                    pass
                    # print('accepted')  

            self.t0 += self.h
            self.Itheta += 1
            self.xold = self.xnew

            self.post_step_callback()
            
            # The error is already below the threshold
            if (all(abs(error) < error_threshold) and self.disableAdaptive == False and np.max(error) > 0):
                # Take a bigger step next time, since eps_allowed>max_error
                upsize_factor = step_relax*np.max(error_threshold/error)**(0.2)
                # print('upsizing', upsize_factor, (error_threshold/error)**(0.2), self.h, self.h*upsize_factor)
                self.h *= upsize_factor
               
        if not (self.t0 - tmax) < 1e-3:
            raise AssertionError('(self.t0 - tmax) [' + str(self.t0 - tmax) + '] > 1e-3')
        
        # No termination was requested
        return False
            
if __name__ == '__main__':
    
    class TestIntegrator(object):
        """
        Implements the functions needed to satisfy the ABC requirements
        
        This is the problem from wikipedia, where y' = y, with the explicit solution y = exp(t)
        """
        
        def __init__(self):
            self.x, self.y = [], []
            
        def post_deriv_callback(self): pass
        
        def premature_termination(self): return False
            
        def get_initial_array(self):
            return np.array([1.0])
        
        def pre_step_callback(self): 
            if self.Itheta == 0:
                self.x.append(self.t0)
                self.y.append(self.xold[0])
        
        def post_step_callback(self): 
            self.x.append(self.t0)
            self.y.append(self.xold[0])
        
        def derivs(self, t0, xold):
            return np.array([xold[0]])
            
    class TestEulerIntegrator(TestIntegrator, AbstractSimpleEulerODEIntegrator):
        """ Mixin class using the functions defined in TestIntegrator """
        pass

    class TestHeunIntegrator(TestIntegrator, AbstractHeunODEIntegrator):
        """ Mixin class using the functions defined in TestIntegrator """
        pass
            
    class TestRK45Integrator(TestIntegrator, AbstractRK45ODEIntegrator):
        """ Mixin class using the functions defined in TestIntegrator """
        pass
    
    for N in [4, 11, 21]:
        TEI = TestEulerIntegrator()
        TEI.do_integration(N, 0.0, 4.0)
        plt.plot(TEI.x, TEI.y, 'o-', label = 'Euler: ' + str(N))
        
    for N in [4, 11]:
        THI = TestHeunIntegrator()
        THI.do_integration(N, 0.0, 4.0)
        plt.plot(THI.x, THI.y, '^-', label = 'Heun: ' + str(N))
        
    TRKI = TestRK45Integrator()
    TRKI.do_integration(0.0, 4.0, rtol=1e-10)
    plt.plot(TRKI.x, TRKI.y, '^-', label = 'RK45')
        
    t = np.linspace(0, 4, 500)
    plt.plot(t, np.exp(t), 'k', lw = 1)
    plt.legend(loc='best')
    plt.show()
