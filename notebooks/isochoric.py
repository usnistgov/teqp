# Standard libraries
import timeit, sys

# Scipy stack packages
import numpy as np
import pandas
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Some integration tools from PDSim
from PDSIMintegrators import AbstractRK45ODEIntegrator, AbstractSimpleEulerODEIntegrator

# Special packages
import teqp
import CoolProp.CoolProp as CP

def get_dxdp(*, rhovec, drhovecdp):
    rhomolar = np.sum(rhovec)
    molefrac = rhovec/rhomolar
    drhodp = np.sum(drhovecdp)
    return 1.0/rhomolar*(drhovecdp-molefrac*drhodp)

def get_drhovecdx(*, i, model, T, rhovec):
    rhovecL = rhovec[0:2]
    rhovecV = rhovec[2::]
    drhovecdpL, drhovecdpV = get_drhovecdp_sat(model, T, rhovecL, rhovecV)
    dpdxL = 1.0/get_dxdp(rhovec=rhovecL, drhovecdp=drhovecdpL)
    return np.array((drhovecdpL*dpdxL[i]).tolist()+(drhovecdpV*dpdxL[i]).tolist())

def traceT_x(model, T, rhovec0, *, kwargs={}):
    R = model.get_R(np.array([0.0, 1.0]))
    def rhovecprime(x, rhovec):
        print(x)
        return get_drhovecdx(i=0, model=model, T=T, rhovec=rhovec)
    tic = timeit.default_timer()
    sol = solve_ivp(rhovecprime, [1.0, 0.2], y0=rhovec0, method='RK45',dense_output=True, t_eval = np.linspace(1,0.21,1000), rtol=1e-8)
    p = [teqp.get_pr(model, T, sol.y[0:2,j]) + R*T*sol.y[0:2,j].sum() for j in range(len(sol.t))]
    toc = timeit.default_timer()
    x = sol.y[0,:]/(sol.y[0,:] + sol.y[1,:])
    y = sol.y[2,:]/(sol.y[2,:] + sol.y[3,:])
    plt.plot(x, p, 'r', **kwargs)
    plt.plot(y, p, 'g', **kwargs)

def traceT_arclength(model, T, rhovec0, *, kwargs={}):

    R = model.get_R(np.array([0.0, 1.0]))
    print('T:', T, 'K')

    c = 1.0
    def norm(x): return (x*x).sum()**0.5

    def rhovecprime(t, rhovec):
        tic = timeit.default_timer()
        drhovecdpL, drhovecdpV = teqp.get_drhovecdp_Tsat(model, T, rhovec[0:2], rhovec[2::])
        # drhovecdpL, drhovecdpV = tracer.get_drhovecdp_sat(rhovec[0:2], rhovec[2::], T)[0:2]
        # print(, drhovecdpL, drhovecdpV)

        toc = timeit.default_timer()
        # print(toc-tic)
        dpdt = (norm(drhovecdpL) + norm(drhovecdpV))**-0.5
        der = np.zeros_like(rhovec)
        der[0:2] = (drhovecdpL*dpdt).squeeze()
        der[2:4] = (drhovecdpV*dpdt).squeeze()
        # print(t, rhovec, rhovec[0]+rhovec[1], rhovec[2]+rhovec[3], rhovec[0]/(rhovec[0]+rhovec[1]))
        # if any(~np.isfinite(rhovec)):
        #     raise ValueError()
        return c*der

    class TestIntegrator(object):
        """
        Implements the functions needed to satisfy the ABC requirements
        """
        
        def __init__(self, Nmax):
            self.x, self.y = [], []
            self.Itheta = 0
            self.Nmax = Nmax
            self.termination_reason = None
            self.minstepcount = 0
            
        def post_deriv_callback(self): pass
        
        def premature_termination(self): 
            if self.Itheta > self.Nmax: 
                self.termination_reason = 'too many steps'
                return True
            if self.minstepcount > 10: 
                self.termination_reason = '10 tiny steps in a row'
                return True
            if np.any(self.xold<0): 
                self.termination_reason = 'negative x'
                return True
            if np.any(~np.isfinite(self.xold)): 
                self.termination_reason = 'nan value of x'
                return True
            return False
            
        def get_initial_array(self):
            return rhovec0.copy()
        
        def pre_step_callback(self): 
            if self.Itheta == 0:
                self.x.append(0)
                self.y.append(self.get_initial_array().tolist())
        
        def post_step_callback(self): 
            self.x.append(self.t0_cache)
            self.y.append(self.xold_cache.tolist())
            # Polish the solution...
        
        def derivs(self, t0, xold):
            self.t0_cache = t0
            self.xold_cache = xold.copy()
            return rhovecprime(t0, xold)
            
    class EulerIntegrator(TestIntegrator, AbstractSimpleEulerODEIntegrator):
        """ Mixin class using the functions defined in TestIntegrator """
        pass

    class RK45Integrator(TestIntegrator, AbstractRK45ODEIntegrator):
        """ Mixin class using the functions defined in TestIntegrator """
        pass
    
    der_init = rhovecprime(0, rhovec0)
    if np.any(rhovec0 + der_init*1e-6 < 0):
        c *= -1
        print('flip c', der_init, rhovec0)   

    events = [lambda t,x: x[0], lambda t,x: x[1],
              lambda t,x: x[2], lambda t,x: x[3]]
    events.append(lambda t, z: ((z[0]+z[1])/(z[2]+z[3])-1)-0.2)
    for e in events:
        e.direction = -1
        e.terminal = True

    # tic = timeit.default_timer()
    # sol = solve_ivp(rhovecprime, [0.0, 200000], y0=rhovec0.copy(), method='RK45',dense_output=True, events=events, rtol=1e-8, atol=1e-10)
    # # print(sol.t_events, sol.y_events)
    # # print(sol.message)
    # print(sol.t, sol.nfev)
    # toc = timeit.default_timer()
    # print(toc-tic,'s for parametric w/ solve_ivp')
    # t = np.linspace(np.min(sol.t), np.max(sol.t), 10000)
    # soly = sol.sol(t)

    tic = timeit.default_timer()
    i = RK45Integrator(Nmax=1000)
    i.do_integration(0, 5000000, atol=0, rtol=1e-8, hmin=4.46772051e-03)   
    t = i.x
    soly = np.array(i.y).T
    toc = timeit.default_timer()
    print(toc-tic,'s for parametric w/ my RK45')
    reason = i.termination_reason
    # if reason:
    #     print('reason:', reason)
    print(i.Itheta)

    # print(t[0:len(t)-1], np.diff(t))
    x = soly[0,:]/(soly[0,:] + soly[1,:])
    y = soly[2,:]/(soly[2,:] + soly[3,:])
    pL = [teqp.get_pr(model, T, soly[0:2,j]) + R*T*soly[0:2,j].sum() for j in range(len(t))]
    pV = [teqp.get_pr(model, T, soly[2:4,j]) + R*T*soly[2:4,j].sum() for j in range(len(t))]
    plt.plot(x, pL, 'r', ms=0.1, **kwargs)
    plt.plot(y, pV, 'g', ms=0.1, **kwargs)
    # plt.plot(y, p,  'g', **kwargs)

def prettyPX(model, puremodels, names, ipure, Tvec, *, plot_critical=True, show=True, ofname=None):
    """
    ipure: int
        index (0-based) of the pure fluid model to use 
    """
    def plot_critical_curve():
        tic = timeit.default_timer()
        Tcvec = model.get_Tcvec()
        rhocvec = 1/model.get_vcvec()
        k = 1
        T0 = Tcvec[k]
        rho0 = rhocvec
        rho0[1-k] = 0
        curveJSON = teqp.trace_critical_arclength_binary(model, T0, rho0, "")
        toc = timeit.default_timer()
        print(toc-tic, 'to trace critical locus')
        df = pandas.DataFrame(curveJSON)
        df['z0 / mole frac.'] = df['rho0 / mol/m^3']/(df['rho0 / mol/m^3']+df['rho1 / mol/m^3'])
        plt.plot(df['z0 / mole frac.'], df['p / Pa'], lw=2)
        #print(toc-tic, 'complete critical locus')
        plt.ylim(1e-3, np.max(df['p / Pa']))

    def plot_isotherms():
        for T in Tvec:
            puremodel = puremodels[ipure]
            # rhoL, rhoV = robust_VLE(puremodel, T, puremodel.get_Tcvec()[0], 1/puremodel.get_vcvec()[0])
            rhoL = CP.PropsSI('Dmolar','T',T,'Q',0,names[ipure])
            rhoV = CP.PropsSI('Dmolar','T',T,'Q',1,names[ipure])
            if ipure == 0:
                # Both phases are pure of the first component with index 0
                rhovec = np.array([rhoL, 0, rhoV, 0])
            else:
                # Both phases are pure of the second component with index 1
                rhovec = np.array([0, rhoL, 0, rhoV])
            try:
                traceT_arclength(model, T, rhovec.copy(), kwargs={'lw': 0.5})
            except BaseException as BE:
                print(BE)
                # raise

    if plot_critical:
        plot_critical_curve()
    plot_isotherms()
    plt.gca().set(xlabel='$x_1$ / mole frac.', ylabel='$p$ / Pa')
    plt.tight_layout(pad=0.2)
    if ofname:
        plt.savefig(ofname)
    if show:
        plt.show()

if __name__ == '__main__':
    names = ['Methane', 'n-Butane']
    model = teqp.build_multifluid_model(names, '../mycp', '../mycp/dev/mixtures/mixture_binary_pairs.json')#,{'estimate':'Lorentz-Berthelot'})
    puremodels = [teqp.build_multifluid_model([name], '../mycp', '../mycp/dev/mixtures/mixture_binary_pairs.json') for name in names]
    prettyPX(model, puremodels, names, 1, np.linspace(190, 423, 20), show=False, plot_critical=True, ofname='pxdiagram.pdf')