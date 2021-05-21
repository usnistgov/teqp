import timeit, sys

import numpy as np
import pandas
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

import VLEIsoTracer as vle
# print('VLEIsoTracer is located at:', vle.__file__)
IMPOSED_T = vle.VLEIsolineTracer.imposed_variable.IMPOSED_T

sys.path.append('../bld/Release')
import teqp
import CoolProp.CoolProp as CP

def get_drhovecdp_sat(model, T, rhovecL, rhovecV):
    tic = timeit.default_timer()
    Hliq = teqp.build_Psi_Hessian_autodiff(model, T, rhovecL)
    Hvap = teqp.build_Psi_Hessian_autodiff(model, T, rhovecV)
    Hvap[~np.isfinite(Hvap)] = 1e20
    Hliq[~np.isfinite(Hliq)] = 1e20

    N = len(rhovecL)
    A = np.zeros((N, N))
    b = np.ones((N, 1))
    assert(len(rhovecL)==len(rhovecV))
    if np.all(rhovecL != 0) and np.all(rhovecV != 0):
        A[0,0] = np.dot(Hliq[0,:], rhovecV)
        A[0,1] = np.dot(Hliq[1,:], rhovecV)
        A[1,0] = np.dot(Hliq[0,:], rhovecL)
        A[1,1] = np.dot(Hliq[1,:], rhovecL)
        
        drhodp_liq = np.linalg.solve(A, b).squeeze()
        drhodp_vap = np.linalg.solve(Hvap, np.dot(Hliq,drhodp_liq))

        return drhodp_liq, drhodp_vap
    else:
        # Special treatment for infinite dilution
        
        murL = teqp.build_Psir_gradient_autodiff( model, T, rhovecL)
        murV = teqp.build_Psir_gradient_autodiff( model, T, rhovecV)
        RL = model.get_R(rhovecL/rhovecL.sum())
        RV = model.get_R(rhovecV/rhovecV.sum())

        # First, for the liquid part
        for i in range(N):
            for j in range(N):
                if rhovecL[j] == 0:
                    # Analysis is special if j is the index that is a zero concentration. If you are multiplying by the vector
                    # of liquid concentrations, a different treatment than the case where you multiply by the vector
                    # of vapor concentrations is required
                    # ...
                    # Initial values
                    Aij = Hliq[j,:]*(rhovecV if i == 0 else rhovecL) # coefficient-wise product
                    # A throwaway boolean for clarity
                    is_liq = (i==1)
                    # Apply correction to the j term (RT if liquid, RT*phi for vapor)
                    Aij[j] = RL*T if is_liq else RL*T*np.exp(-(murV[j] - murL[j])/(RL*T))
                    # Fill in entry
                    A[i, j] = Aij.sum()
                else:
                    # Normal
                    A[i, j] = np.dot(Hliq[j,:], rhovecV if (i == 0) else rhovecL)
        drhodp_liq = np.linalg.solve(A, b)

        # Then, for the vapor part, also requiring special treatment
        # Left-multiplication of both sides of equation by diagonal matrix with liquid concentrations along diagonal, all others zero
        diagrhovecL = np.diagflat(rhovecL)
        PSIVstar = diagrhovecL@Hvap
        PSILstar = diagrhovecL@Hliq
        for j in range(N):
            if rhovecL[j] == 0:
                PSILstar[j, j] = RL*T
                PSIVstar[j, j] = RV*T/np.exp(-(murV[j] - murL[j])/(RV*T))
        drhodp_vap = np.linalg.solve(PSIVstar, PSILstar@drhodp_liq)
        toc = timeit.default_timer()
        # print(toc-tic)
        return drhodp_liq.squeeze(), drhodp_vap.squeeze()

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

def traceT(model, T, rhovec0, *, kwargs={}):
    R = model.get_R(np.array([0.0, 1.0]))
    def rhovecprime(x, rhovec):
        return get_drhovecdx(i=0, model=model, T=T, rhovec=rhovec)
    tic = timeit.default_timer()
    sol = solve_ivp(rhovecprime, [0.0, 0.5], y0=rhovec0, method='RK45',dense_output=True, t_eval = np.linspace(0,0.49,1000), rtol=1e-8)
    p = [teqp.get_pr(model, T, sol.y[0:2,j]) + R*T*sol.y[0:2,j].sum() for j in range(len(sol.t))]
    toc = timeit.default_timer()
    print(toc-tic)
    y = sol.y[2,:]/(sol.y[2,:] + sol.y[3,:])
    plt.plot(sol.t, p, 'r',**kwargs)
    plt.plot(y, p, 'g',**kwargs)

def traceT_arclength(model, T, rhovec0, *, kwargs={}):

    R = model.get_R(np.array([0.0, 1.0]))
    print(T)
    c = 1.0
    def rhovecprime(t, rhovec):
        tic = timeit.default_timer()
        drhovecdpL, drhovecdpV = teqp.get_drhovecdp_Tsat(model, T, rhovec[0:2], rhovec[2::])
        toc = timeit.default_timer()
        # print(toc-tic)

        def norm(x):
            return (x*x).sum()**0.5
        dpdt = (norm(drhovecdpL) + norm(drhovecdpV))**-0.5
        der = np.zeros_like(rhovec)
        der[0:2] = (drhovecdpL*dpdt).squeeze()
        der[2:4] = (drhovecdpV*dpdt).squeeze()
        # print(rhovec, rhovec[0]+rhovec[1], rhovec[2]+rhovec[3])
        return c*der

    der_init = rhovecprime(0, rhovec0)
    if np.any(rhovec0 + der_init*1e-6 < 0):
        c *= -1
    # print('init', der_init, T)

    events = [lambda t,x: x[0], lambda t,x: x[1],
              lambda t,x: x[2], lambda t,x: x[3]]
    events.append(lambda t, z: ((z[0]+z[1])/(z[2]+z[3])-1)-0.2)
    # class StepCounter():
    #     def __init__(self, Nmax):
    #         self.counter = 0
    #         self.Nmax = Nmax
    #     def __call__(self, t, z):
    #         self.counter += 1
    #         v = self.Nmax-self.counter+0.5
    #         return v
    # events.append(StepCounter(200))
    for e in events:
        e.direction = -1
        e.terminal = True

    tic = timeit.default_timer()
    sol = solve_ivp(rhovecprime, [0.0, 3500000], y0=rhovec0.copy(), method='RK45',dense_output=True, events=events)#, rtol=1e-8)
    # print(sol.t_events, sol.y_events)
    # print(sol.message)
    toc = timeit.default_timer()
    print(toc-tic,'s for parametric')
    t = np.linspace(np.min(sol.t), np.max(sol.t), 1000)
    soly = sol.sol(t)
    x = soly[0,:]/(soly[0,:] + soly[1,:])
    y = soly[2,:]/(soly[2,:] + soly[3,:])
    p = [teqp.get_pr(model, T, soly[0:2,j]) + R*T*soly[0:2,j].sum() for j in range(len(t))]
    plt.plot(x, p, 'r', **kwargs)
    plt.plot(y, p, 'g', **kwargs)
    # print(x,p)

def drhovecdT_isopleth(model, T, rhovec):
    assert(len(rhovec)==4)
    rhovecL = rhovec[0:2]
    rhovecV = rhovec[2::]
    Hliq = teqp.build_Psi_Hessian_autodiff(model, T, rhovecL)
    Hvap = teqp.build_Psi_Hessian_autodiff(model, T, rhovecV)
    rhoL = sum(rhovecL)
    rhoV = sum(rhovecV)
    xL = rhovecL/rhoL
    xV = rhovecV/rhoV
    R = model.get_R(xL)
    deltas = teqp.get_dchempotdT_autodiff(model, T, rhovecV) - teqp.get_dchempotdT_autodiff(model, T, rhovecL)
    deltarho = rhovecV-rhovecL
    dpdTV = R*rhoV*(1+model.get_Ar01(T, rhoV, xV)-model.get_Ar11(T, rhoV, xV))
    dpdTL = R*rhoL*(1+model.get_Ar01(T, rhoL, xL)-model.get_Ar11(T, rhoL, xL))
    deltabeta = dpdTV-dpdTL
    drhovecdTL = (np.dot(deltas,rhovecV)-deltabeta)/np.dot(Hliq@deltarho, xL)*xL
    drhovecdTV = np.linalg.solve(Hvap, Hliq@drhovecdTL-deltas)
    return np.array(drhovecdTL.tolist()+drhovecdTV.tolist())

def robust_VLE(model, T, Tc, rhoc):
    DeltaT = 1e-2*Tc
    Tg = Tc-DeltaT
    if Tg < 0:
        raise ValueError("T > Tc")
    rholiq, rhovap = teqp.extrapolate_from_critical(model, Tc, rhoc, Tg)
    if rholiq < 0 or rhovap < 0:
        raise ValueError("Negative density obtained from critical extrapolation")
    rholiq, rhovap = teqp.pure_VLE_T(model, Tg, rholiq, rhovap, 100)

    z = np.array([1.0])
    R = model.get_R(z)

    def dpdTsat(model, T, rhoL, rhoV):
        """ dp/dT = ((hV-hL)/T)/(vV-vL) 
        The ideal parts of enthalpy cancel, leaving just the residual parts
        """
        numV = R*(model.get_Ar01(T, rhoV, z) + model.get_Ar10(T,rhoV,z))
        numL = R*(model.get_Ar01(T, rhoL, z) + model.get_Ar10(T,rhoL,z))
        num = numV-numL
        den = 1/rhoV-1/rhoL
        return num/den

    def get_drhodT_sat(model, T, rhoL, rhoV, Q):
        dpdT = dpdTsat(model, T, rhoL, rhoV)
        
        rho = rhoL if Q == 0 else rhoV
        Ar01 = model.get_Ar01(T, rho, z)
        Ar02 = model.get_Ar02n(T, rho, z)[2]
        Ar11 = model.get_Ar11(T, rho, z)

        dpdrho__T = R*T*(1 + 2*Ar01 + Ar02)
        dpdT__rho = R*rho*(1 + Ar01 - Ar11)

        drhodT__p = -dpdT__rho/dpdrho__T
        drhodp__T = 1/dpdrho__T
        return drhodT__p + drhodp__T*dpdT

    def rhoprime(T, rhoLrhoV):
        rhoL, rhoV = rhoLrhoV
        drhoLdT = get_drhodT_sat(model, T, rhoL, rhoV, 0)
        drhoVdT = get_drhodT_sat(model, T, rhoL, rhoV, 1)
        return np.array([drhoLdT, drhoVdT])

    sol = solve_ivp(rhoprime, [Tg, T], y0=[rholiq, rhovap], method='RK45', dense_output=False, rtol=1e-8)
    rholiq, rhovap = sol.y[:,-1]
    rholiq, rhovap = teqp.pure_VLE_T(model, T, rholiq, rhovap, 100)
    return rholiq, rhovap

def prettyPX(model, puremodels, ipure, Tvec):
    """
    ipure: int
        index (0-based) of the pure fluid model to use 
    """
    def plot_critical_curve():
        tic = timeit.default_timer()
        Tcvec = model.get_Tcvec()
        rhocvec = 1/model.get_vcvec()
        k = 1 # Have to start at pure second component for now... In theory either are possible.
        T0 = Tcvec[k]
        rho0 = rhocvec
        rho0[1-k] = 0
        curveJSON = teqp.trace_critical_arclength_binary(model, T0, rho0, "")
        toc = timeit.default_timer()
        print(toc-tic, 'to trace critical locus')
        df = pandas.DataFrame(curveJSON)
        df['z0 / mole frac.'] = df['rho0 / mol/m^3']/(df['rho0 / mol/m^3']+df['rho1 / mol/m^3'])
        plt.plot(df['z0 / mole frac.'], df['p / Pa'], lw=2)

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
    plot_critical_curve()
    plot_isotherms()

    plt.gca().set(xlabel='$x_1$ / mole frac.', ylabel='$p$ / Pa')
    plt.tight_layout(pad=0.2)
    plt.savefig('pxdiagram.pdf')
    plt.show()

def main(names):
    T = 300

    backend = 'HEOS'
    tracer = vle.VLEIsolineTracer(IMPOSED_T, T, backend, names)
    # Set flags
    tracer.set_allowable_error(1e-6)
    tracer.polishing(True)
    tracer.set_debug_polishing(False)

    tracer.set_forwards_integration(True)
    tracer.set_unstable_termination(False)
    tracer.set_stepping_variable(vle.VLEIsolineTracer.stepping_variable.STEP_IN_RHO1)

    tracer.trace()
    data = tracer.get_tracer_data()
    print(tracer.get_termination_reason())
    print(tracer.get_tracing_time())
    
    fig, ax4 = plt.subplots(1,1,figsize=(6, 4))
    x = np.array(data.x).T[0]
    x1 = np.array(data.x).T[1]
    y = np.array(data.y).T[0]
    pL = np.array(data.pL)
    rhoLmat = np.array(data.rhoL)
    rhoVmat = np.array(data.rhoV)
    # ax4.plot(x, rhoLmat[:,0],color='b')
    # ax4.plot(x, rhoLmat[:,1],color='r')
    # ax4.plot(x, rhoVmat[:,0],dashes = [2,2], color='b')
    # ax4.plot(x, rhoVmat[:,1],dashes = [2,2], color='r')
    dx0dpL = np.diff(x)/np.diff(pL)
    dx1dpL = np.diff(x1)/np.diff(pL)
    ax4.plot(x, pL, dashes=[2,2])
    ax4.plot(y, pL, dashes=[2,2])
    
    model = teqp.build_multifluid_model(names, '../mycp', '../mycp/dev/mixtures/mixture_binary_pairs.json')

    # For now, using the EOS directly to get started... (ultimately either superancillary or extrapolation from critical)
    AS = CP.AbstractState('HEOS', names[1])
    AS.update(CP.QT_INPUTS, 0, T)
    rhoL = AS.saturated_liquid_keyed_output(CP.iDmolar)
    rhoV = AS.saturated_vapor_keyed_output(CP.iDmolar)
    rhovec = np.array([0, rhoL, 0, rhoV])

    # Initial slopes of dpdx1 and dpdy1
    for m in [model]:
        drhodp_liq, drhodp_vap = get_drhovecdp_sat(m, T, rhovec[0:2], rhovec[2::])
        dpdxL = 1.0/get_dxdp(rhovec=rhovec[0:2], drhovecdp=drhodp_liq)
        dpdxV = 1.0/get_dxdp(rhovec=rhovec[2::], drhovecdp=drhodp_vap)
        xx = np.linspace(0, 0.4)
        plt.plot(xx, xx*dpdxL[0]+AS.p(), dashes=[2, 2])
        plt.plot(xx, xx*dpdxV[0]+AS.p(), dashes=[2, 2])

    traceT(model, T, rhovec.copy())

    # for f in np.linspace(0.9, 1.1, 100):
    #     modeltweak = teqp.build_BIPmodified(model, {'betaT': 1.0, 'gammaT': 1.001633952*f, 'betaV': 1.0, 'gammaV': 1.006268954})
    #     try:
    #         traceT(modeltweak, T, rhovec.copy(), kwargs=dict(lw=0.5, dashes=[3,1,1,1]))
    #     except:
    #         pass

    for increment in np.linspace(-120,60,30):
        T2 = T+increment
        try:
            AS = CP.AbstractState('HEOS', names[1])
            AS.update(CP.QT_INPUTS, 0, T2)
            rhoL = AS.saturated_liquid_keyed_output(CP.iDmolar)
            rhoV = AS.saturated_vapor_keyed_output(CP.iDmolar)
            rhovec = np.array([0, rhoL, 0, rhoV])
            traceT_arclength(model, T2, rhovec.copy(), kwargs={'lw':0.5})
        except BaseException as BE:
            print(T2, BE)

    tic = timeit.default_timer()
    Tcvec = model.get_Tcvec()
    rhocvec = 1/model.get_vcvec()
    T0 = Tcvec[1]
    rho0 = rhocvec*np.array([1,0])
    curveJSON = teqp.trace_critical_arclength_binary(model, T0, rho0, "")
    toc = timeit.default_timer()
    print(toc-tic, 'to trace critical locus')

    df = pandas.DataFrame(curveJSON)
    df['z0 / mole frac.'] = df['rho0 / mol/m^3']/(df['rho0 / mol/m^3']+df['rho1 / mol/m^3'])
    plt.plot(df['z0 / mole frac.'], df['p / Pa'], lw=2)

    # plt.ylim(0.00019e6, 0.026e6)
    plt.gca().set(xlabel='$x_1$ / mole frac.', ylabel='$p$ / Pa')
    plt.tight_layout(pad=0.2)
    plt.show()

def trace_isopleth(names):
    AS = CP.AbstractState('HEOS', '&'.join(names))
    AS.set_mole_fractions([0.25, 0.75])
    AS.build_phase_envelope("")
    PE = AS.get_phase_envelope_data()

    model = teqp.build_multifluid_model(names, '../mycp', '../mycp/dev/mixtures/mixture_binary_pairs.json')
    i = 0
    rhovec0 = np.array([
        PE.rhomolar_liq[i]*PE.x[0][i], PE.rhomolar_liq[i]*PE.x[1][i], 
        PE.rhomolar_vap[i]*PE.y[0][i], PE.rhomolar_vap[i]*PE.y[1][i]])
    T = PE.T[i]
    dT = 1e-1
    R = model.get_R(np.array([1.0]))
    rhovec = rhovec0.copy()

    def rhovecprime(T, rhovec):
        return drhovecdT_isopleth(model=model, T=T, rhovec=rhovec)
    tic = timeit.default_timer()
    Tmax = 430
    sol = solve_ivp(rhovecprime, [T, Tmax], y0=rhovec0.copy(), method='RK45',dense_output=True, t_eval = np.linspace(T,Tmax,1000))
    p = [teqp.get_pr(model, T, sol.y[0:2,j]) + R*T*sol.y[0:2,j].sum() for T,j in zip(sol.t,range(len(sol.t)))]
    x = sol.y[0,:]/(sol.y[0,:]+sol.y[1,:])
    y = sol.y[2,:]/(sol.y[2,:]+sol.y[3,:])
    toc = timeit.default_timer()
    plt.plot(sol.t, p)
    print(toc-tic)

    # plt.plot(PE.T, PE.rhomolar_vap)
    # plt.plot(PE.T, PE.rhomolar_liq)
    plt.plot(PE.T, PE.p)
    plt.show()

if __name__ == '__main__':
    # main(names)
    # trace_isopleth(names)

    # names = ['Methane', 'n-Propane']
    # model = teqp.build_multifluid_model(names, '../mycp', '../mycp/dev/mixtures/mixture_binary_pairs.json')
    # puremodel = teqp.build_multifluid_model(['n-Propane'], '../mycp', '../mycp/dev/mixtures/mixture_binary_pairs.json')
    # prettyPX(model, puremodel, np.linspace(180, 365, 100))

    names = ['n-Butane', 'HydrogenSulfide']
    model = teqp.build_multifluid_model(names, '../mycp', '../mycp/dev/mixtures/mixture_binary_pairs.json',{'estimate':'linear'})
    puremodels = [teqp.build_multifluid_model([name], '../mycp', '../mycp/dev/mixtures/mixture_binary_pairs.json') for name in names]
    prettyPX(model, puremodels, 0, np.linspace(230, 420, 100))