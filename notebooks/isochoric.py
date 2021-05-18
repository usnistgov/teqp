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

def traceT(model, T, rhovec0):
    R = model.get_R(np.array([0.0, 1.0]))

    # rhovec = rhovec0.copy()
    # x = 0; dx = 1e-3
    # o = []
    # tic = timeit.default_timer()
    # while x < 0.5:
    #     pr = teqp.get_pr(model, T, rhovec[0:2])
    #     molefrac = rhovec[0:2]/rhovec[0:2].sum()
    #     p = pr + R*T*rhovec[0:2].sum()
    #     f1 = get_drhovecdx(i=0, model=model, T=T, rhovec=rhovec)
    #     ypred = rhovec + dx*f1
    #     f2 = get_drhovecdx(i=0, model=model, T=T, rhovec=ypred)
    #     favg = (f1+f2)/2.0
    #     increment = dx*favg
    #     x += dx
    #     rhovec += increment
    #     o.append(dict(x=x, p=p, y=rhovec[2]/rhovec[2:4].sum()))
    # toc = timeit.default_timer()
    # print(toc-tic, 'seconds with python+teqp tracing')
    # df = pandas.DataFrame(o)
    # plt.plot(df.x, df.p, 'k.', ms=1)
    # plt.plot(df.y, df.p, 'k.', ms=1)

    def rhovecprime(x, rhovec):
        return get_drhovecdx(i=0, model=model, T=T, rhovec=rhovec)

    tic = timeit.default_timer()
    sol = solve_ivp(rhovecprime, [0.0, 1.0], y0=rhovec0, method='RK45')
    p = [teqp.get_pr(model, T, sol.y[0:2,j]) + R*T*sol.y[0:2,j].sum() for j in range(len(sol.t))]
    toc = timeit.default_timer()
    print(toc-tic)
    print(sol.nfev)
    plt.plot(sol.t, p, 'r')

    plt.ylim(0.00019e6, 0.026e6)
    plt.gca().set(xlabel='$x_1$ / mole frac.', ylabel='$p$ / Pa')
    plt.tight_layout(pad=0.2)
    plt.show()

def main():
    T = 300
    names = ['n-Hexane', 'n-Octane']

    backend = 'HEOS'
    tracer = vle.VLEIsolineTracer(IMPOSED_T, T, backend, names)
    # Set flags
    tracer.set_allowable_error(1e-6)
    tracer.polishing(False)
    tracer.set_debug_polishing(True)

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
    ax4.plot(x, pL)
    ax4.plot(y, pL, dashes=[2, 2])
    
    model = teqp.build_multifluid_model(names, '../mycp', '../mycp/dev/mixtures/mixture_binary_pairs.json')

    # For now, using the EOS directly to get started... (ultimately either superancillary or extrapolation from critical)
    AS = CP.AbstractState('HEOS', names[1])
    AS.update(CP.QT_INPUTS, 0, T)
    rhoL = AS.saturated_liquid_keyed_output(CP.iDmolar)
    rhoV = AS.saturated_vapor_keyed_output(CP.iDmolar)
    rhovec = np.array([0, rhoL, 0, rhoV])

    # Initial slopes of dpdx1 and dpdy1
    drhodp_liq, drhodp_vap = get_drhovecdp_sat(model, T, rhovec[0:2], rhovec[2::])
    dpdxL = 1.0/get_dxdp(rhovec=rhovec[0:2], drhovecdp=drhodp_liq)
    dpdxV = 1.0/get_dxdp(rhovec=rhovec[2::], drhovecdp=drhodp_vap)
    xx = np.linspace(0, 0.4)
    plt.plot(xx, xx*dpdxL[0]+AS.p(), dashes=[2, 2])
    plt.plot(xx, xx*dpdxV[0]+AS.p(), dashes=[2, 2])

    traceT(model, T, rhovec)


if __name__ == '__main__':
    main()