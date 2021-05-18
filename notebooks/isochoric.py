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

def traceT(model, T, rhovec0, *, kwargs={}):
    R = model.get_R(np.array([0.0, 1.0]))
    def rhovecprime(x, rhovec):
        return get_drhovecdx(i=0, model=model, T=T, rhovec=rhovec)
    tic = timeit.default_timer()
    sol = solve_ivp(rhovecprime, [0.0, 0.5], y0=rhovec0, method='RK45',dense_output=True, t_eval = np.linspace(0,0.49,1000), rtol=1e-8)
    p = [teqp.get_pr(model, T, sol.y[0:2,j]) + R*T*sol.y[0:2,j].sum() for j in range(len(sol.t))]
    toc = timeit.default_timer()
    print(toc-tic)
    y = sol.y[2,:]/(sol.y[2,:]+sol.y[3,:])
    plt.plot(sol.t, p, 'r',**kwargs)
    plt.plot(y, p, 'g',**kwargs)

def drhovecdT_oldschool(model,T, rhovec, fluids):
    rhovecL = rhovec[0:2]
    rhovecV = rhovec[2::]
    tracer = vle.VLEIsolineTracer(vle.VLEIsolineTracer.imposed_variable.IMPOSED_T, 20000, 'HEOS', fluids)
    def get_derivs(T, rhovec):
        return tracer.get_derivs(T,rhovec)#vle.VLEIsolineTracer(vle.VLEIsolineTracer.imposed_variable.IMPOSED_T, 20000, 'HEOS', fluids).get_derivs(T, rhovec)
    
    derV = get_derivs(T, rhovecV)
    derL = get_derivs(T, rhovecL)
    # dT = 1e-3
    # print(derL.dpdT__constrhovec(), (get_derivs(T+dT, rhovecV).p() - get_derivs(T-dT, rhovecV).p())/(2*dT))
    AS = tracer.get_AbstractState_pointer()
    R = AS.gas_constant()
    # AS.set_mole_fractions(z)
    z = rhovecL/rhovecL.sum()

    rhoV = sum(rhovecV)
    rhoL = sum(rhovecL)
    DELTAs2 = np.array([derV.d2psir_dTdrhoi__constrhoj(i) - derL.d2psir_dTdrhoi__constrhoj(i) + R*np.log(rhovecV[i]/rhoV/(rhovecL[i]/rhoL)) for i in range(2)]) # Column vector
    DELTAs = np.array([derV.d2psir_dTdrhoi__constrhoj(i) - derL.d2psir_dTdrhoi__constrhoj(i) + R*np.log(rhovecV[i]/rhovecL[i]) for i in range(2)]) # Column vector
    DELTAbetarho = derV.dpdT__constrhovec() - derL.dpdT__constrhovec() # double
    PSIL = derL.get_Hessian()
    PSIV = derV.get_Hessian()
    drhovecL_dT = (np.dot(DELTAs,rhovecV)-DELTAbetarho)/np.dot(np.dot(PSIL,rhovecV-rhovecL),z)*np.array(z)
    drhovecV_dT = np.linalg.solve(PSIV, np.dot(PSIL,drhovecL_dT)-DELTAs)
    return np.array(drhovecL_dT.tolist()+drhovecV_dT.tolist())

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
    R = 8.31446161815324
    deltas = teqp.get_dchempotdT_autodiff(model, T, rhovecV) - teqp.get_dchempotdT_autodiff(model, T, rhovecL)
    deltarho = rhovecV-rhovecL
    dpdTV = R*rhoV*(1+model.get_Ar01(T, rhoV, xV)-model.get_Ar11(T, rhoV, xV))
    dpdTL = R*rhoL*(1+model.get_Ar01(T, rhoL, xL)-model.get_Ar11(T, rhoL, xL))
    deltabeta = dpdTV-dpdTL
    drhovecdTL = (np.dot(deltas,rhovecV)-deltabeta)/np.dot(Hliq@deltarho, xL)*xL
    drhovecdTV = np.linalg.solve(Hvap, Hliq@drhovecdTL-deltas)
    return np.array(drhovecdTL.tolist()+drhovecdTV.tolist())

def main(names):
    T = 300

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
    for m in [model]:
        drhodp_liq, drhodp_vap = get_drhovecdp_sat(m, T, rhovec[0:2], rhovec[2::])
        dpdxL = 1.0/get_dxdp(rhovec=rhovec[0:2], drhovecdp=drhodp_liq)
        dpdxV = 1.0/get_dxdp(rhovec=rhovec[2::], drhovecdp=drhodp_vap)
        xx = np.linspace(0, 0.4)
        plt.plot(xx, xx*dpdxL[0]+AS.p(), dashes=[2, 2])
        plt.plot(xx, xx*dpdxV[0]+AS.p(), dashes=[2, 2])

    traceT(model, T, rhovec.copy())

    for f in np.linspace(0.9, 1.1, 100):
        modeltweak = teqp.build_BIPmodified(model, {'betaT': 1.0, 'gammaT': 1.001633952*f, 'betaV': 1.0, 'gammaV': 1.006268954})
        try:
            traceT(modeltweak, T, rhovec.copy(), kwargs=dict(lw=0.5, dashes=[3,1,1,1]))
        except:
            pass

    plt.ylim(0.00019e6, 0.026e6)
    plt.gca().set(xlabel='$x_1$ / mole frac.', ylabel='$p$ / Pa')
    plt.tight_layout(pad=0.2)
    plt.show()

def trace_isopleth(names):
    AS = CP.AbstractState('HEOS', '&'.join(names))
    AS.set_mole_fractions([0.25, 0.75])
    AS.build_phase_envelope("")
    PE = AS.get_phase_envelope_data()
    print(dir(PE))

    model = teqp.build_multifluid_model(names, '../mycp', '../mycp/dev/mixtures/mixture_binary_pairs.json')
    i = 0
    rhovec0 = np.array([
        PE.rhomolar_liq[i]*PE.x[0][i], PE.rhomolar_liq[i]*PE.x[1][i], 
        PE.rhomolar_vap[i]*PE.y[0][i], PE.rhomolar_vap[i]*PE.y[1][i]])
    T = PE.T[0]
    dT = 1e-2
    R = model.get_R(np.array([1.0]))
    rhovec = rhovec0.copy()
    for i in range(10000):
        xL0 = rhovec[0]/rhovec[0:2].sum()
        molefrac = np.array([xL0, 1-xL0])
        rhovecL = rhovec[0:2]
        rhoL = rhovecL.sum()
        rhovecV = rhovec[2::]
        rhoV = rhovecV.sum()

        # plt.plot(T, rhoL, 'o')
        # plt.plot(T, rhoV, 'x')

        vec1 = drhovecdT_isopleth(model, T, rhovec)
        rhovec += vec1*dT
        T += dT

        rhovecL = rhovec[0:2]
        rhoL = rhovecL.sum()
        p = rhoL*R*T + teqp.get_pr(model, T, rhovecL)
        print(T, xL0, rhoL)
        plt.plot(T, p, 'x')

    # plt.plot(PE.T, PE.rhomolar_vap)
    # plt.plot(PE.T, PE.rhomolar_liq)
    plt.plot(PE.T, PE.p)
    plt.show()

if __name__ == '__main__':
    names = ['n-Hexane', 'n-Octane']
    # main(names)
    trace_isopleth(names)