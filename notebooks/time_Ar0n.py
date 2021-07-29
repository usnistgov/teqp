import os
import timeit
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas
import scipy.optimize

import teqp

water = {
    "a0i / Pa m^6/mol^2": 0.12277 , "bi / m^3/mol": 0.000014515, "c1": 0.67359, 
    "Tc / K": 647.096, "epsABi / J/mol": 16655.0, "betaABi": 0.0692, "class": "4C"
}
j = {"cubic": "SRK", "pures": [water], "R_gas / J/mol/K": 8.3144598}
mH2O = teqp.CPAfactory(j)

def build_models():
    return [
        teqp.PCSAFTEOS(['Methane']),
        teqp.vdWEOS([150.687], [4863000.0]),
        teqp.vdWEOS1(1, 2),
        mH2O,
        teqp.build_multifluid_model(["Methane"], '../mycp', '../mycp/dev/mixtures/mixture_binary_pairs.json')
    ]

import CoolProp.CoolProp as CP
def time_CoolProp(n, Nrep, backend):
    AS = CP.AbstractState(backend, 'METHANE')
    AS.specify_phase(CP.iphase_gas)
    if n == 0:
        f = lambda AS: AS.alphar()
    elif n == 1:
        f = lambda AS: AS.delta()*AS.dalphar_dDelta()
    elif n == 2:
        f = lambda AS: AS.delta()**2*AS.d2alphar_dDelta2()
    elif n == 3:
        f = lambda AS: AS.delta()**3*AS.d3alphar_dDelta3()
    else:
        raise ValueError(n)

    T = 300
    rho = 3.0
    tic = timeit.default_timer()
    for i in range(Nrep):
        AS.update(CP.DmolarT_INPUTS, rho+1e-3*i, T)
        f(AS)
    toc = timeit.default_timer()
    elap = (toc-tic)/Nrep
    return elap

from ctREFPROP.ctREFPROP import REFPROPFunctionLibrary
def time_REFPROP(n, Nrep):
    root = os.environ['RPPREFIX']
    RP = REFPROPFunctionLibrary(root)
    RP.SETFLUIDSdll('METHANE')
    if n == 0:
        f = lambda D,T: RP.REFPROP1dll(D,T)
    elif n == 1:
        f = lambda AS: AS.delta()*AS.dalphar_dDelta()
    elif n == 2:
        f = lambda AS: AS.delta()**2*AS.d2alphar_dDelta2()
    elif n == 3:
        f = lambda AS: AS.delta()**3*AS.d3alphar_dDelta3()
    else:
        raise ValueError(n)

    T = 300
    rho = 3.0
    z = [1.0]
    tic = timeit.default_timer()
    for i in range(Nrep):
        RP.PRESSdll(T, rho/1e3, z)
    toc = timeit.default_timer()
    elap = (toc-tic)/Nrep
    return elap

def time(*, model, n, Nrep, use_gen):
    T = 300
    rho = 3.0
    molefrac = np.array([1.0])

    f = getattr(model, f"get_Ar0{n}n") if use_gen else getattr(model, f"get_Ar0{n}")
    for i in range(Nrep):
        f(T, rho, molefrac)
    tic = timeit.default_timer()
    for i in range(Nrep):
        f(T, rho, molefrac)
    toc = timeit.default_timer()
    elap = (toc-tic)/Nrep
    return elap

def time_virials(*, Ncomp, Nrep):
    T = 300
    molefrac = np.array([1.0])

    model = teqp.vdWEOS(
        np.linspace(150.687, 160, Ncomp).tolist(), 
        np.linspace(4863000.0, 4.9e6, Ncomp).tolist()
    )

    f = getattr(model, f"get_B2vir")
    # Warm up the core with some useless calls
    for i in range(Nrep):
        f(T, molefrac)
    # Do the calculations
    tic = timeit.default_timer()
    for i in range(Nrep):
        f(T, molefrac)
    toc = timeit.default_timer()
    elap = (toc-tic)/Nrep
    return elap

for Ncomp in np.arange(1, 20, 1):
    print(Ncomp, time_virials(Nrep=10000, Ncomp=Ncomp))

def timeall(*, models, Nrep):
    o = []
    for use_gen in [True,False]:
        for model in models:
            for n in [0,1,2,3,4,5]:
                try:
                    t = time(model=model, n=n, Nrep=Nrep, use_gen=use_gen)
                    o.append({'model': str(model), 'n': n, 't / s': t, 't / us': t*1e6, 'all': use_gen})
                except AttributeError as ae:
                    # print(ae)
                    pass
    df = pandas.DataFrame(o)
    for model, ggp in df.groupby('model'):
        for all_, gp in ggp.groupby('all'):
            modname = str(model).split(' object')[0].rsplit('teqp.',1)[-1]
            plt.plot(gp['n'], gp['t / us'], label=modname if all_ else '', dashes=[] if all_ else [3,1,1,1])

    # o = []
    # backends = ['PR', 'SRK', 'HEOS','REFPROP']
    # for backend in backends:
    #     for n in [0,1,2,3]:
    #         t = time_CoolProp(backend=backend, n=n, Nrep=Nrep)
    #         o.append({'backend': 'CoolProp:'+backend, 'n': n, 't / s': t, 't / us': t*1e6})
    # df = pandas.DataFrame(o)
    # for model,gp in df.groupby('backend'):
    #     modname = str(model).split(' object')[0].rsplit('teqp.',1)[-1]
    #     plt.plot(gp['n'], gp['t / us'], label=modname, dashes=[2,2])

    # o = []
    # for n in [0,1,2,3]:
    #     t = time_REFPROP(n=n, Nrep=Nrep)
    #     o.append({'backend': 'REFPROP(ct)', 'n': n, 't / s': t, 't / us': t*1e6})
    # df = pandas.DataFrame(o)
    # for model,gp in df.groupby('backend'):
    #     modname = str(model).split(' object')[0].rsplit('teqp.',1)[-1]
    #     plt.plot(gp['n'], gp['t / us'], label=modname, dashes=[3,1,5,1])

    plt.axhline(1.3, lw=3)

    plt.gca().set(xlabel='n', ylabel=r't / $\mu$s')
    plt.legend(loc='best')
    plt.yscale('log')
    plt.title(r'Timing of $A^{\rm r}_{0n}$')
    plt.savefig('timing of Aron.pdf')
    plt.close()

if __name__ == '__main__':

    timeall(models=build_models(), Nrep= 10000)

    def time_overhead(x):
        N = 1000
    
        tic = timeit.default_timer()
        for i in range(N):
            teqp.___mysummerref(3, x)
        toc = timeit.default_timer()
        print((toc-tic)/N*1e6, 'us/call w/ ref')

        tic = timeit.default_timer()
        for i in range(N):
            teqp.___mysummer(3, x)
        toc = timeit.default_timer()
        print((toc-tic)/N*1e6, 'us/call w/ copy')

    molefrac1 = np.array([1.0])
    molefrac2 = [1.0]
    time_overhead(molefrac1)
    # time_overhead(molefrac2)