import timeit
import sys
sys.path.append('../bld/Release')
import teqp

import numpy as np
import matplotlib.pyplot as plt
import pandas
import scipy.optimize

def build_models():
    return [
        teqp.PCSAFTEOS(['Methane']),
        teqp.vdWEOS([150.687], [4863000.0]),
        teqp.build_multifluid_model(["Methane"], '../mycp', '../mycp/dev/mixtures/mixture_binary_pairs.json')
    ]

def time(*, model, n, Nrep):
    molefrac = [1.0]
    rho = 3.0
    T = 300

    f = getattr(model, f"get_Ar0{n}n")# if n > 2 else getattr(teqp, f"get_Ar0{n}")
    tic = timeit.default_timer()
    for i in range(Nrep):
        f(T, rho, molefrac)
    toc = timeit.default_timer()
    elap = (toc-tic)/Nrep
    return elap

def timeall(*, models, Nrep):
    o = []
    for model in models:
        for n in [1,2,3,4,5,6]:
            t = time(model=model, n=n, Nrep=Nrep)
            o.append({'model': str(model), 'n': n, 't / s': t, 't / us': t*1e6})
    df = pandas.DataFrame(o)    
    for model,gp in df.groupby('model'):
        modname = str(model).split(' object')[0].rsplit('teqp.',1)[-1]
        plt.plot(gp['n'], gp['t / us'], label=modname)
    plt.gca().set(xlabel='n', ylabel=r't / $\mu$s')
    plt.legend(loc='best')
    plt.yscale('log')
    plt.title(r'Timing of $A^{\rm r}_{0n}$')
    plt.show()

if __name__ == '__main__':
    timeall(models=build_models(), Nrep= 1000)