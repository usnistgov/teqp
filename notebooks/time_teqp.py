import timeit
import sys
sys.path.append('bld/Release')
import teqp

import numpy as np
import matplotlib.pyplot as plt
import pandas
import scipy.optimize

def build_models():
    return [
        teqp.PCSAFTEOS(['Methane']),
        teqp.vdWEOS([150.687], [4863000.0])
    ]

def time(*, model, n, Nrep):
    molefrac = [1.0]
    rho = 3.0
    T = 300

    f = getattr(teqp, f"get_Ar0{n}n") if n > 2 else getattr(teqp, f"get_Ar0{n}")
    tic = timeit.default_timer()
    for i in range(Nrep):
        f(model, T, rho, molefrac)
    toc = timeit.default_timer()
    elap = (toc-tic)/Nrep
    return elap

def timeall(*, models, Nrep):
    o = []
    for model in models:
        for n in [1,2,3,4,5,6]:
            t = time(model=model, n=n, Nrep=Nrep)
            o.append({'model': str(model), 'n': n, 't / s': t})
    df = pandas.DataFrame(o)
    for model,gp in df.groupby('model'):
        plt.plot(gp['n'], gp['t / s']*1e6, label=model)
    plt.gca().set(xlabel='n', ylabel=r't / $\mu$s')
    plt.legend(loc='best')
    # plt.xscale('log')
    plt.yscale('log')
    plt.title(r'Timing of $A^{\rm r}_{0n}$')
    plt.show()

if __name__ == '__main__':
    timeall(models=build_models(), Nrep= 10000)