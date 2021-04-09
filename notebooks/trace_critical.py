import timeit
import sys
sys.path.append('../bld/Release')
import teqp

import numpy as np
import matplotlib.pyplot as plt
import pandas
import scipy.optimize

def trace():
    model = teqp.build_multifluid_model(["Methane", "Ethane"], '../mycp', '../mycp/dev/mixtures/mixture_binary_pairs.json')
    Tcvec = model.get_Tcvec()
    rhocvec = 1/model.get_vcvec()
    models = [model]
    
    pc_Pa = []
    for i in range(2):
        z = [0,0]; z[i] = 1
        z = np.array(z)
        p = 8.314462618*Tcvec[i]*rhocvec[i]*(1+model.get_Ar01(Tcvec[i], rhocvec[i]*z))
        pc_Pa.append(p)
    model = teqp.vdWEOS(Tcvec, pc_Pa)
    models.append(model)

    i = 1
    T0 = Tcvec[i]
    z = np.array([0, 0]); z[i] = 1
    if isinstance(model, teqp.vdWEOS):
        rho0 = z*pc_Pa[i]/(8.314462618*Tcvec[i])*8/3
    else:
        rho0 = z*rhocvec[i]

    fig1, ax1 = plt.subplots(1,1)
    fig2, ax2 = plt.subplots(1,1)
    fig3, ax3 = plt.subplots(1,1)
    for model in models:
        tic = timeit.default_timer()
        curveJSON = teqp.trace_critical_arclength_binary(model, T0, rho0, "")
        toc = timeit.default_timer()
        print(toc-tic, 's')
        df = pandas.DataFrame(curveJSON)
        df['z0 / mole frac.'] = df['rho0 / mol/m^3']/(df['rho0 / mol/m^3']+df['rho1 / mol/m^3'])
        ax1.plot(df['z0 / mole frac.'], df['T / K'])
        ax2.plot(df['T / K'], df['p / Pa'])
        ax3.plot(df['z0 / mole frac.'], df['rho0 / mol/m^3']+df['rho1 / mol/m^3'])
    plt.show()

if __name__ == '__main__':
    trace()