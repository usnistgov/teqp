import timeit
import sys
sys.path.append('../bld/Release')
import teqp

import numpy as np
import matplotlib.pyplot as plt
import pandas
import scipy.optimize

def trace():
    model = teqp.build_multifluid_model(["R23", "R1234YF"], '../mycp', '../mycp/dev/mixtures/mixture_binary_pairs.json')
    Tcvec = model.get_Tcvec()
    rhocvec = 1/model.get_vcvec()
    
    # pc_Pa = []
    # for i in range(2):
    #     z = [0,0]; z[i] = 1
    #     z = np.array(z)
    #     p = 8.314462618*Tcvec[i]*rhocvec[i]*(1+model.get_Ar01(Tcvec[i], rhocvec[i]*z))
    #     pc_Pa.append(p)
    # model = teqp.vdWEOS(Tcvec, pc_Pa)
    # print(pc_Pa)

    i = 1
    T0 = Tcvec[i]
    z = np.array([0, 0]); z[i] = 1
    rho0 = z*rhocvec[i]
    curveJSON = teqp.trace_critical_arclength_binary(model, T0, rho0, "")
    df = pandas.DataFrame(curveJSON)
    df['z0 / mole frac.'] = df['rho0 / mol/m^3']/(df['rho0 / mol/m^3']+df['rho1 / mol/m^3'])
    print(df['z0 / mole frac.'])
    plt.plot(df['z0 / mole frac.'], df['T / K'])
    plt.show()
    
    plt.plot(df['T / K'], df['p / Pa'])
    plt.show()

if __name__ == '__main__':
    trace()