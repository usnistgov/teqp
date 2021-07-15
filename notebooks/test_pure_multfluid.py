import glob, os
import numpy as np

import CoolProp.CoolProp as CP 

import teqp

for jname in glob.glob('../mycp/dev/fluids/*.json'):
    name = os.path.split(jname)[1].split('.')[0]
    try:
        model = teqp.build_multifluid_model([name], '../mycp', '../mycp/dev/mixtures/mixture_binary_pairs.json',{'estimate':'linear'})
        Tc = model.get_Tcvec()[0]
        rhoc = 1/model.get_vcvec()[0]
        z = np.array([1.0])
        # print(Tc, rhoc)
        diff = model.get_Ar00(Tc+10, rhoc, z) - CP.PropsSI('alphar', 'T', Tc+10, 'Dmolar', rhoc, name)

        if abs(diff) > 1e-13:
            print(name, diff, rhoc, Tc+10, model.get_Ar00(Tc+10, rhoc, z), CP.PropsSI('alphar','T',Tc+10, 'Dmolar',rhoc,name))
    except BaseException as BE:
        print(name, BE)