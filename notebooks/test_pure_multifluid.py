import glob, os, json
import numpy as np

import CoolProp.CoolProp as CP 
CP.set_config_string(CP.ALTERNATIVE_REFPROP_PATH, r'C:\Users\ihb\Code\REFPROP-cmake\bld\Release')
import teqp

root = teqp.get_datapath()
for jname in glob.glob(root+'/dev/fluids/*.json'):
    name = os.path.split(jname)[1].split('.')[0]
    try:
        model = teqp.build_multifluid_model([name], root, root+'/dev/mixtures/mixture_binary_pairs.json',{'estimate':'linear'})
        Tc = model.get_Tcvec()[0]
        rhoc = 1/model.get_vcvec()[0]
        z = np.array([1.0])
        # print(Tc, rhoc)
        diff = model.get_Ar00(Tc+10, rhoc, z) - CP.PropsSI('alphar', 'T', Tc+10, 'Dmolar', rhoc, name)
        if abs(diff) > 1e-13:
            print(name, diff, rhoc, Tc+10, model.get_Ar00(Tc+10, rhoc, z), CP.PropsSI('alphar','T',Tc+10, 'Dmolar',rhoc,name))

        def get_REFPROP_name(name):
            j = json.load(open(root+f'/dev/fluids/{name}.json'))
            return j['INFO']['REFPROP_NAME']
        RPname = get_REFPROP_name(name)
        TcRP, rhocRP = [CP.PropsSI('REFPROP::'+RPname,k) for k in ['Tcrit','rhomolar_critical']]
        # If EOS is the same according to the DOI:
        if abs(TcRP-Tc) > 1e-10:
            print(name, TcRP, Tc, '!!!!!!!! TC(RP,teqp) !!!!!!')
        if abs(rhocRP-rhoc) > 1e-10:
            print(name, rhocRP, rhoc, '!!!!!!!! RHOC(RP,teqp) !!!!!!')
            continue
        
        if RPname != 'N/A':
            diffRP = model.get_Ar00(Tc+10, rhoc, z) - CP.PropsSI('alphar', 'T', Tc+10, 'Dmolar', rhoc, 'REFPROP::'+RPname)
            if abs(diffRP) > 1e-5:
                print(TcRP, Tc)
                print(rhocRP, rhoc)
                print(name, diffRP, rhoc, Tc+10, model.get_Ar00(Tc+10, rhoc, z), CP.PropsSI('alphar','T',Tc+10, 'Dmolar',rhoc,'REFPROP::'+RPname))

    except BaseException as BE:
        print(name, BE)