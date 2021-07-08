import sys
import teqp

import numpy as np

import CoolProp.CoolProp as CP

def crit_density_guess(model, Tc, rhoc):
    """
    """
    R = 8.31446261815324
    z = np.array([1.0])

    ders = model.get_Ar04n(Tc, rhoc, z)
    dpdrho = R*Tc*(1 + 2*ders[1] + ders[2])
    d2pdrho2 = R*Tc/rhoc*(2*ders[1] + 4*ders[2] + ders[3])
    d3pdrho3 = R*Tc/rhoc**2*(6*ders[2] + 6*ders[3] + ders[4])

    def fd2pdrho2(rho):
        ders = model.get_Ar04n(Tc, rho, z)
        return 1/rhoc*(2*ders[1] + 4*ders[2] + ders[3])
    drho = 0.001*rhoc
    d3pdrho3chk = (fd2pdrho2(rhoc+drho)-fd2pdrho2(rhoc-drho))/(2*drho)
    print(d3pdrho3, d3pdrho3chk)
    print(rhoc)

    Ar11 = model.get_Ar11(Tc, rhoc, z)
    Ar12 = model.get_Ar12(Tc, rhoc, z)
    d2pdrhodT = R*(1 + 2*ders[1] + ders[2] - 2*Ar11 - Ar12)

    def fd2pdrho2B(T):
        ders = model.get_Ar04n(T, rhoc, z)
        return Tc*R*(1 + 2*ders[1] + ders[2])
    dT = 0.00001*Tc
    d2pdrhodTchk = (fd2pdrho2B(Tc+dT)-fd2pdrho2B(Tc))/(dT)
    print(d2pdrhodT, d2pdrhodTchk*1e6/1e6, 'p_1rho1T / MPa cm3/(mol K)')
    
    # AS = CP.AbstractState('HEOS', 'PROPANE')
    # AS.update(CP.DmolarT_INPUTS, rhoc, Tc)
    # print(AS.dalphar_dDelta(), ders[1], 'Ar01')
    # print(AS.d2alphar_dDelta2(), ders[2], 'Ar02')
    # print(AS.d3alphar_dDelta3(), ders[3], 'Ar03')
    # print(AS.d4alphar_dDelta4(), ders[4], 'Ar04')
    # print(AS.d2alphar_dDelta_dTau(), Ar11, 'Ar11')
    # print(AS.d3alphar_dDelta2_dTau(), Ar12, 'Ar12')
    # print(AS.second_partial_deriv(CP.iP, CP.iT, CP.iDmolar, CP.iDmolar, CP.iT), 'd2pdrhodT', d2pdrhodT)

    Brho = (6*d2pdrhodT*Tc/d3pdrho3)**0.5

    drhohat_dT = Brho/Tc
    dT = 0.001*Tc
    T = Tc-dT
    drhohat = dT*drhohat_dT
    rho1 = drhohat/(1-T/Tc)**0.5 + rhoc
    rho2 = -drhohat/(1-T/Tc)**0.5 + rhoc
    rholiq = max(rho1, rho2)
    rhovap = min(rho1, rho2)

    return T, rholiq, rhovap, Brho

model = teqp.build_multifluid_model(['n-Propane'], '../mycp', '../mycp/dev/mixtures/mixture_binary_pairs.json')
Tc = model.get_Tcvec()[0]
vc = model.get_vcvec()[0]
print(Tc, 1/vc)

T, rhoL, rhoV, slope = crit_density_guess(model, Tc, 1/vc)
# print(CP.PropsSI('Dmolar','T',T,'Q',0,'REFPROP::PROPANE'), rhoL)
# print(CP.PropsSI('Dmolar','T',T,'Q',1,'REFPROP::PROPANE'), rhoV)

rhoc = 1/vc
import numpy as np
import matplotlib.pyplot as plt
Ts = np.linspace(0.999*Tc, Tc)
rhoL = CP.PropsSI('Dmolar','T',Ts,'Q',0,'REFPROP::PROPANE')
rhoV = CP.PropsSI('Dmolar','T',Ts,'Q',1,'REFPROP::PROPANE')

linel, = plt.plot(Ts, (rhoL-rhoc)*(1-Ts/Tc)**0.5)
linev, = plt.plot(Ts, (rhoV-rhoc)*(1-Ts/Tc)**0.5)

# Extrapolation with near-critical slope
yL = (rhoL-rhoc)*(1-Ts/Tc)**0.5
mc = (yL[-1]-yL[-4])/(Ts[-1]-Ts[-4])
mc = -slope/Tc
yy = mc*(Ts-Ts[-1]) + yL[-1]
plt.plot(Ts, yy, dashes=[2,2], color=linel.get_color())

yV = (rhoV-rhoc)*(1-Ts/Tc)**0.5
mc = (yV[-1]-yV[-4])/(Ts[-1]-Ts[-4])
mc = slope/Tc
yy = mc*(Ts-Ts[-1]) + yV[-1]
print(mc, slope)
plt.plot(Ts, yy, dashes=[2,2], color=linev.get_color())

plt.gca().set(xlabel='$T$ / K', ylabel=r'$(\rho^\alpha-\rho_{\rm crit})\sqrt{1-T/T_{\rm crit}}$')
plt.title('propane (Lemmon)')
plt.savefig('propane_critical_scaling.pdf')
plt.show()