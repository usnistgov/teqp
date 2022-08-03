from math import log
import numpy as np
import teqp

c0 = 4
a1 = -6.59406093943886; a2 = 5.60101151987913
Tcrit = 405.56; rhocrit = 13696.0
n = [ 2.224, 3.148, 0.9579 ]
theta = [ 1646, 3965, 7231 ]

jNH3 = [
    {"type": "Lead", "a_1": a1 - log(rhocrit), "a_2": a2 * Tcrit},
    {"type": "LogT",  "a": -(c0 - 1)},
    {"type": "Constant", "a": (c0 - 1) * log(Tcrit)}, # Term from ln(tau)
    {"type": "PlanckEinstein", "n": n, "theta": theta}
]

a0 = teqp.IdealHelmholtz([jNH3])

T = 300; rho = 10
molefrac = np.array([1.0])
print(dir(a0))
print((-a0.get_Aig20(T, rho, molefrac)+1)*8.314462618153254)
import CoolProp.CoolProp as CP
print(CP.PropsSI('Cp0molar','T',T,'Dmolar',1e-10,'Ammonia'))
print(CP.PropsSI('Cp0molar','T',T,'Dmolar',1e-10,'REFPROP::Ammonia'))
