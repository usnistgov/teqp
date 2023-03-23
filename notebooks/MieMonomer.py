import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
from collections import namedtuple

import sys
sys.path.append('../../(J) SRdilute/potentials')
import LennardJones126

Tterms = namedtuple("Tterms",['dstar'])

class MieSAFT(object):
    def __init__(self, *, naMie, nrMie):
        self.naMie = naMie
        self.nrMie = nrMie
        self.C = self.nrMie/(self.nrMie-self.naMie)*(self.nrMie/self.naMie)**(self.naMie/(self.nrMie-self.naMie))
        self.alpha = self.C*(1/(self.naMie-3) - 1/(self.nrMie-3))

    def get_dstar(self, Tstar):
        """ Eq. 7 """
        def integrand(rstar):
            return 1-np.exp(-self.C/Tstar*(rstar**(-self.nrMie)-rstar**(-self.naMie)))
        def complex_quadrature(func, a, b, **kwargs):
            def real_func(x):
                return scipy.real(func(x))
            def imag_func(x):
                return scipy.imag(func(x))
            real_integral = scipy.integrate.quad(real_func, a, b, **kwargs)
            imag_integral = scipy.integrate.quad(imag_func, a, b, **kwargs)
            return (real_integral[0] + 1j*imag_integral[0], real_integral[1:], imag_integral[1:])

        if np.isreal(Tstar):
            val, err = scipy.integrate.quad(integrand,0,1,epsabs=1e-16)
            return val
        else:
            val, err_real, err_imag = complex_quadrature(integrand,0,1,epsabs=1e-16)
            return val

    def get_Tterms(self, Tstar):
        return Tterms(self.get_dstar(Tstar))

    def get_pf_eff(self, pf, nMie):
        """ Eq. 40, 41 """
        A = np.array(
            [[0.81096,  1.7888, -37.578,   92.284],
             [1.0205,  -19.341,  151.26, -463.50],
             [-1.9057, 22.845,  -228.14,  973.92],
             [1.0885,  -6.1962,  106.98, -677.64]]) 
        r = np.array([1, 1/nMie, 1/nMie**2, 1/nMie**3],ndmin=2).T # .T to make it a column vector
        cn = np.dot(A, r)
        pfterms = np.array([pf, pf**2, pf**3, pf**4], ndmin=2).T # .T to make it a column vector
        return np.sum(cn*pfterms)

    def get_a1Sstar(self, pf, nMie):
        """ Eq. 39 """
        pf_eff = self.get_pf_eff(pf, nMie)
        return -12*pf*(1/(nMie-3))*(1-pf_eff/2)/(1-pf_eff)**3

    def get_In(self, dstar, nMie):
        """ Eq. 28 """
        x0 = 1/dstar
        return -(x0**(3-nMie)-1)/(nMie-3)

    def get_Jn(self, dstar, nMie):
        """ Eq. 29 """
        x0 = 1/dstar
        return -(x0**(4-nMie)*(nMie-3)-x0**(3-nMie)*(nMie-4)-1)/((nMie-3)*(nMie-4))

    def get_Bstar(self, pf, dstar, nMie):
        """ Eq. 33 """
        return 12*pf*((1-pf/2)/(1-pf)**3*self.get_In(dstar, nMie) - 9*pf*(1+pf)/(2*(1-pf)**3)*self.get_Jn(dstar, nMie))

    def get_a1star(self, Tstar, pf):
        Tterms = self.get_Tterms(Tstar)
        def one_term(nMie, pf, dstar):
            return dstar**(-nMie)*(self.get_a1Sstar(pf, nMie)+self.get_Bstar(pf, dstar, nMie))
        return self.C*(one_term(self.naMie, pf, Tterms.dstar) - one_term(self.nrMie, pf, Tterms.dstar))

    def get_KHS(self, pf):
        return (1-pf)**4/(1 + 4*pf + 4*pf**2 - 4*pf**3 + pf**4)

    def get_fi(self, alpha, i):
        """ Eq. 20 """
        phi = np.array([
            [7.5365557, -359.44,  1550.9, -1.19932,  -1911.28,    9236.9,   10],
            [-37.60463, 1825.6,   -5070.1, 9.063632,  21390.175, -129430,   10],
            [71.745953, -3168.0,  6534.6, -17.9482,  -51320.7,    357230,   0.57 ],
            [-46.83552, 1884.2,   -3288.7, 11.34027,  37064.54,   -315530,   -6.7],
            [-2.467982, -0.82376, -2.7171, 20.52142,  1103.742,    1390.2,   -8],
            [-0.50272,  -3.1935,  2.0883, -56.6377,  -3264.61,    -4518.2,   0],
            [8.0956883, 3.7090,   0,       40.53683,  2556.181,    4241.6,   0]
        ])
        c = phi[:,i-1]
        return sum([c[n]*alpha**n for n in range(0,4)])/(1+sum([c[n]*alpha**(n-3) for n in range(4,7)]))

    def get_chi(self, pf, dstar):
        """ Eq. 17 """
        X = pf/dstar**3
        return self.get_fi(self.alpha, 1)*X + self.get_fi(self.alpha, 2)*X**5 + self.get_fi(self.alpha, 3)*X**8

    def get_a2star(self, Tstar, pf):
        """ Eq. 36 """
        dstar = self.get_dstar(Tstar)
        x0 = 1/dstar
        chi = self.get_chi(pf, dstar)
        def term(nMie, pf, dstar):
            return x0**nMie*(self.get_a1Sstar(pf, nMie)+self.get_Bstar(pf, dstar, nMie))
        return 0.5*self.get_KHS(pf)*(1+chi)*self.C**2*(
                term(2*self.naMie, pf, dstar)
                -2*term(self.naMie+self.nrMie, pf, dstar)
                +term(2*self.nrMie, pf, dstar))

    def get_a3star(self, Tstar, pf, dstar):
        """  Eq. 19 """
        X = pf/dstar**3
        return -self.get_fi(self.alpha, 4)*X*np.exp(self.get_fi(self.alpha, 5)*X + self.get_fi(self.alpha, 6)*X**2)

    def get_alphar_monomer(self, Tstar, rhostarS):
        dstar = self.get_dstar(Tstar)
        pf = rhostarS*np.pi/6*dstar**3
        pf_HS = pf#/dstar**3
        alphar_HS = (4*pf_HS-3*pf_HS**2)/(1-pf_HS)**2
        return alphar_HS + self.get_a1star(Tstar, pf)/Tstar + self.get_a2star(Tstar, pf)/Tstar**2 + self.get_a3star(Tstar, pf, dstar)/Tstar**3

    def get_rhostardalphardrhostar(self, Tstar, rhostarS):
        # See https://sinews.siam.org/Details-Page/differentiation-without-a-difference
        h = 1e-100
        deriv1 = (self.get_alphar_monomer(Tstar, rhostarS+1j*h)/h).imag
        # drho = 1e-6*rhostarS
        # deriv2 = (self.get_alphar_monomer(Tstar, rhostarS+drho) - self.get_alphar_monomer(Tstar, rhostarS-drho))/(2*drho)
        # print(deriv1, deriv2)
        return rhostarS*deriv1

    def get_pstar(self, Tstar, rhostarS):
        # See https://sinews.siam.org/Details-Page/differentiation-without-a-difference
        h = 1e-100
        return rhostarS*Tstar*(1+self.get_rhostardalphardrhostar(Tstar, rhostarS))

    def neg_sr_over_kB(self, Tstar, rhostarS):
        # See https://sinews.siam.org/Details-Page/differentiation-without-a-difference
        h = 1e-100
        return Tstar*(self.get_alphar_monomer(Tstar+(0+1j)*h, rhostarS)/h).imag + self.get_alphar_monomer(Tstar, rhostarS)

# def VLE_calculation(nrMie, naMie, start = (1.25, 0.1, 0.55), s = 1):

#     mie = MieSAFT(naMie=naMie, nrMie=nrMie)
#     Tstar, rhoL0, rhoV0 = start
    
#     def objective(x):
#         rhoL, rhoV = x
        
#         # pstarL_Thol = LennardJones126.LJ_p(Tstar, rhoL)
#         # pstarV_Thol = LennardJones126.LJ_p(Tstar, rhoV)
#         # ArL00 = LennardJones126.get_alphar_deriv(1.32/Tstar, rhoL/0.31,0,0)
#         # ArL01 = LennardJones126.get_alphar_deriv(1.32/Tstar, rhoL/0.31,0,1)
#         # ArV00 = LennardJones126.get_alphar_deriv(1.32/Tstar, rhoV/0.31,0,0)
#         # ArV01 = LennardJones126.get_alphar_deriv(1.32/Tstar, rhoV/0.31,0,1)
#         # grL_Thol = np.log(rhoL) + ArL00 + ArL01
#         # grV_Thol = np.log(rhoV) + ArV00 + ArV01

#         # return (pstarL_Thol-pstarV_Thol)/pstarV_Thol, grL_Thol-grV_Thol

#         pstarL = mie.get_pstar(Tstar, rhoL)
#         pstarV = mie.get_pstar(Tstar, rhoV)
#         ArL00 = mie.get_alphar_monomer(Tstar, rhoL)
#         ArL01 = mie.get_rhostardalphardrhostar(Tstar, rhoL)
#         ArV00 = mie.get_alphar_monomer(Tstar, rhoV)
#         ArV01 = mie.get_rhostardalphardrhostar(Tstar, rhoV)
#         grL = np.log(rhoL) + ArL00 + ArL01
#         grV = np.log(rhoV) + ArV00 + ArV01

#         # print(Tstar, x, grL, grL_Thol, grV, grV_Thol)
#         return (pstarL-pstarV)/pstarV, grL-grV
    
#     res = scipy.optimize.fsolve(objective, [rhoL0, rhoV0])
#     print(res)
#     # return
#     TT,RL,RV = [],[],[]
#     for counter in range(0, 2000):
#         Tstar += s*0.0025
#         x0 = res
#         res = scipy.optimize.fsolve(objective, x0)
#         TT.append(Tstar)
#         RL.append(res[0])
#         RV.append(res[1])
#         if abs(res[0]/res[1]-1) < 0.05:
#             break
#         if Tstar < 0.3:
#             break
#         print(Tstar, x0, '-->', res)

#     return TT, RL, RV

# mie = MieSAFT(nrMie=12, naMie=6)
# for s in [1,-1]:
#     T, rhoL, rhoV = VLE_calculation(20, 6, (0.75, 0.78, 0.02964), s = s)
#     plt.plot(rhoL, T)
#     plt.plot(rhoV, T)
# plt.xlim(0,1)
# plt.ylim(0.5, 2)
# plt.savefig('VLE.png')
# plt.show()

def plot_a1Sstar(Tstar=1):
    """ Fig 3. """
    naMie = 6
    for nrMie in [5,12,20]:
        mie = MieSAFT(naMie=naMie, nrMie=nrMie)
        rhostarS = np.linspace(0,1)
        pf = rhostarS*np.pi/6#*mie.get_dstar(Tstar)**3
        a1Sstar = [mie.get_a1Sstar(_pf, nrMie) for _pf in pf]
        plt.plot(rhostarS, a1Sstar)
    plt.show()

    for nrMie in [20, 60, 100]:
        rhostarS = np.linspace(0,1)
        pf = rhostarS*np.pi/6#*mie.get_dstar(Tstar)**3
        a1Sstar = [mie.get_a1Sstar(_pf, nrMie) for _pf in pf]
        plt.plot(rhostarS, a1Sstar)
    plt.show()

def plot_astar(Tstar=1):
    """ Fig 3. """
    naMie = 6

    fig, axes = plt.subplots(3,1, figsize=(3,10), sharex=True)
    ax0, ax1, ax2 = axes

    for nrMie in [8,12,14,20,30]:
        mie = MieSAFT(naMie=naMie, nrMie=nrMie)
        rhostarS = np.linspace(0,1)
        dstar = mie.get_dstar(Tstar)
        pf = rhostarS*np.pi/6*dstar**3
        a1star = [mie.get_a1star(Tstar, _pf) for _pf in pf]
        ax0.plot(rhostarS, a1star)
    ax0.set_ylabel(r'$a_1^*=a_1/\epsilon^1$')
    ax0.set_xlim(0,1)
    ax0.set_ylim(-8,0)

    for nrMie in [8,12,14,20,30]:
        mie = MieSAFT(naMie=naMie, nrMie=nrMie)
        rhostarS = np.linspace(0,1)
        dstar = mie.get_dstar(Tstar) 
        pf = rhostarS*np.pi/6*dstar**3
        a1star = [mie.get_a2star(Tstar, _pf) for _pf in pf]
        ax1.plot(rhostarS, a1star)
    ax1.set_ylabel(r'$a_2^*=a_2/\epsilon^2$')
    ax1.set_xlim(0,1)
    ax1.set_ylim(-0.295,0)

    for nrMie in [8,12,14,20,30]:
        mie = MieSAFT(naMie=naMie, nrMie=nrMie)
        rhostarS = np.linspace(0,1)
        dstar = mie.get_dstar(Tstar) 
        pf = rhostarS*np.pi/6*dstar**3
        a3star = [mie.get_a3star(Tstar, _pf, dstar) for _pf in pf]
        ax2.plot(rhostarS, a3star)
    ax2.set_xlim(0,1)
    ax2.set_ylim(-0.06,0)
    ax2.set_ylabel(r'$a_3^*=a_3/\epsilon^3$')
    ax2.set_xlabel(r'$\rho*_{\rm S}=\rho_{\rm S}\sigma^3$')
    plt.tight_layout(pad=0.2)
    plt.show()

if __name__ == '__main__':
    plt.style.use('classic')
    # plot_a1Sstar()
    plot_astar()
    mie = MieSAFT(naMie=12, nrMie=6)
    print(mie.neg_sr_over_kB(2,0.3))

    Tstar = 7.377
    rhostar = 0.02
    mie = MieSAFT(naMie=9, nrMie=6)
    print(mie.get_alphar_monomer(Tstar, rhostar))