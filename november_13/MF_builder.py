#Riccardo Seppi - MPE - HEG (2019)
#This code builds and compares halo mass functions

import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from scipy import integrate
from scipy.integrate import simps

#trying to write the PS function by myself
#there are already a lot of assumptions here: for example the choice of rhoav depends on cosmology itself, this value is already an approximation (assumption)
#sigma is used as a constant, which is wrong, it is a function of mass
def PressSchechter(M,sigmam,n,z):
    rhoav=1e-26*u.kg.to(u.Msun)/(u.m.to(u.Mpc))**3*(1+z)**3
    Ms=(rhoav**(1-n/3)/2/sigmam**2)**(3/(3+n))
   # Ms=1e14
    print('rhoav=',rhoav,'Ms=',Ms)
   # N=1./np.sqrt(np.pi)*(1+n/3)*(M/Ms)**((3+n)/6)*np.exp(-(M/Ms)**(3+n)/3)
    dndM=1./np.sqrt(np.pi)*(1+n/3)*(rhoav/M**2)*(M/Ms)**((3+n)/6)*np.exp(-(M/Ms)**(3+n)/3)
    return dndM

#define PS parameters
M=np.logspace(11.0,15.0,100)
n=0.96
sigmam=0.8

#Build and plot different PS
n1=PressSchechter(M,sigmam,n,z=1)
n2=PressSchechter(M,sigmam,n,z=2)

plt.figure(figsize=(8,8))
plt.plot(M,n1,label='z=1')
plt.plot(M,n2,label='z=2')
plt.legend()
plt.title('Halo Mass Function', fontsize=25)
plt.xlabel(r'$M\ [M_\odot]$', fontsize=18)
plt.ylabel(r'$dn/dM\ [Mpc^{-3}]$', fontsize=18)
plt.xscale('log')
plt.ylim(1e-10,0.0002)
plt.yscale('log')
plt.grid(True)
plt.tight_layout()
plt.show()


#Use colossus to build Mass Functions - B.Diemer (2017)
from colossus.lss import mass_function as mf
#import inspect
#print(inspect.getsource(mass_function.massFunction))

from colossus.cosmology import cosmology
cosmology.setCosmology('planck18')
z=np.arange(0.0,4.0,0.5)
#area(n) are used to integrate the mass function to obtain the cluster count
#the integration process can be improved, it is just the sum of areas of different rectangles 
area=np.zeros(len(z))
area_simp=np.zeros(len(z)) #this will be integrated with the simpson's rule
area_tink=np.zeros(len(z)) #this will be used to integrate the tinker08 MF
area1=np.zeros(len(z))
area2=np.zeros(len(z))
area3=np.zeros(len(z))
area3_tink=np.zeros(len(z))

#Mass Function for planck18 cosmology
plt.figure()
plt.title('HMF - Planck18', fontsize=25)
plt.xlabel(r'$ Mvir\ [M_\odot]$',fontsize=18)
plt.ylabel(r'$dn/dln(M)\ [Mpc^{-3}]$',fontsize=18)
plt.ylim(1e-7, 1e-1)
plt.xscale('log')
plt.yscale('log')
for i in range(len(z)):
    mass_func=mf.massFunction(M,z[i],mdef = 'vir', model = 'comparat17', q_out = 'dndlnM')
    mass_func_tink=mf.massFunction(M,z[i],mdef = '200m', model = 'tinker08', q_out = 'dndlnM')
    plt.plot(M,mass_func, label='z=%.1f'%(z[i]))
    for j in range(len(M)-1):
        area[i]=area[i]+mass_func[j]*(M[j+1]-M[j])
    area_simp[i]=simps(mass_func, M)
    area_tink[i]=simps(mass_func_tink,M)
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()

#Mass function for the cosmology defined by params 1 - flat and full of matter
params1 = {'flat': True, 'H0': 67.2, 'Om0': 1.0-0.049, 'Ob0': 0.049, 'sigma8': 0.81, 'ns': 0.96}
cosmology.addCosmology('myCosmo1', params1)
cosmo = cosmology.setCosmology('myCosmo1')

plt.figure()
plt.title(r'HMF - $\Omega_{0M}=1$', fontsize=25)
plt.xlabel(r'$ Mvir\ [M_\odot]$',fontsize=18)
plt.ylabel(r'$dn/dln(M)\ [Mpc^{-3}]$',fontsize=18)
plt.ylim(1e-7, 1e-1)
plt.xscale('log')
plt.yscale('log')
for i in range(len(z)):
    mass_func=mf.massFunction(M,z[i],mdef = '200m', model = 'tinker08', q_out = 'dndlnM')
    plt.plot(M,mass_func, label='z=%.1f'%(z[i]))
    area1[i]=simps(mass_func,M)
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()

#Mass function for the cosmology defined by params 2 - standard
params2 = {'flat': True, 'H0': 67.2, 'Om0': 0.3, 'Ode0':0.7, 'Ob0': 0.049, 'sigma8': 0.81, 'ns': 0.96}
cosmology.addCosmology('myCosmo2', params2)
cosmo = cosmology.setCosmology('myCosmo2')

plt.figure()
plt.title(r'HMF - $\Omega_{0M} = 0.3\ \Omega_{0\Lambda}=0.7$', fontsize=25)
plt.xlabel(r'$ Mvir\ [M_\odot]$',fontsize=18)
plt.ylabel(r'$dn/dln(M)\ [Mpc^{-3}]$',fontsize=18)
plt.ylim(1e-7, 1e-1)
plt.xscale('log')
plt.yscale('log')
for i in range(len(z)):
    mass_func=mf.massFunction(M,z[i],mdef = '200m', model = 'tinker08', q_out = 'dndlnM')
    plt.plot(M,mass_func, label='z=%.1f'%(z[i]))
    area2[i]=simps(mass_func, M)
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()

#Mass function for the cosmology defined by params 3 - flat almost only DE
params3 = {'flat': True, 'H0': 67.2, 'Om0': 0.001, 'Ode0':1.0, 'Ob0': 0.0001, 'sigma8': 0.81, 'ns': 0.95}
cosmology.addCosmology('myCosmo3', params3)
cosmo = cosmology.setCosmology('myCosmo3')

plt.figure()
plt.title(r'HMF - $\Omega_{0\Lambda}\simeq 1$ ', fontsize=25)
plt.xlabel(r'$ Mvir\ [M_\odot]$',fontsize=18)
plt.ylabel(r'$dn/dln(M)\ [Mpc^{-3}]$',fontsize=18)
plt.ylim(1e-7, 1e-1)
plt.xscale('log')
plt.yscale('log')
for i in range(len(z)):
    mass_func=mf.massFunction(M,z[i],mdef = 'vir', model = 'comparat17', q_out = 'dndlnM')        #comparat17 works with 1 cosmology
    mass_func_tink = mf.massFunction(M,z[i],mdef = '200m', model = 'tinker08', q_out = 'dndlnM')   #tinker08 is cosmology independent
    plt.plot(M,mass_func, label='z=%.1f'%(z[i]))
    for j in range(len(M)-1):
        area3[i]=area3[i]+mass_func[j]*(M[j+1]-M[j])
        area3_tink[i]=simps(mass_func_tink,M)
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()


plt.figure()
#plt.xscale('log')
plt.yscale('log')
plt.title('Cluster count', fontsize=25)
plt.xlabel(r'$z$', fontsize=18)
plt.ylabel(r'$n(z)/n(z=0)$', fontsize=18)
plt.plot(z,area,label='planck18')
plt.plot(z,area_simp,label='planck18 simpson')
plt.plot(z,area_tink,label='planck18 tinker08')
plt.plot(z,area1,label=r'$flat\ \Omega_{0M}=1$')
plt.plot(z,area2,label=r'$flat\ \Omega_{0M}=0.3, \Omega_{0\Lambda}=0.7$')
plt.plot(z,area3,label=r'$flat\ \Omega_{0\Lambda}=1$  comparat')
plt.plot(z,area3_tink,label=r'$flat\ \Omega_{0\Lambda}=1$ tinker')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()



##Now make plots of P(k) and sigma(M)
cosmo=cosmology.setCosmology('planck18')
k = 10**np.arange(-5,2,0.02)
P_k1 = cosmo.matterPowerSpectrum(k)
R = 10**np.arange(0,2.4,0.005)
sigma1_tophat = cosmo.sigma(R, 0.0)
sigma3_tophat = cosmo.sigma(R, 2.0)
fig,ax = plt.subplots(2,1)
ax[0].loglog()
ax[1].loglog()
ax[0].set(xlabel='k(Mpc/h)',ylabel='P(k)')
ax[0].plot(k, P_k1, label='planck18')
ax[0].grid(True)
ax[1].grid(True)
ax[1].plot(R, sigma1_tophat, label='planck18')
ax[1].plot(R, sigma3_tophat, label=r'$\Omega_M$ = 1, z=2.0')
ax[1].set(xlabel='R(Mpc/h)',ylabel=r'$\sigma$(R)')
cosmo=cosmology.setCosmology('myCosmo1')
P_k2 = cosmo.matterPowerSpectrum(k)
sigma2_tophat = cosmo.sigma(R, 0.0)
ax[0].plot(k,P_k2, label=r'$\Omega_M$ = 1')
ax[1].plot(R, sigma2_tophat, label=r'$\Omega_M$ = 1')
plt.legend()
plt.tight_layout()
plt.show()





