#Riccardo Seppi - MPE - HEG (2019) - 25 October
#This code reads halo masses from DM simulations (GLAM)
#builds HMF and fits them to models with fixed cosmological parameters

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import uniform
from scipy.stats import norm
from bayescorner import bayescorner

#read the MFs
infile='mass_histogram0096.txt'
cosmo_params = np.loadtxt(infile, skiprows=1, max_rows=1, dtype=float)
z, Omega0, hubble, Xoff = cosmo_params
print(cosmo_params)
params = {'flat': True, 'H0': hubble*100, 'Om0': Omega0, 'Ob0': 0.049, 'sigma8': 0.828, 'ns': 0.96}

mass_data = np.loadtxt(infile, skiprows=3, dtype=float)
print(mass_data[::10,:-1])
mass_bins_pl = mass_data[:,0] 
total = mass_data[:,1]
mass_number = mass_data[:,2]
counts_error = mass_data[:,3]

mass_bins_pl1 = mass_data[:,4] 
total1 = mass_data[:,5]
mass_number1 = mass_data[:,6]
counts_error1 = mass_data[:,7]

mass_bins_pl2 = mass_data[:,8] 
total2 = mass_data[:,9]
mass_number2 = mass_data[:,10]
counts_error2 = mass_data[:,11]
#Now I want to fit it
#consider the model (comparat17, tinker08...)
#NB: z will have to be the same of the simulation analyzed!!!
from colossus.lss import mass_function as mf
from colossus.cosmology import cosmology
from colossus.lss import peaks
#cosmology.setCosmology('planck18')
cosmology.addCosmology('myCosmo', params) #params was defined at line 16
cosmo=cosmology.setCosmology('myCosmo')
#print(cosmo.rho_m(0.0))

#fitting with Bhattacharya 2011
A0 = 0.333
a0 = 0.788
p0 = 0.807
q0 = 1.795
def mass_function_rseppi(Mass,A0,a0,p0,q0):
    cosmo=cosmology.getCurrent()    
    delta_c = peaks.collapseOverdensity(z=z)
    R = peaks.lagrangianR(Mass)
    sigma = cosmo.sigma(R=R,z=z)
    nu = delta_c / sigma
    nu2 = nu**2
    zp1 = 1.0+z
    A = A0 * zp1**-0.11
    a = a0 * zp1**-0.01
    p = p0
    q = q0
    f = A * np.sqrt(2 / np.pi) * np.exp(-a * nu2 * 0.5) * (1.0 + (a*nu2)**-p) * (nu * np.sqrt(a))**q

    d_ln_sigma_d_ln_R = cosmo.sigma(R, z, derivative = True)
    rho_Mpc = cosmo.rho_m(0.0) * 1E9
    mass_func_model = -1/3*f*rho_Mpc/Mass*d_ln_sigma_d_ln_R
    return mass_func_model


mass_func_model = mass_function_rseppi(mass_bins_pl,A0,a0,p0,q0)
#print('model values = ', mass_func_model)
#mass_func_model=mf.massFunction(mass_bins_pl,z=z,mdef = 'vir', model = 'tinker08', q_out = 'dndlnM')
figure, ax = plt.subplots(2,1)
ax[0].loglog()
ax[1].set_xlabel(r'M [$M_{\odot}$/h]')
ax[1].set_ylabel('ratio')
ax[0].set_ylabel(r'dn/dlnM [$(Mpc/h)^{-3}$]')
mf_test = mf.massFunction(mass_bins_pl,z=z,mdef = 'fof', model = 'bhattacharya11', q_out = 'dndlnM')
ax[0].plot(mass_bins_pl, mf_test, label='Bhattacharya11')
ax[0].plot(mass_bins_pl, mass_func_model, label='model_rseppi')
ax[1].plot(mass_bins_pl, mass_func_model/mf_test, color='r')
ax[0].legend()
#plt.show()

# Use MCMC method - emcee
#import lmfit
import pymultinest
from tqdm import tqdm
#import emcee
#p=lmfit.Parameters()
#p.add_many(('A0', 0.333, True, 0.001,0.5),('a0', 0.788, True,0.2,4.0),('p0', 0.807, True,-2.0,7.0),('q0', 1.795, True,0.5,10.0))

parameters = ['A0', 'a0', 'p0', 'q0']

'''
def log_prior(A0,a0,p0,q0):
    #v=p.valuesdict()
    #A0,a0,p0,q0 = pa
    mu = np.array([0.333, 0.788, 0.807, 1.795])
    sigma = np.array([0.15, 0.4, 0.45, 1.0])    
    #if 0.2 < v['A0'] < 0.5 and 0.6 < v['a0'] < 1.5 and 0.0 < v['p0'] < 2.0 and 1.5 < v['q0'] < 2.5:
    if(0.2 < A0 < 0.5 and 0.6 < a0 < 1.5 and 0.0 < p0 < 2.0 and 1.5 < q0 < 2.5):
        return np.sum((1/np.sqrt(2*np.pi*sigma*sigma))-0.5*((mu-p)/sigma)**2)
    return -np.inf
'''
plt.figure()
cube=np.arange(-1,1,0.002)
print(cube)
cube_pdf=norm.pdf(cube,loc=0.333,scale=1)
print(cube_pdf)
plt.hist(cube_pdf,bins=30,range=[0,1])
cube_ppf=norm.ppf(cube,loc=0.333,scale=1)
print(cube_ppf)
plt.hist(cube_ppf,bins=30,range=[0,1])
plt.plot(cube,cube_pdf, label = 'pdf')
plt.plot(cube,cube_ppf, label = 'ppf')
plt.show()

def prior(cube,ndim,nparams):
    #cube[0] = (cube[0] +  0.333)    
    #cube[1] = (cube[1]*500 +  0.788)
    #cube[2] = (cube[2]*500 +  0.807)
    #cube[3] = (cube[3]*500 +  1.795)
    cube[0] = norm.ppf(cube[0], loc = 0.333, scale = 0.3)       
    cube[1] = norm.ppf(cube[1], loc = 0.788, scale = 0.5)   
    cube[2] = norm.ppf(cube[2], loc = 0.807, scale = 0.5)   
    cube[3] = norm.ppf(cube[3], loc = 1.795, scale = 0.5)  
'''
for i in range(len(counts_error)):
    if(i<10):
        counts_error[i] = 1000*counts_error[i]
    elif(10<=i<20):
        counts_error[i] = 50*counts_error[i]
'''
'''
#define residual function
def residual(p):
    v=p.valuesdict()     
    res = (mass_number - mass_function_rseppi(mass_bins_pl,v['A0'],v['a0'],v['p0'],v['q0']))/counts_error
    return res

def residual1(p):
    v=p.valuesdict()     
    res = (mass_number1 - mass_function_rseppi(mass_bins_pl1,v['A0'],v['a0'],v['p0'],v['q0']))/counts_error1
    return res

def residual2(p):
    v=p.valuesdict()     
    res = (mass_number2 - mass_function_rseppi(mass_bins_pl2,v['A0'],v['a0'],v['p0'],v['q0']))/counts_error2
    return res    
#print('residuals =',residual(p))

mi = lmfit.minimize(residual, p, method='leastsq')
# print report on the leastsq fit 
lmfit.printfuncs.report_fit(mi.params, min_correl=0.5)
print(mi.params)

mi1 = lmfit.minimize(residual1, p, method='leastsq')
# print report on the leastsq fit 
lmfit.printfuncs.report_fit(mi1.params, min_correl=0.5)
print(mi1.params)

mi2 = lmfit.minimize(residual2,p, method='leastsq')
# print report on the leastsq fit 
lmfit.printfuncs.report_fit(mi2.params, min_correl=0.5)
print(mi2.params)
'''

#now I want to use MCMC method
'''
def loglike(p):
    resid = residual(p)
    resid = resid**2
    resid = resid + np.log(2*np.pi*counts_error**2)
    logL = -0.5*np.sum(resid)
    if(np.isnan(logL)):
        logL=-np.inf
    return logL
'''

def loglike(cube, ndim, nparams):
    A0 = cube[0]
    a0 = cube[1]
    p0 = cube[2]    
    q0 = cube[3]
    ymodel = mass_function_rseppi(mass_bins_pl, A0, a0, p0, q0)
    resid = (mass_number - ymodel)/counts_error
    resid = resid**2
    resid = resid + np.log(2*np.pi*counts_error**2)
    logL = -0.5*np.sum(resid)
    if(np.isnan(logL)):
        logL=-np.inf
    return logL

def loglike1(cube, ndim, nparams):
    A0 = cube[0]
    a0 = cube[1]
    p0 = cube[2]    
    q0 = cube[3]
    ymodel = mass_function_rseppi(mass_bins_pl1, A0, a0, p0, q0)
    resid = (mass_number1 - ymodel)/counts_error1
    resid = resid**2
    resid = resid + np.log(2*np.pi*counts_error1**2)
    logL = -0.5*np.sum(resid)
    if(np.isnan(logL)):
        logL=-np.inf
    return logL

def loglike2(cube, ndim, nparams):
    A0 = cube[0]
    a0 = cube[1]
    p0 = cube[2]    
    q0 = cube[3]
    ymodel = mass_function_rseppi(mass_bins_pl2, A0, a0, p0, q0)
    resid = (mass_number2 - ymodel)/counts_error2
    resid = resid**2
    resid = resid + np.log(2*np.pi*counts_error2**2)
    logL = -0.5*np.sum(resid)
    if(np.isnan(logL)):
        logL=-np.inf
    return logL

n_params = len(parameters)

plt.figure() 
#plt.scatter(mass_bins_pl, mass_number)
plt.errorbar(mass_bins_pl, mass_number, yerr = counts_error, fmt='.', label='full sample')
plt.errorbar(mass_bins_pl1, mass_number1, yerr=counts_error1, fmt='.', label='Xoff < %.3g'%Xoff)
plt.errorbar(mass_bins_pl2,mass_number2,yerr=counts_error2, fmt='.', label='Xoff > %.3g'%Xoff)

# run MultiNest
import json
datafile = 'output/datafile'
print('Running multinest...')
resum = False
pymultinest.run(loglike, prior, n_params, outputfiles_basename=datafile, resume = resum, verbose = True)
print('Done!')
json.dump(parameters, open(datafile + 'params.json', 'w')) # save parameter names
print('Running Analyzer...')
a = pymultinest.Analyzer(outputfiles_basename=datafile, n_params = n_params)
bestfit_params = a.get_best_fit()
print(bestfit_params)
v=list(bestfit_params.values())
print(v)
A0,a0,p0,q0 = v[1]
plt.plot(mass_bins_pl, mass_function_rseppi(mass_bins_pl,A0, a0, p0, q0), ls='solid', label='fit full sample')

pymultinest.run(loglike1, prior, n_params, outputfiles_basename=datafile + '_1_', resume = resum, verbose = True)
json.dump(parameters, open(datafile + '_1_params.json', 'w')) # save parameter names
a1 = pymultinest.Analyzer(outputfiles_basename=datafile + '_1_', n_params = n_params)
bestfit_params1 = a1.get_best_fit()
v1=list(bestfit_params1.values())
print(v1)
A0,a0,p0,q0 = v1[1]
plt.plot(mass_bins_pl1, mass_function_rseppi(mass_bins_pl1,A0, a0, p0, q0), label='fit Xoff < %.3g'%Xoff)

pymultinest.run(loglike2, prior, n_params, outputfiles_basename=datafile + '_2_', resume = resum, verbose = True)
json.dump(parameters, open(datafile + '_2_params.json', 'w')) # save parameter names
a2 = pymultinest.Analyzer(outputfiles_basename=datafile + '_2_', n_params = n_params)
bestfit_params2 = a2.get_best_fit()
v2=list(bestfit_params2.values())
print(v2)
A0,a0,p0,q0 = v2[1]
plt.plot(mass_bins_pl2, mass_function_rseppi(mass_bins_pl2,A0, a0, p0, q0), label='fit Xoff > %.3g'%Xoff)
plt.plot(mass_bins_pl, mass_function_rseppi(mass_bins_pl,0.333, 0.788, 0.807, 1.795), label='Bhattacharya 2011')

plt.loglog()
plt.xlabel(r'M [$M_{\odot}/h]$', fontsize=18)
plt.ylabel(r'dn/dlnM $[(Mpc/h)^{-3}]$',fontsize=18)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(datafile + 'all_data.pdf')
plt.show()
plt.close()

a_lnZ = a.get_stats()['global evidence']

import corner
data = a.get_data()[:,2:]
weights = a.get_data()[:,0]
data1 = a1.get_data()[:,2:]
weights1 = a1.get_data()[:,0]
data2 = a2.get_data()[:,2:]
weights2 = a2.get_data()[:,0]

mask = weights.cumsum() > 1e-5
mask = weights > 1e-4
mask1 = weights1.cumsum() > 1e-5
mask1 = weights1 > 1e-4
mask2 = weights2.cumsum() > 1e-5
mask2 = weights2 > 1e-4

#fig = bayescorner(params = [data[:,0],data[:,1],data[:,2],data[:,3]], param_names = ['A0', 'a0', 'p0', 'q0'], color_base = '#1f77b4', figsize=(14,14))
corner.corner(data[mask,:], weights=weights[mask], 
	labels=parameters, show_titles=True, truths = v[1], title = 'full sample')
corner.corner(data1[mask1,:], weights=weights1[mask1], 
	labels=parameters, show_titles=True, truths = v1[1], title = 'fit Xoff < %.3g'%Xoff)
corner.corner(data2[mask2,:], weights=weights2[mask2], 
	labels=parameters, show_titles=True, truths = v2[1], title = 'fit Xoff > %.3g'%Xoff)
plt.show()
'''
def loglike1(p):
    resid = residual1(p)
    resid = resid**2
    resid = resid + np.log(2*np.pi*counts_error1**2)
    logL = -0.5*np.sum(resid)
    if(np.isnan(logL)):
        logL=-np.inf
    return logL

def loglike2(p):
    resid = residual2(p)
    resid = resid**2
    resid = resid + np.log(2*np.pi*counts_error2**2)
    logL = -0.5*np.sum(resid)
    if(np.isnan(logL)):
        logL=-np.inf
    return logL

def logPoisson(p):
    v=p.valuesdict()     
    logL = - np.sum(mass_function_rseppi(mass_bins_pl,v['A0'],v['a0'],v['p0'],v['q0'])) + np.sum(mass_number*np.log(mass_function_rseppi(mass_bins_pl,v['A0'],v['a0'],v['p0'],v['q0'])))
    if(np.isnan(logL)):
        logL=-np.inf
    return logL

priors = np.array([0.333, 0.788, 0.807, 1.795])
print(type(priors))
print(priors[0])
def logProb(A0,a0,p0,q0):
    lp = log_prior(A0,a0,p0,q0)
    if not np.isfinte(lp):
        return -np.inf
    return lp + loglike

# build a general minimizer for curve fitting and optimization.
mini = lmfit.Minimizer(loglike, mi.params, nan_policy='propagate')
#mini = lmfit.Minimizer(logPoisson, mi.params, nan_policy='propagate')
# sampling of the posterion distribution
res = mini.emcee(burn=300, steps=2000, thin=10,params=mi.params)

# show corner plot (confidence limits, parameter distributions, correlations)
print('parameters plot')
figure=corner.corner(res.flatchain, labels=res.var_names, 
                     truths=list(res.params.valuesdict().values()),
                     show_titles=True, title_kwargs={"fontsize": 12})
#plt.show()
print("median of posterior probability distribution")
print('------------------------------------------')
lmfit.report_fit(res.params)

mini1 = lmfit.Minimizer(loglike1, mi1.params, nan_policy='propagate')
res1 = mini1.emcee(burn=300, steps=2000, thin=10,params=mi1.params)
figure=corner.corner(res1.flatchain, labels=res1.var_names, 
                     truths=list(res1.params.valuesdict().values()),
                     show_titles=True, title_kwargs={"fontsize": 12})


mini2 = lmfit.Minimizer(loglike2, mi2.params, nan_policy='propagate')
res2 = mini2.emcee(burn=300, steps=2000, thin=10,params=mi2.params)
figure=corner.corner(res2.flatchain, labels=res2.var_names, 
                     truths=list(res2.params.valuesdict().values()),
                     show_titles=True, title_kwargs={"fontsize": 12})
plt.figure()
plt.errorbar(mass_bins_pl,mass_number,yerr=counts_error, fmt='.', label='full sample')
plt.errorbar(mass_bins_pl1,mass_number1,yerr=counts_error1, fmt='.', label='Xoff low')
plt.errorbar(mass_bins_pl2,mass_number2,yerr=counts_error2, fmt='.', label='Xoff high')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'M [$M_{\odot}/h]$', fontsize=18)
plt.ylabel(r'dn/dlnM $[(Mpc/h)^{-3}]$',fontsize=18)
plt.grid(True)
plt.tight_layout()
mf_test_tinker = mf.massFunction(mass_bins_pl,z=z,mdef = '200m', model = 'tinker08', q_out = 'dndlnM')
plt.plot(mass_bins_pl, mf_test_tinker, label='tinker08')
plt.plot(mass_bins_pl,mass_function_rseppi(mass_bins_pl,mi.params['A0'],mi.params['a0'],mi.params['p0'],mi.params['q0']), label='fit full')
plt.plot(mass_bins_pl1,mass_function_rseppi(mass_bins_pl1,mi1.params['A0'],mi1.params['a0'],mi1.params['p0'],mi1.params['q0']), label='fit low')
plt.plot(mass_bins_pl2,mass_function_rseppi(mass_bins_pl2,mi2.params['A0'],mi2.params['a0'],mi2.params['p0'],mi2.params['q0']), label='fit high')
plt.legend()
plt.show()

'''

