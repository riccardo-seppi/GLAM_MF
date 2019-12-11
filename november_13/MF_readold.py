#Riccardo Seppi - MPE - HEG (2019) - 25 October
#This code reads halo masses from DM simulations (GLAM)
#and counts halos in different mass bins, calculates dn/dlnM 
#and writes bins, counts in each bin, dn/dlnM, error in an output file
#This code is intended a step 1of2 alonside MF_fit.py, which fits the Mass Functions

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
from scipy import stats
from scipy import histogram
import camb
import hankel
#from hankel import SphericalHankelTransform
import scipy.spatial.ckdtree as t
from scipy.interpolate import interp1d



catalog='/data26s/comparat/simulations/GLAM/1Gpc2000x4000/CATALOGS/CatshortV.0092.0001.DAT'

#read here the cosmological parameters
a = np.loadtxt(catalog, usecols=[2], skiprows=1, max_rows=1, dtype=float)
z = 1/a - 1
print('z = ',z)
Omega0 = np.loadtxt(catalog, usecols=[1], skiprows=3, max_rows=1, dtype=float)
print('Omega0 = ', Omega0)
#Omega0DE = np.loadtxt(catalog, usecols=[3], skiprows=3, max_rows=1, dtype=float)
#print('Omega0DE = ', Omega0DE)
hubble = np.loadtxt(catalog, usecols=[6], skiprows=3, max_rows=1, dtype=float)
print('h = ', hubble)
#params = {'flat': False, 'H0': hubble*100, 'Om0': Omega0, 'Ode0': Omega0DE}
params = {'flat': True, 'H0': hubble*100, 'Om0': Omega0, 'Ob0': 0.049, 'sigma8': 0.828, 'ns': 0.96}

#work on the linear correlation function
#Set up a new set of parameters for CAMB
pars = camb.CAMBparams()
#This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency
pars.set_cosmology(H0=hubble*100, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)
pars.set_matter_power(redshifts=[z], kmax=2.0)
#Linear spectra
pars.NonLinear = camb.model.NonLinear_none
results = camb.get_results(pars)
kh, redshift, pk_lin = results.get_matter_power_spectrum(minkh=1e-4, maxkh=1, npoints = 10000)
s8 = np.array(results.get_sigma8())
print(results.get_sigma8())
#for i, (redshift, line) in enumerate(zip(z,['-','--'])):
#plt.loglog(kh, pk[i,:], color='k', ls = line)  #this line was in the for loop
plt.loglog(kh, pk_lin[0,:], color='r', label = 'z = %.3g'%redshift[0])
plt.xlabel('k/h Mpc',fontsize=10);
plt.legend()
plt.title('Matter power spectrum')
plt.ylabel('P(k)',fontsize=10)
plt.tight_layout()
plt.show()

#obtain Xi lin(r) from P(k)
pk = interp1d(np.hstack((1e-20,kh.min()/2., kh, kh.max()*2.,1e20)), np.hstack((0.,0.,pk_lin[0],0.,0.)))
h = hankel.SphericalHankelTransform(nu=0,N=10000,h=0.00001)  #Create the HankelTransform instance
fr = lambda x, r : (x/r)**2. * pk(x/r)
Rs= np.arange(0.1,200,0.1)
xiR = np.empty_like(Rs)
for i, R in enumerate(Rs):
	#print R, time.time()
	f = lambda x : fr(x,R)
	xiR[i] = h.transform(f)[0] / (2*np.pi**2 * R)

plt.figure()
plt.plot(Rs,xiR*Rs**2)
#plt.yscale('log')
#plt.ylim(-0.001,0.003)
plt.xlim(25,200)
#plt.title('Correlation function')
plt.xlabel('Mpc/h',fontsize=10)
plt.ylabel(r'$\xi(r)$',fontsize=10)
plt.tight_layout()
plt.show()


#read masses of the halos identified by BDM
print('reading masses from the catalog...')
masses = []
masses1 = []
masses2 = []
Xoff = []
Xcoord = []
Ycoord=[]
Zcoord=[]
ncat=0
for catalog in glob.glob('/data26s/comparat/simulations/GLAM/1Gpc2000x4000/CATALOGS/CatshortV.0092.0001.DAT'):
    ncat = ncat+1
    print('reading masses from catalog ', catalog)
    data = np.loadtxt(catalog, skiprows=8, dtype=float, unpack=True)
    mass = data[7]
    Xof = data[15]
    X_ = data[0]
    Y_ = data[1]
    Z_ = data[2]
    masses = np.append(masses,mass)	
    Xoff = np.append(Xoff, Xof)
    Xcoord = np.append(Xcoord, X_)
    Ycoord = np.append(Ycoord, Y_)
    Zcoord = np.append(Zcoord, Z_)
print('Done!')
print(data)
print('mass = ',masses, 'Xoff = ',Xoff)

xoff_med = np.median(Xoff)
print('Dividing mass1 with clusters with Xoff < %.3g and mass2 with Xoff > %.3g...' %(xoff_med, xoff_med))
masses1 = masses[Xoff <= xoff_med]
masses2 = masses[Xoff > xoff_med]

print('Lowest mass1: %.3g' %np.min(masses1))
print('Maximun mass1: %.3g' %np.max(masses1))
print('Lowest mass2: %.3g' %np.min(masses2))
print('Maximun mass2: %.3g' %np.max(masses2))

#create mass bins and count halos in different bins
nbins=50
bins = np.logspace(12.1,14.2,nbins)
bins1 = np.logspace(12.1,14.2,nbins)
bins2 = np.logspace(12.1,14.2,nbins)
#create histogram
mass_number_tot, mass_bins = histogram(masses, bins=bins)
mass_number_tot1, mass_bins1 = histogram(masses1, bins=bins1)
mass_number_tot2, mass_bins2 = histogram(masses2, bins=bins2)
#measure mass average in each bin (dn/dlnM will be plotted at that position)
mass_bins_average, mass_bins, bin_number = stats.binned_statistic(masses, masses,  bins=bins, statistic='mean')
mass_bins_average1, mass_bins1, bin_number1 = stats.binned_statistic(masses1, masses1,  bins=bins1, statistic='mean')
mass_bins_average2, mass_bins2, bin_number2 = stats.binned_statistic(masses2, masses2,  bins=bins2, statistic='mean')
#check with prints and hist
print('mass bins1 = ', mass_bins1, 'mass bins average1 = ', mass_bins_average1, ' bin number = ', bin_number1)
plt.hist(masses1,mass_bins_average1)
plt.hist(masses2,mass_bins_average2)
plt.loglog()
#plt.show()

#measure dlnM to convert the count to dn/dlnM    
diff = np.diff(np.log(mass_bins))
diff = diff[0]
diff1 = np.diff(np.log(mass_bins1))
print('diff1=',diff1)
diff1=diff1[0]
diff2 = np.diff(np.log(mass_bins2))
print('diff2=',diff2)
diff2=diff2[0]

print('number of halos in each bin 1: ',mass_number_tot1)   
print('total number of halos used 1: ',np.sum(mass_number_tot1))
print('total number of halos in the simulation 1: ', len(masses1))

print('number of halos in each bin 2: ',mass_number_tot2)   
print('total number of halos used 2: ',np.sum(mass_number_tot2))
print('total number of halos in the simulation 2: ', len(masses2))
             
#compute Poisson error in each bin
#the simulation is a cube of 1 Gpc/h...
#divide the number by 10^9 to get the number per (Mpc/h)^3
counts_error = np.sqrt(mass_number_tot)/diff/10**9/ncat#*np.log10(np.e)
mass_number = mass_number_tot/(ncat*10**9)/diff#*np.log10(np.e)

counts_error1 = np.sqrt(mass_number_tot1)/diff1/10**9/ncat#*np.log10(np.e)
mass_number1 = mass_number_tot1/(ncat*10**9)/diff1#*np.log10(np.e)

counts_error2 = np.sqrt(mass_number_tot2)/diff2/10**9/ncat#*np.log10(np.e)
mass_number2 = mass_number_tot2/(ncat*10**9)/diff2#*np.log10(np.e)


#calculate Xi from eq 22 of comparat17 paper
rmax = 20 #Mpc/h
dr = 0.1 #Mpc/h
Lbox = 1000 #Mpc/h
volume = Lbox**3
iselect = (Xcoord > rmax)&(Xcoord< (Lbox-rmax))&(Ycoord > rmax)&(Ycoord< (Lbox-rmax))&(Zcoord > rmax)&(Zcoord< (Lbox-rmax))
xis=[]
#bias2 = np.zeros(len(mass_bins_average))
out = open('correlation_function0092.txt','w+')
for i in range(len(mass_bins_average)):
    print(i,'of',len(mass_bins_average))
    isel_m = (masses>mass_bins[i])&(masses<mass_bins[i+1])
    isel = (iselect)&(isel_m) #(masses>mass_bins[i])&(masses<mass_bins[i+1])
    print('Randoms')
    treeRandoms=t.cKDTree(np.transpose([Xcoord[isel_m],Ycoord[isel_m],Zcoord[isel_m]]),1000.0) 
    print('Data...')
    treeData=t.cKDTree(np.transpose([Xcoord[isel],Ycoord[isel],Zcoord[isel]]),1000.0)
    nD=len(treeData.data)
    nR=len(treeRandoms.data)
    #print nD, nR
    bin_xi3D=np.arange(0, rmax, dr)
    # now does the pair counts :
    print('pair counting...')
    pairs=treeData.count_neighbors(treeRandoms, bin_xi3D)
    #t3 = time.time()
    DR=pairs[1:]-pairs[:-1]
    dV= 4*np.pi*(bin_xi3D[1:]**3 - bin_xi3D[:-1]**3 )/3.
    pairCount=nD*nR#-nD*(nD-1)/2.
    xi = DR*volume/(dV * pairCount) -1.  
 #   xis = append(xis,xi)
  #  bias2[i] = 1/200 * np.sum(xi/xiR)  
    out.write("  M_low   M_high  corr_func\r\n") 
    out.write("%.4g %.4g %.4g \r\n" %(mass_bins[i], mass_bins[i+1], xi))
    out.close()

plt.figure()
plt.plot(mass_bins_average, bias2)
#plt.show()
#now save to outfile: cosmological parameters,
#mass bins, counts - dn/dlnM - error in each bin
#outfile has to be used as infile by MF_fit.py

outfile = open('mass_histogram0092_test.txt','w+')
outfile.write("  z   OmegaM  h  Xoff_median\r\n") 
outfile.write("%.4g %.4g %.4g  %.4g\r\n" %(z, Omega0, hubble, xoff_med))
outfile.write('Mass_Bins   Counts   dn/dlnM     error   bias2   Mass_Bins1   Counts1   dn/dlnM1     error1   Mass_Bins2   Counts2   dn/dlnM2     error2\n')
for i in range(len(mass_bins_average)):
	outfile.write("%.4g  %.4g %.4g  %.4g   %.4g   %.4g  %.4g  %.4g   %.4g   %.4g  %.4g  %.4g   %.4g\r\n" %(mass_bins_average[i], mass_number_tot[i], mass_number[i], counts_error[i], bias2[i], mass_bins_average1[i], mass_number_tot1[i], mass_number1[i], counts_error1[i], mass_bins_average2[i], mass_number_tot2[i], mass_number2[i], counts_error2[i]))
outfile.close()
plt.show()

