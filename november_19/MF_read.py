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
import sys
#from hankel import SphericalHankelTransform
import scipy.spatial.ckdtree as t
from scipy.interpolate import interp1d



#catalog='/data26s/comparat/simulations/GLAM/1Gpc2000x4000/CATALOGS/CatshortV.0096.0001.DAT'
snap=sys.argv[1]
catalog = '/data26s/comparat/simulations/GLAM/1Gpc2000x4000/CATALOGS/CatshortV.0'+snap+'.0001.DAT'
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

#read masses of the halos identified by BDM
print('reading masses from the catalog...')
masses = []
a,b = 45, 49
mass_number_25 = np.zeros((a,b))
mass_number_50 = np.zeros((a,b))
mass_number_75 = np.zeros((a,b))
mass_number_100 = np.zeros((a,b))
i=0
out = open('mass_histogram0'+snap+'.txt','w+')
out.write("  z   OmegaM  h\r\n") 
out.write("%.4g %.4g %.4g\r\n" %(z, Omega0, hubble))
out.write('Mass_Bins    dn/dlnM ... \n')
for catalog in glob.glob('/data26s/comparat/simulations/GLAM/1Gpc2000x4000/CATALOGS/CatshortV.0'+snap+'.00*.DAT'):
    print('reading masses from catalog ', catalog)
    data = np.loadtxt(catalog, skiprows=8, dtype=float, unpack=True)
    mass = data[7]
    Xof = data[15]
    quartiles = np.quantile(Xof, [0.25, 0.50, 0.75, 1.00])
    print(quartiles)
    masses_25 = mass[Xof<=quartiles[0]]
    masses_50 = mass[(Xof>quartiles[0]) & (Xof<=quartiles[1])] 
 #   print('masses_50 = ', masses_50.min(), masses_50.max())
    masses_75 = mass[(Xof>quartiles[1]) & (Xof<=quartiles[2])] 
    masses_100 = mass[(Xof>quartiles[2]) & (Xof<=quartiles[3])] 
    #create mass bins and count halos in different bins
    nbins=50
    bins = np.logspace(12.1,14.2,nbins)
    #create histogram and measure mass average in each bin (dn/dlnM will be plotted at that position)
    mass_number_tot_25, mass_bins = histogram(masses_25, bins=bins)
    mass_bins_average_25, mass_bins, bin_number_25 = stats.binned_statistic(masses_25, masses_25,  bins=bins,     statistic='mean')
    mass_number_tot_50, mass_bins = histogram(masses_50, bins=bins)
    mass_bins_average_50, mass_bins, bin_number_50 = stats.binned_statistic(masses_50, masses_50,  bins=bins,     statistic='mean')
    mass_number_tot_75, mass_bins = histogram(masses_75, bins=bins)
    mass_bins_average_75, mass_bins, bin_number_75 = stats.binned_statistic(masses_75, masses_75,  bins=bins,     statistic='mean')
    mass_number_tot_100, mass_bins = histogram(masses_100, bins=bins)
    mass_bins_average_100, mass_bins, bin_number_100 = stats.binned_statistic(masses_100, masses_100,  bins=bins,     statistic='mean')
    #measure dlnM to convert the count to dn/dlnM    
    diff = np.diff(np.log(mass_bins))
    diff = diff[0]
    #the simulation is a cube of 1 Gpc/h...
    #divide the number by 10^9 to get the number per (Mpc/h)^3
    mass_number_25[i,:] = mass_number_tot_25/(10**9)/diff#*np.log10(np.e)
    mass_number_50[i,:] = mass_number_tot_50/(10**9)/diff#*np.log10(np.e)
    mass_number_75[i,:] = mass_number_tot_75/(10**9)/diff#*np.log10(np.e)
    mass_number_100[i,:] = mass_number_tot_100/(10**9)/diff#*np.log10(np.e)
    i=i+1

out.write("Here follow 4 mass functions, one for each quartile of the parameter Xoff\n")
out.write('1st quartile\n')
for i in range(b):
    if(i==(b-1)):
        out.write('bin%.d'%i)
    else:
        out.write('bin%.d,'%i)
out.write('\n')
for i in range(b):
    if(i==(b-1)):
       out.write("%.4g" %(mass_bins_average_25[i]))      
    else:
        out.write("%.4g," %(mass_bins_average_25[i]))
out.write("\n")
for i in range(a):
    for j in range(b):  
        if(j==(b-1)):
            out.write("%.4g" %(mass_number_25[i,j]))
        else:
            out.write("%.4g," %(mass_number_25[i,j]))            
    out.write("\n")

out.write("2nd quartile\n")
for i in range(b):  
    if(i==(b-1)):
       out.write("%.4g" %(mass_bins_average_50[i]))      
    else:
        out.write("%.4g," %(mass_bins_average_50[i]))
out.write("\n")
for i in range(a):
    for j in range(b):  
        if(j==(b-1)):
            out.write("%.4g" %(mass_number_50[i,j]))
        else:
            out.write("%.4g," %(mass_number_50[i,j])) 
    out.write("\n")
'''
for i in range(b): 
    out.write("%.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g\r\n" %(mass_bins_average_50[i], mass_number_50[0,i], mass_number_50[1,i], mass_number_50[2,i], mass_number_50[3,i], mass_number_50[4,i], mass_number_50[5,i], mass_number_50[6,i], mass_number_50[7,i], mass_number_50[8,i], mass_number_50[9,i], mass_number_50[10,i], mass_number_50[11,i], mass_number_50[12,i], mass_number_50[13,i], mass_number_50[14,i], mass_number_50[15,i], mass_number_50[16,i], mass_number_50[17,i], mass_number_50[18,i], mass_number_50[19,i], mass_number_50[20,i], mass_number_50[21,i], mass_number_50[22,i], mass_number_50[23,i], mass_number_50[24,i], mass_number_50[25,i], mass_number_50[26,i], mass_number_50[27,i], mass_number_50[28,i], mass_number_50[29,i], mass_number_50[30,i], mass_number_50[31,i], mass_number_50[32,i], mass_number_50[33,i], mass_number_50[34,i], mass_number_50[35,i], mass_number_50[36,i], mass_number_50[37,i], mass_number_50[38,i], mass_number_50[39,i], mass_number_50[40,i], mass_number_50[41,i], mass_number_50[42,i], mass_number_50[43,i], mass_number_50[44,i]))
'''

out.write("3rd quartile\n")
for i in range(b):  
    if(i==(b-1)):
       out.write("%.4g" %(mass_bins_average_75[i]))      
    else:
        out.write("%.4g," %(mass_bins_average_75[i]))
out.write("\n")
for i in range(a):
    for j in range(b):  
        if(j==(b-1)):
            out.write("%.4g" %(mass_number_75[i,j]))
        else:
            out.write("%.4g," %(mass_number_75[i,j])) 
    out.write("\n")
'''
for i in range(b): 
    out.write("%.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g\r\n" %(mass_bins_average_75[i], mass_number_75[0,i], mass_number_75[1,i], mass_number_75[2,i], mass_number_75[3,i], mass_number_75[4,i], mass_number_75[5,i], mass_number_75[6,i], mass_number_75[7,i], mass_number_75[8,i], mass_number_75[9,i], mass_number_75[10,i], mass_number_75[11,i], mass_number_75[12,i], mass_number_75[13,i], mass_number_75[14,i], mass_number_75[15,i], mass_number_75[16,i], mass_number_75[17,i], mass_number_75[18,i], mass_number_75[19,i], mass_number_75[20,i], mass_number_75[21,i], mass_number_75[22,i], mass_number_75[23,i], mass_number_75[24,i], mass_number_75[25,i], mass_number_75[26,i], mass_number_75[27,i], mass_number_75[28,i], mass_number_75[29,i], mass_number_75[30,i], mass_number_75[31,i], mass_number_75[32,i], mass_number_75[33,i], mass_number_75[34,i], mass_number_75[35,i], mass_number_75[36,i], mass_number_75[37,i], mass_number_75[38,i], mass_number_75[39,i], mass_number_75[40,i], mass_number_75[41,i], mass_number_75[42,i], mass_number_75[43,i], mass_number_75[44,i]))
'''

out.write("4th quartile\n")
for i in range(b):  
    if(i==(b-1)):
       out.write("%.4g" %(mass_bins_average_100[i]))      
    else:
        out.write("%.4g," %(mass_bins_average_100[i]))
out.write("\n")
for i in range(a):
    for j in range(b):  
        if(j==(b-1)):
            out.write("%.4g" %(mass_number_100[i,j]))
        else:
            out.write("%.4g," %(mass_number_100[i,j])) 
    out.write("\n")

print('Done!')
'''
for i in range(b): 
    out.write("%.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g  %.4g\r\n" %(mass_bins_average_100[i], mass_number_100[0,i], mass_number_100[1,i], mass_number_100[2,i], mass_number_100[3,i], mass_number_100[4,i], mass_number_100[5,i], mass_number_100[6,i], mass_number_100[7,i], mass_number_100[8,i], mass_number_100[9,i], mass_number_100[10,i], mass_number_100[11,i], mass_number_100[12,i], mass_number_100[13,i], mass_number_100[14,i], mass_number_100[15,i], mass_number_100[16,i], mass_number_100[17,i], mass_number_100[18,i], mass_number_100[19,i], mass_number_100[20,i], mass_number_100[21,i], mass_number_100[22,i], mass_number_100[23,i], mass_number_100[24,i], mass_number_100[25,i], mass_number_100[26,i], mass_number_100[27,i], mass_number_100[28,i], mass_number_100[29,i], mass_number_100[30,i], mass_number_100[31,i], mass_number_100[32,i], mass_number_100[33,i], mass_number_100[34,i], mass_number_100[35,i], mass_number_100[36,i], mass_number_100[37,i], mass_number_100[38,i], mass_number_100[39,i], mass_number_100[40,i], mass_number_100[41,i], mass_number_100[42,i], mass_number_100[43,i], mass_number_100[44,i]))
out.close()
print('Done!')
'''
'''
#compute Poisson error in each bin
#the simulation is a cube of 1 Gpc/h...
#divide the number by 10^9 to get the number per (Mpc/h)^3
counts_error = np.sqrt(mass_number_tot)/diff/10**9/ncat#*np.log10(np.e)
mass_number = mass_number_tot/(ncat*10**9)/diff#*np.log10(np.e)

counts_error1 = np.sqrt(mass_number_tot1)/diff1/10**9/ncat#*np.log10(np.e)
mass_number1 = mass_number_tot1/(ncat*10**9)/diff1#*np.log10(np.e)

counts_error2 = np.sqrt(mass_number_tot2)/diff2/10**9/ncat#*np.log10(np.e)
mass_number2 = mass_number_tot2/(ncat*10**9)/diff2#*np.log10(np.e)
'''

'''
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
'''

'''
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
'''
#plt.figure()
#plt.plot(Rs,xiR*Rs**2)
#plt.yscale('log')
#plt.ylim(-0.001,0.003)
#plt.xlim(25,200)
#plt.title('Correlation function')
#plt.xlabel('Mpc/h',fontsize=10)
#plt.ylabel(r'$\xi(r)$',fontsize=10)
#plt.tight_layout()
#plt.show()


