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
a,b = 45, 199
mass_number_25 = np.zeros((a,b))
mass_number_50 = np.zeros((a,b))
mass_number_75 = np.zeros((a,b))
mass_number_100 = np.zeros((a,b))
mass_number_tot = np.zeros((a,b))
i=0
ncat = 0
out = open('mass_histogram0'+snap+'.txt','w+')
out.write("  z   OmegaM  h\r\n") 
out.write("%.4g %.4g %.4g\r\n" %(z, Omega0, hubble))
out.write('1st line: Mass_Bins   2nd line: dn/dlnM 1st realization  3rd line: dn/dlnM 2nd realization ... \n')
for catalog in glob.glob('/data26s/comparat/simulations/GLAM/1Gpc2000x4000/CATALOGS/CatshortV.0'+snap+'.00*.DAT'):
    ncat = ncat+1
    print('reading masses from catalog ', catalog)
    realization = catalog.split('/data26s/comparat/simulations/GLAM/1Gpc2000x4000/CATALOGS/CatshortV.0'+snap+'.0')[1].split('.DAT')[0]
    print('realization = ', realization)
    data = np.loadtxt(catalog, skiprows=8, dtype=float, unpack=True)
    mass = data[7]
    masses = np.append(masses,mass)	
    Xof = data[15]
#I fix the Xoff boundaries at z=0. I measured them one time and I report them here
#so I don't need the quartiles every time
#    quartiles = np.quantile(Xof, [0.25, 0.50, 0.75, 1.00])
    quartiles = np.array([0.05992, 0.1026, 0.159])
    print(quartiles)
    masses_25 = mass[Xof<=quartiles[0]]
    masses_50 = mass[(Xof>quartiles[0]) & (Xof<=quartiles[1])] 
 #   print('masses_50 = ', masses_50.min(), masses_50.max())
    masses_75 = mass[(Xof>quartiles[1]) & (Xof<=quartiles[2])] 
    masses_100 = mass[(Xof>quartiles[2])] 
    #create mass bins and count halos in different bins
    nbins=200
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
    mass_number_tot_tot, mass_bins = histogram(mass, bins=bins)
    mass_bins_average_tot, mass_bins, bin_number_tot = stats.binned_statistic(mass, mass,  bins=bins,     statistic='mean')
 #   number_per_bin = {'mass bin': np.array(mass_bins),'1st Xoff': np.array(mass_number_tot_25), '2nd Xoff': np.array(mass_number_tot_50), '3rd Xoff': np.array(mass_number_tot_75), '4th Xoff': np.array(mass_number_tot_100)}
    print(list(mass_bins))
    number_per_bin = (list(mass_bins_average_25), list(mass_number_tot_25), list(mass_number_tot_50), list(mass_number_tot_75), list(mass_number_tot_100), list(mass_number_tot_tot))
    #number_per_bin = np.transpose(number_per_bin)
  #  print('number per bin\n',number_per_bin)
    npb_file = pd.DataFrame(data=np.c_[number_per_bin], columns = ['mass bin', '1st Xoff', '2nd Xoff', '3rd Xoff', '4th Xoff', 'full sample'])
    outfile2 = 'halos_per_bin/number_per_bin_'+snap+'_'+realization+'.txt'
    npb_file.to_csv(outfile2)
    #np.savetxt(outfile2, number_per_bin,fmt='%s')    
    #measure dlnM to convert the count to dn/dlnM    
    diff = np.diff(np.log(mass_bins))
    diff = diff[0]
    #the simulation is a cube of 1 Gpc/h...
    #divide the number by 10^9 to get the number per (Mpc/h)^3
    mass_number_25[i,:] = mass_number_tot_25/(10**9)/diff#*np.log10(np.e)
    mass_number_50[i,:] = mass_number_tot_50/(10**9)/diff#*np.log10(np.e)
    mass_number_75[i,:] = mass_number_tot_75/(10**9)/diff#*np.log10(np.e)
    mass_number_100[i,:] = mass_number_tot_100/(10**9)/diff#*np.log10(np.e)
    mass_number_tot[i,:] = mass_number_tot_tot/(10**9)/diff
    i=i+1

#compute total mass function
mass_number_tot_sample, mass_bins = histogram(masses, bins=bins)
mass_bins_average_tot_sample, mass_bins, bin_number_tot_sample = stats.binned_statistic(masses, masses, bins=bins, statistic='mean')
#the simulation is a cube of 1 Gpc/h...
#divide the number by 10^9 to get the number per (Mpc/h)^3
mass_number_dn_dlnM = mass_number_tot_sample/(ncat*10**9)/diff

out.write("Here follow 6 mass functions, one for each quartile of the parameter Xoff and then the full realization and the full sample (all realizations)\n")
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

out.write("Full realization\n")
for i in range(b):  
    if(i==(b-1)):
       out.write("%.4g" %(mass_bins_average_tot[i]))      
    else:
        out.write("%.4g," %(mass_bins_average_tot[i]))
out.write("\n")
for i in range(a):
    for j in range(b):  
        if(j==(b-1)):
            out.write("%.4g" %(mass_number_tot[i,j]))
        else:
            out.write("%.4g," %(mass_number_tot[i,j])) 
    out.write("\n")

out.write("Full sample\n")
for i in range(b):  
    if(i==(b-1)):
       out.write("%.4g" %(mass_bins_average_tot_sample[i]))      
    else:
        out.write("%.4g," %(mass_bins_average_tot_sample[i]))
out.write("\n")

for j in range(b):  
    if(j==(b-1)):
        out.write("%.4g" %(mass_number_dn_dlnM[j]))
    else:
        out.write("%.4g," %(mass_number_dn_dlnM[j])) 
out.write("\n")


print('Done!')

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


