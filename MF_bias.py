import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
from scipy import stats
from scipy import histogram
from scipy import integrate
import camb
import hankel
import sys
import os
import scipy.spatial.ckdtree as t
from scipy.interpolate import interp1d
from nbodykit.lab import *
from nbodykit import setup_logging, style
from nbodykit.source.catalog import CSVCatalog

#read which version (092 096 .. 105 128 132) and which realization I am analyzing
snap=sys.argv[1]
realization = sys.argv[2]
#read cosmological parameters from catalog
catalog = '/data26s/comparat/simulations/GLAM/1Gpc2000x4000/CATALOGS/CatshortV.0'+snap+'.0'+realization+'.DAT'
a = np.loadtxt(catalog, usecols=[2], skiprows=1, max_rows=1, dtype=float)
z = 1/a - 1
print('z = ',z)
Omega0 = np.loadtxt(catalog, usecols=[1], skiprows=3, max_rows=1, dtype=float)
print('Omega0 = ', Omega0)
hubble = np.loadtxt(catalog, usecols=[6], skiprows=3, max_rows=1, dtype=float)
print('h = ', hubble)
#can't read Lbox because in the catalog there are no spaces in this line
#Lbox = np.loadtxt(catalog, usecols=[1], max_rows=1, dtype=float)
Lbox = 1000
print('Lbox = ', Lbox, 'Mpc/h')

#work on the linear correlation function
#Set up a new set of parameters for CAMB
pars = camb.CAMBparams()
#This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency
pars.set_cosmology(H0=hubble*100, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)
pars.set_matter_power(redshifts=[z], kmax=3.0)
#Linear spectra
pars.NonLinear = camb.model.NonLinear_none
results = camb.get_results(pars)
kh, redshift, pk_lin = results.get_matter_power_spectrum(minkh=1e-4, maxkh=2, npoints = 10000)
s8 = np.array(results.get_sigma8())
print('sigma8 from model P(k) = ', results.get_sigma8())
plt.loglog(kh, pk_lin[0,:], color='r', label = 'z = %.3g'%redshift[0])
plt.xlabel('k/h',fontsize=10)
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
	f = lambda x : fr(x,R)
	xiR[i] = h.transform(f)[0] / (2*np.pi**2 * R)

plt.figure()
plt.plot(Rs,xiR*Rs**2)
plt.xlim(25,200)
#plt.title('Correlation function')
plt.xlabel('Mpc/h',fontsize=10)
plt.ylabel(r'$r^2\xi(r)$',fontsize=10)
plt.tight_layout()
plt.grid(True)
plt.show()

#now work on catalogs - measure P(k) and integrate it to get sigma8
for catalog in glob.glob('/data26s/comparat/simulations/GLAM/1Gpc2000x4000/CATALOGS/CatshortV.0'+snap+'.0'+realization+'.DAT'):
    print('reading masses from catalog ', catalog)
    data = np.loadtxt(catalog, skiprows=8, dtype=float, unpack=True)
    masses = data[7]
    Xcoord = data[0]
    Ycoord = data[1]
    Zcoord = data[2]
    coordinates_mass = [Xcoord, Ycoord, Zcoord, masses]
    coordinates_mass = np.transpose(coordinates_mass)    
    np.savetxt('coords_nbodykit'+snap+'_'+realization+'.txt',coordinates_mass,fmt='%.7g')
print(coordinates_mass)
print('mass = ',masses)

#Now I have a catalog with the poistion of the halos
#I can put them on a grid, FFT to get deltak and then P(k)
redshift = z
#cosmo = cosmology.Planck15
#Plin = cosmology.LinearPower(cosmo, redshift, transfer='EisensteinHu')
#b1 = 2.0
#cat = LogNormalCatalog(Plin=Plin, nbar=3e-4, BoxSize=1000., Nmesh=256, bias=b1, seed=42)
names = ['x','y','z','m']
cat = CSVCatalog('coords_nbodykit'+snap+'_'+realization+'.txt',names)
# add RSD
#line_of_sight = [0,0,1]
#cat['RSDPosition'] = cat['Position'] + cat['VelocityOffset'] * line_of_sight
cat['Position'] = cat['x'][:,None]*[1, 0, 0] + cat['y'][:,None] * [0, 1, 0] + cat['z'][:,None] * [0, 0, 1]
#cat['Mass'] = cat['m']
cat.attrs['BoxSize'] = Lbox
print(cat['Position'])
mesh = cat.to_mesh(Nmesh=637)
#density = mesh.preview(Nmesh=256, axes=(0,1))
#plt.imshow(density)
#plt.show()
r = FFTPower(mesh, mode='1d', dk=(2*np.pi/Lbox), kmin=(2*np.pi/Lbox))
Pk_measured = r.power
k_measured = Pk_measured['k']
print('k_meas = ', k_measured)
Pk_meas = Pk_measured['power'].real
Pk_data = [k_measured, Pk_meas]
Pk_data = np.transpose(Pk_data)
Pk_dat = pd.DataFrame(Pk_data)
Pk_dat.to_csv('power/data_'+snap+'_'*realization+'.txt')
plt.loglog(k_measured, Pk_meas, label='z=%.3g'%redshift)
plt.loglog(kh, pk_lin[0,:], color='r', label = 'model')
plt.xlabel('k/h Mpc',fontsize=10)
plt.legend()
plt.title('Matter power spectrum')
plt.ylabel('P(k)',fontsize=10)
plt.tight_layout()
plt.savefig('power_spectra/power_0'+snap+'_0'+realization+'.pdf')
#plt.show()
Pk_meas_shotnoise = Pk_measured['power'].real - Pk_measured.attrs['shotnoise']

#integrate Pk to get sigma8
R_filter = 8 #8 Mpc/h
#integrate until k=0.3 to stay in linear regime
isel = (k_measured<0.3)
k_measured = k_measured[isel]
Pk_meas = Pk_meas[isel]
tophat = 3*(np.sin(R_filter*k_measured)-R_filter*k_measured*np.cos(R_filter*k_measured))/(R_filter*k_measured)**3
integrand = k_measured**2*Pk_meas*tophat**2
sigma8_2= 1./(2.*np.pi**2)*integrate.simps(integrand,k_measured)
sigma8=np.sqrt(sigma8_2)
print('sigma8 from catalog = ',sigma8)

nbins=200
bins = np.logspace(12.1,14.2,nbins)
mass_number_tot, mass_bins = histogram(masses, bins=bins)
mass_bins_average, mass_bins, bin_number = stats.binned_statistic(masses, masses,  bins=bins, statistic='mean')
#calculate Xi from eq 22 of comparat17 paper
rmax = 50 #Mpc/h
dr = 0.1 #Mpc/h
volume = Lbox**3
iselect = (Xcoord > rmax)&(Xcoord< (Lbox-rmax))&(Ycoord > rmax)&(Ycoord< (Lbox-rmax))&(Zcoord > rmax)&(Zcoord< (Lbox-rmax))
bin_xi3D=np.arange(0, rmax, dr)
xi_tot=np.zeros((len(mass_bins_average), len(bin_xi3D)-1))
bias2 = np.zeros(len(mass_bins_average))
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
    # now does the pair counts :
    print('pair counting...')
    pairs=treeData.count_neighbors(treeRandoms, bin_xi3D)
    #t3 = time.time()
    DR=pairs[1:]-pairs[:-1]
    dV= 4*np.pi*(bin_xi3D[1:]**3 - bin_xi3D[:-1]**3 )/3.
    pairCount=nD*nR#-nD*(nD-1)/2.
    xi_tot[i,:] = DR*volume/(dV * pairCount) -1.  
    #np.savetxt(outf, xi)
    #plt.plot(bin_xi3D[1:],xi)
#    plt.yscale('log')
    #plt.show()
    rapporto=np.zeros(420)
    for j in range(420):
        rapporto[j] = xi_tot[i,79+j]/xiR[79+j]
    bias2[i] = np.average(rapporto)
    print(bias2[i])
 #   xis = append(xis,xi)
  #  bias2[i] = 1/200 * np.sum(xi/xiR)  
#    out.write("  M_low   M_high  corr_func\r\n") 
#    out.write("%.4g %.4g %.4g \r\n" %(mass_bins[i], mass_bins[i+1], xi))
#    out.close()
#bias = np.sqrt(bias2)

outf = open('Correlation_Function/corr_func_'+snap+'_'+realization+'.txt','w+')
outf.write('sigma8   rmax(Mpch)\n')
outf.write('%.5g  %.2g\n'%(sigma8,rmax))
outf.write('                     mass_bins\n')
dftot = pd.DataFrame(mass_bins_average,xi_tot,dtype=float)
dftot.to_csv(outf)
outf.close()
plt.figure()
plt.plot(mass_bins_average, bias2)
plt.xscale('log')
plt.xlabel(r'M $[M_\odot/h]$',fontsize=15)
plt.ylabel(r'b$^{2}$',fontsize=15)
plt.tight_layout()
plt.savefig('bias/bias_0'+snap+'_0'+realization+'.pdf')
outbias = 'bias/bias_'+snap+'_'+realization+'.txt'
bias_out = [mass_bins_average, bias2]
bias_out=np.transpose(bias_out)
biasfile = pd.DataFrame(bias_out, columns=['mass bin', 'bias'])
biasfile.to_csv(outbias)
#remove coords file, which is just a copy of data without the header
os.remove('coords_nbodykit'+snap+'_'+realization+'.txt')



