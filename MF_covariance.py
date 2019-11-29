#Riccardo Seppi - MPE - HEG (2019) - 25 October
#This code reads halo masses from DM simulations (GLAM)
#builds HMF and fits them to models with fixed cosmological parameters

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import itertools as IT

#read the MFs
snap = sys.argv[1]
infile='mass_histogram0'+snap+'.txt'
print(infile)
outfigure ='correlation_figures/0'+snap+'_'
outfile = 'correlation_matrix/0'+snap+'_'
cosmo_params = np.loadtxt(infile, skiprows=1, max_rows=1, dtype=float)
z, Omega0, hubble = cosmo_params
print(cosmo_params)
params = {'flat': True, 'H0': hubble*100, 'Om0': Omega0, 'Ob0': 0.049, 'sigma8': 0.828, 'ns': 0.96}

#mass_data = np.loadtxt(infile, skiprows=6, dtype=float)
quartiles = np.array([0.05992, 0.1026, 0.159])
mass_data_ = pd.read_csv(infile, skiprows=[0,1,2,3,4,52,99,146,193], dtype=np.float64, sep=',')
print(mass_data_)
mass_data = mass_data_.to_numpy()
#mass_data = mass_data_.as_matrix()
print(mass_data)
#mass_data= np.transpose(mass_data)
#print(len(mass_data))
#print(type(mass_data[10]))
mass_bins_pl_1st = np.array(mass_data[0])
mass_bins_pl_2nd = mass_data[46]
mass_bins_pl_3rd = mass_data[92]
mass_bins_pl_4th = mass_data[138]
mass_bins_pl_tot = mass_data[184]
print(mass_bins_pl_2nd,mass_bins_pl_3rd,mass_bins_pl_4th,mass_bins_pl_tot)
print(mass_data[140])
mass_functions_25 = np.array(mass_data[1:45])
mass_functions_50 = mass_data[47:91]
mass_functions_75 = mass_data[93:137]
mass_functions_100 = mass_data[139:183]
mass_function_tot = mass_data[185]
print(type(mass_functions_25[2]))

#plot total MF
plt.loglog(mass_bins_pl_tot, mass_function_tot)
plt.xlabel(r'M $[M_\odot/h]$', fontsize = 15)
plt.ylabel(r'dn\dlnM $[(Mpc/h)^{-3}]$', fontsize = 15)
plt.tight_layout()
plt.grid(True)
plt.savefig(outfigure+'combined_mass_function.pdf')
plt.show()


fig, axes = plt.subplots(nrows=2,ncols=2,figsize=(10,10),sharey='all')
fig.suptitle('Mass Functions', fontsize=16)
axes[0,0].grid()
axes[0,1].grid()
axes[1,0].grid()
axes[1,1].grid()
axes[0,0].loglog(mass_bins_pl_1st, mass_functions_25[0],label='Xoff<%.3g'%quartiles[0])
axes[0,0].set_ylim([1e-7,1e-3])
axes[0,0].set_ylabel(r'dn\dlnM $[(Mpc/h)^{-3}]$')
axes[0,1].loglog(mass_bins_pl_2nd, mass_functions_50[0],label='%.3g<Xoff<%.3g'%(quartiles[0],quartiles[1]))
axes[0,1].set_ylim([1e-7,1e-3])
axes[1,0].loglog(mass_bins_pl_3rd, mass_functions_75[0],label='%.3g<Xoff<%.3g'%(quartiles[1],quartiles[2]))
axes[1,0].set_ylabel(r'dn\dlnM $[(Mpc/h)^{-3}]$')
axes[1,0].set_ylim([1e-7,1e-3])
axes[1,0].set_xlabel(r'M $[M_\odot/h]$')
axes[1,1].loglog(mass_bins_pl_4th, mass_functions_100[0],label='Xoff>%.3g'%quartiles[2])
axes[1,1].set_ylim([1e-7,1e-3])
axes[1,1].set_xlabel(r'M $[M_\odot/h]$')
axes[0,0].legend()
axes[0,1].legend()
axes[1,0].legend()
axes[1,1].legend()
#for i in range(44):
 #   axes[0,0].plot(mass_bins_pl_1st, mass_functions_25[i])
  #  axes[0,1].plot(mass_bins_pl_2nd, mass_functions_50[i])
   # axes[1,0].plot(mass_bins_pl_3rd, mass_functions_75[i])
    #axes[1,1].plot(mass_bins_pl_4th, mass_functions_100[i])
plt.savefig(outfigure+'mass_functions.pdf')
plt.show()

mass_func_25 = pd.DataFrame(mass_functions_25)
mass_func_50 = pd.DataFrame(mass_functions_50)
mass_func_75 = pd.DataFrame(mass_functions_75)
mass_func_100 = pd.DataFrame(mass_functions_100)
covariance_matrix_25 = mass_func_25.cov().to_numpy()
covariance_matrix_50 = mass_func_50.cov().to_numpy()
covariance_matrix_75 = mass_func_75.cov().to_numpy()
covariance_matrix_100 = mass_func_100.cov().to_numpy()
'''
print('mean', np.mean(mass_functions_25[:,100]))
print('cov mat', covariance_matrix_25[0])
for i in range(len(covariance_matrix_25[0])):
    for j in range(len(covariance_matrix_25[1])):
        covariance_matrix_25[i,j] = covariance_matrix_25[i,j]/np.mean(mass_functions_25[:,i])/np.mean(mass_functions_25[:,j])
        covariance_matrix_50[i,j] = covariance_matrix_50[i,j]/np.mean(mass_functions_50[:,i])/np.mean(mass_functions_50[:,j])
        covariance_matrix_75[i,j] = covariance_matrix_75[i,j]/np.mean(mass_functions_75[:,i])/np.mean(mass_functions_75[:,j])
        covariance_matrix_100[i,j] = covariance_matrix_100[i,j]/np.mean(mass_functions_100[:,i])/np.mean(mass_functions_100[:,j])
'''
np.savetxt(outfile+'covariance_25.txt',covariance_matrix_25)
np.savetxt(outfile+'covariance_50.txt',covariance_matrix_50)
np.savetxt(outfile+'covariance_75.txt',covariance_matrix_75)
np.savetxt(outfile+'covariance_100.txt',covariance_matrix_100)

#plot covariance matrix
extent = [mass_bins_pl_tot[0],mass_bins_pl_tot[198],mass_bins_pl_tot[0],mass_bins_pl_tot[198]]
vmin, vmax = -1.0, 1.0
f, axes = plt.subplots(2,2,figsize=(19, 15))
im=axes[0,0].imshow(covariance_matrix_25, extent = extent)
clim=im.properties()['clim']
axes[0,1].imshow(covariance_matrix_50, clim=clim, extent = extent)
axes[1,0].imshow(covariance_matrix_75, clim=clim, extent = extent)
axes[1,1].imshow(covariance_matrix_100, clim=clim, extent = extent)
#plt.xticks(range(df.shape[1]), df.columns, fontsize=14, rotation=45)
#plt.yticks(range(df.shape[1]), df.columns, fontsize=14)
cb = f.colorbar(im, ax=axes.ravel().tolist())
cb.ax.tick_params(labelsize=14)
f.suptitle('Covariance Matrix', fontsize=16)
plt.savefig(outfigure+'covariance_matrix.pdf')
plt.show()

#print(covariance_matrix)

correlation_matrix_25=mass_func_25.corr()
correlation_matrix_50=mass_func_50.corr()
correlation_matrix_75=mass_func_75.corr()
correlation_matrix_100=mass_func_100.corr()
correlation_matrix_25.to_csv(outfile+'correlation_25.csv')
correlation_matrix_50.to_csv(outfile+'correlation_50.csv')
correlation_matrix_75.to_csv(outfile+'correlation_75.csv')
correlation_matrix_100.to_csv(outfile+'correlation_100.csv')
f, axes = plt.subplots(2,2,figsize=(19, 15))
im=axes[0,0].imshow(correlation_matrix_25, extent = extent)
clim=im.properties()['clim']
axes[0,1].imshow(correlation_matrix_50, clim=clim, extent = extent)
axes[1,0].imshow(correlation_matrix_75, clim=clim, extent = extent)
axes[1,1].imshow(correlation_matrix_100, clim=clim, extent = extent)
#plt.xticks(range(df.shape[1]), df.columns, fontsize=14, rotation=45)
#plt.yticks(range(df.shape[1]), df.columns, fontsize=14)
cb = f.colorbar(im, ax=axes.ravel().tolist())
cb.ax.tick_params(labelsize=14)
f.suptitle('Correlation Matrix', fontsize=16)
plt.savefig(outfigure+'correl_matrix.pdf')
plt.show()






