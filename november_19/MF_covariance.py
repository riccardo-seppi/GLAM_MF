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
mass_data_ = pd.read_csv(infile, skiprows=[0,1,2,3,4,52,99,146], dtype=np.float64, sep=',')
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
print(mass_bins_pl_2nd,mass_bins_pl_3rd,mass_bins_pl_4th)
print(mass_data[140])
mass_functions_25 = np.array(mass_data[1:45])
mass_functions_50 = mass_data[47:91]
mass_functions_75 = mass_data[93:137]
mass_functions_100 = mass_data[139:183]
print(type(mass_functions_25[2]))
plt.plot(mass_bins_pl_1st, mass_functions_25[0])
#plt.ylim([1e-8,1e-2])
plt.show()
fig, axes = plt.subplots(nrows=2,ncols=2,figsize=(7,7))
axes[0,0].loglog(mass_bins_pl_1st, mass_functions_25[0])
axes[0,1].loglog(mass_bins_pl_2nd, mass_functions_50[0])
axes[1,0].loglog(mass_bins_pl_3rd, mass_functions_75[0])
axes[1,1].loglog(mass_bins_pl_4th, mass_functions_100[0])
#for i in range(44):
 #   axes[0,0].plot(mass_bins_pl_1st, mass_functions_25[i])
  #  axes[0,1].plot(mass_bins_pl_2nd, mass_functions_50[i])
   # axes[1,0].plot(mass_bins_pl_3rd, mass_functions_75[i])
    #axes[1,1].plot(mass_bins_pl_4th, mass_functions_100[i])
plt.grid(True)
plt.savefig(outfigure+'mass_functions.pdf')
plt.show()

'''
#transpose the MF becuase np.cov wants in each row a variable and in each column different observations of that variable
mass_functions_25 = np.transpose(mass_functions_25)
mass_functions_50 = np.transpose(mass_functions_50)
mass_functions_75 = np.transpose(mass_functions_75)
mass_functions_100 = np.transpose(mass_functions_100)
'''
covariance_matrix_25 = np.cov(mass_functions_25)
covariance_matrix_50 = np.cov(mass_functions_50)
covariance_matrix_75 = np.cov(mass_functions_75)
covariance_matrix_100 = np.cov(mass_functions_100)
np.savetxt(outfile+'covariance_25.txt',covariance_matrix_25)
np.savetxt(outfile+'covariance_50.txt',covariance_matrix_50)
np.savetxt(outfile+'covariance_75.txt',covariance_matrix_75)
np.savetxt(outfile+'covariance_100.txt',covariance_matrix_100)

#plot covariance matrix
f, axes = plt.subplots(2,2,figsize=(19, 15))
im=axes[0,0].imshow(covariance_matrix_25)
clim=im.properties()['clim']
axes[0,1].imshow(covariance_matrix_50, clim=clim)
axes[1,0].imshow(covariance_matrix_75, clim=clim)
axes[1,1].imshow(covariance_matrix_100, clim=clim)
#plt.xticks(range(df.shape[1]), df.columns, fontsize=14, rotation=45)
#plt.yticks(range(df.shape[1]), df.columns, fontsize=14)
cb = f.colorbar(im, ax=axes.ravel().tolist())
cb.ax.tick_params(labelsize=14)
f.suptitle('Covariance Matrix', fontsize=16)
plt.savefig(outfigure+'covariance_matrix.pdf')
plt.show()

#print(covariance_matrix)
mass_func_25 = pd.DataFrame(mass_functions_25)
mass_func_50 = pd.DataFrame(mass_functions_50)
mass_func_75 = pd.DataFrame(mass_functions_75)
mass_func_100 = pd.DataFrame(mass_functions_100)
correlation_matrix_25=mass_func_25.corr()
correlation_matrix_50=mass_func_50.corr()
correlation_matrix_75=mass_func_75.corr()
correlation_matrix_100=mass_func_100.corr()
mass_func_25.to_csv(outfile+'correlation_25.csv')
mass_func_50.to_csv(outfile+'correlation_50.csv')
mass_func_75.to_csv(outfile+'correlation_75.csv')
mass_func_100.to_csv(outfile+'correlation_100.csv')
f, axes = plt.subplots(2,2,figsize=(19, 15))
im=axes[0,0].imshow(correlation_matrix_25)
clim=im.properties()['clim']
axes[0,1].imshow(correlation_matrix_50, clim=clim)
axes[1,0].imshow(correlation_matrix_75, clim=clim)
axes[1,1].imshow(correlation_matrix_100, clim=clim)
#plt.xticks(range(df.shape[1]), df.columns, fontsize=14, rotation=45)
#plt.yticks(range(df.shape[1]), df.columns, fontsize=14)
cb = f.colorbar(im, ax=axes.ravel().tolist())
cb.ax.tick_params(labelsize=14)
f.suptitle('Correlation Matrix', fontsize=16)
plt.savefig(outfigure+'correl_matrix.pdf')
plt.show()






