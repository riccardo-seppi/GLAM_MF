#Riccardo Seppi - MPE - HEG (2019) - 25 October
#This code reads halo masses from DM simulations (GLAM)
#builds covariance matrix between different MF at the same z, from different realizations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import itertools as IT

#define function that rebins data in order to have LIMIT halos in each bin
def halos_rebin(array,array_mf,limit,bins):
    count = 0
    density_count = 0
    var = 0
    array_new = []
    array_mf_new = []
    merged_bins = []
    new_bins =[]
    for a in range(len(array)):
        count += array[a]
        density_count += array_mf[a]
        var += 1
        if(count>=limit):
            newbinvalue = 10**((np.log10(bins[a])+np.log10(bins[a-(var-1)]))/2)
            new_bins = np.append(new_bins,newbinvalue) 
            array_new = np.append(np.array([array_new]),np.array([count]))
            array_mf_new = np.append(np.array([array_mf_new]),np.array([density_count/var]))            
            merged_bins = np.append(merged_bins, var) #it tells how many bins have been merged 
            count=0
            density_count = 0
            var = 0
#output1: array with the new halo counts in each bin
#output2: array with new MF values in the new bins
#output3: new bin values
#output4: number of merged bins as a sequence  
    return array_new, array_mf_new, new_bins, merged_bins


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

#define the quartiles
quartiles = np.array([0.05992, 0.1026, 0.159])
mass_data_ = pd.read_csv(infile, skiprows=[0,1,2,3,4,52,99,146,193,240], dtype=np.float64, sep=',')
print(mass_data_)
mass_data = mass_data_.to_numpy()
print(mass_data)
mass_bins_pl_1st = np.array(mass_data[0])
mass_bins_pl_2nd = mass_data[46]
mass_bins_pl_3rd = mass_data[92]
mass_bins_pl_4th = mass_data[138]
mass_bins_pl_real = mass_data[184]
mass_bins_pl_tot = mass_data[230]

mass_functions_25_ = mass_data[1:45]
mass_functions_50_ = mass_data[47:91]
mass_functions_75_ = mass_data[93:137]
mass_functions_100_ = mass_data[139:183]
mass_functions_real_ = mass_data[185:229]
mass_function_tot_ = mass_data[231]
print(type(mass_functions_25_[2]))

#plot total MF
plt.loglog(mass_bins_pl_tot, mass_function_tot_)
plt.xlabel(r'M $[M_\odot/h]$', fontsize = 15)
plt.ylabel(r'dn\dlnM $[(Mpc/h)^{-3}]$', fontsize = 15)
plt.tight_layout()
plt.grid(True)
plt.savefig(outfigure+'combined_mass_function.pdf')
plt.show()

#plot MFs with different Xoff (just 1 case)
fig, axes = plt.subplots(nrows=2,ncols=2,figsize=(10,10),sharey='all')
fig.suptitle('Mass Functions', fontsize=16)
axes[0,0].xaxis.set_tick_params(labelsize=20)
axes[0,0].grid()
axes[0,1].grid()
axes[1,0].grid()
axes[1,1].grid()
axes[0,0].loglog(mass_bins_pl_1st, mass_functions_25_[0],label='Xoff<%.3g'%quartiles[0])
axes[0,0].set_ylim([1e-7,1e-3])
axes[0,0].set_ylabel(r'dn\dlnM $[(Mpc/h)^{-3}]$')
axes[0,1].loglog(mass_bins_pl_2nd, mass_functions_50_[0],label='%.3g<Xoff<%.3g'%(quartiles[0],quartiles[1]))
axes[0,1].set_ylim([1e-7,1e-3])
axes[1,0].loglog(mass_bins_pl_3rd, mass_functions_75_[0],label='%.3g<Xoff<%.3g'%(quartiles[1],quartiles[2]))
axes[1,0].set_ylabel(r'dn\dlnM $[(Mpc/h)^{-3}]$')
axes[1,0].set_ylim([1e-7,1e-3])
axes[1,0].set_xlabel(r'M $[M_\odot/h]$')
axes[1,1].loglog(mass_bins_pl_4th, mass_functions_100_[0],label='Xoff>%.3g'%quartiles[2])
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

#now read the halos_per_bin file, in order to rebin halos to have a significant signal
#Note (4 dec 2019): make the rebin only on 1 MF and use the same for all the others
#so that you have always the same number of bins
#this means that in some realizations you might have a lower number of clusters than the
#limit you have set, but since they are all at the same z this should not be important
#but we can discuss about this
infile2 ='halos_per_bin/number_per_bin_'+snap+'_001.txt'
mass_rebin_file = pd.read_csv(infile2, dtype=np.float64, usecols = [1,2,3,4,5,6])
#print(mass_rebin_file.values)

mass_reb_tran = np.transpose(mass_rebin_file.values)
mass_rebin=mass_reb_tran[0]
halos1=mass_reb_tran[1]
halos2=mass_reb_tran[2]
halos3=mass_reb_tran[3]
halos4=mass_reb_tran[4]
halos_real=mass_reb_tran[5] #the index real refers to 'realization': one of the multiple realizations at a fixed z value. I will need it for the bias in MF_bias.py
print(mass_rebin) 

#set the limit and define how many new bins you will need for each Xoff value (1 2 3 4)
limit = 500
nlines = len(mass_functions_25_[:,0])
ncols1 = len(halos_rebin(halos1,mass_functions_25_[0,:],limit,mass_bins_pl_1st)[0])
ncols2 = len(halos_rebin(halos2,mass_functions_50_[0,:],limit,mass_bins_pl_2nd)[0])
ncols3 = len(halos_rebin(halos3,mass_functions_75_[0,:],limit,mass_bins_pl_3rd)[0])
ncols4 = len(halos_rebin(halos4,mass_functions_100_[0,:],limit,mass_bins_pl_4th)[0])
ncols_real = len(halos_rebin(halos_real,mass_functions_real_[0,:],limit,mass_bins_pl_real)[0])

newbins25 = halos_rebin(halos1,mass_functions_25_[0,:],limit,mass_bins_pl_1st)[2]
newbins50 = halos_rebin(halos2,mass_functions_50_[0,:],limit,mass_bins_pl_2nd)[2]
newbins75 = halos_rebin(halos3,mass_functions_75_[0,:],limit,mass_bins_pl_3rd)[2]
newbins100 = halos_rebin(halos4,mass_functions_100_[0,:],limit,mass_bins_pl_4th)[2]
newbins_real = halos_rebin(halos_real,mass_functions_real_[0,:],limit,mass_bins_pl_real)[2]

merged25 = halos_rebin(halos1,mass_functions_25_[0,:],limit,mass_bins_pl_1st)[3]
merged50 = halos_rebin(halos2,mass_functions_50_[0,:],limit,mass_bins_pl_2nd)[3]
merged75 = halos_rebin(halos3,mass_functions_75_[0,:],limit,mass_bins_pl_3rd)[3]
merged100 = halos_rebin(halos4,mass_functions_100_[0,:],limit,mass_bins_pl_4th)[3]
merged_real = halos_rebin(halos_real,mass_functions_real_[0,:],limit,mass_bins_pl_real)[3]

count25 = np.zeros((nlines,ncols1))
count50 = np.zeros((nlines,ncols2))
count75 = np.zeros((nlines,ncols3))
count100 = np.zeros((nlines,ncols4))
count_real = np.zeros((nlines,ncols_real))
mass_functions_25 = np.zeros((nlines,ncols1))
mass_functions_50 = np.zeros((nlines,ncols2))
mass_functions_75 = np.zeros((nlines,ncols3))
mass_functions_100 = np.zeros((nlines,ncols4))
mass_functions_real = np.zeros((nlines,ncols_real))

print(nlines)
print(ncols1)
print(np.array([halos_rebin(halos1,mass_functions_25_[0,:],limit,mass_bins_pl_1st)[1]]).shape)

#NOW BIN EACH SIMULATION
for i in range(len(mass_functions_25_[:,0])):
    #count25 = np.append(count25, np.array([halos_rebin(halos1,mass_functions_25_[i,:],limit)[0]]), axis = 0)
    count25[i,:] = halos_rebin(halos1,mass_functions_25_[i,:],limit,mass_bins_pl_1st)[0]
    mass_functions_25[i,:] = halos_rebin(halos1,mass_functions_25_[i,:],limit,mass_bins_pl_1st)[1]
    #mass_functions_25 = np.append(mass_functions_25, np.array([halos_rebin(halos1,mass_functions_25_[i,:],limit)[1]]),axis = 0)    
   # count50 = np.append(count50, halos_rebin(halos2,mass_functions_50_[i,:],limit)[0], axis = 0)
   # mass_functions_50 = np.append(mass_functions_50, halos_rebin(halos2,mass_functions_50_[i,:],limit)[1], axis = 0)    
   # count75 = np.append(count75, halos_rebin(halos3,mass_functions_75_[i,:],limit)[0], axis = 0)
   # mass_functions_75 = np.append(mass_functions_75, halos_rebin(halos3,mass_functions_75_[i,:],limit)[1], axis = 0)    
   # count100 = np.append(count100, halos_rebin(halos4,mass_functions_100_[i,:],limit)[0], axis = 0)
   # mass_functions_100 = np.append(mass_functions_100, halos_rebin(halos4,mass_functions_100_[i,:],limit)[1], axis = 0)
#mass_functions_25 = np.c_[mass_functions_25]
    count50[i,:] = halos_rebin(halos2,mass_functions_50_[i,:],limit,mass_bins_pl_2nd)[0]
    mass_functions_50[i,:] = halos_rebin(halos2,mass_functions_50_[i,:],limit,mass_bins_pl_2nd)[1]
    count75[i,:] = halos_rebin(halos3,mass_functions_75_[i,:],limit,mass_bins_pl_3rd)[0]
    mass_functions_75[i,:] = halos_rebin(halos3,mass_functions_75_[i,:],limit,mass_bins_pl_3rd)[1]
    count100[i,:] = halos_rebin(halos4,mass_functions_100_[i,:],limit,mass_bins_pl_4th)[0]
    mass_functions_100[i,:] = halos_rebin(halos4,mass_functions_100_[i,:],limit,mass_bins_pl_4th)[1]
    count_real[i,:] = halos_rebin(halos_real,mass_functions_real_[i,:],limit,mass_bins_pl_real)[0]
    mass_functions_real[i,:] = halos_rebin(halos_real,mass_functions_real_[i,:],limit,mass_bins_pl_real)[1]
print(mass_functions_25)    

#save to a file the rebinned data
rebinned_data1 = (list(newbins25), list(merged25), list(np.transpose(count25)), list(np.transpose(mass_functions_25)))
rebinned_data2 = (list(newbins50), list(merged50), list(np.transpose(count50)), list(np.transpose(mass_functions_50)))
rebinned_data3 = (list(newbins75), list(merged75), list(np.transpose(count75)), list(np.transpose(mass_functions_75)))
rebinned_data4 = (list(newbins100), list(merged100), list(np.transpose(count100)), list(np.transpose(mass_functions_100)))
rebinned_data_real = (list(newbins_real), list(merged_real), list(np.transpose(count_real)), list(np.transpose(mass_functions_real)))

outre1 = 'rebinned_data/'+snap+'_Xoff1.csv'
outreb1 = open(outre1, 'w+')
outreb1.write('#1st col: mass bin     2nd col: merged_bins 3-47 col: halos per bin in each realization   48-92 col: dndlnM per bin in each realization\n')
fil1 = pd.DataFrame(data=np.c_[rebinned_data1])
fil1.to_csv(outreb1,float_format='%.5g')
outreb1.close()

outre2 = 'rebinned_data/'+snap+'_Xoff2.csv'
outreb2 = open(outre2, 'w+')
outreb2.write('#1st col: mass bin     2nd col: merged_bins 3-47 col: halos per bin in each realization   48-92 col: dndlnM per bin in each realization\n')
fil2 = pd.DataFrame(data=np.c_[rebinned_data2])
fil2.to_csv(outreb2,float_format='%.5g')
outreb2.close()

outre3 = 'rebinned_data/'+snap+'_Xoff3.csv'
outreb3 = open(outre3, 'w+')
outreb3.write('#1st col: mass bin     2nd col: merged_bins 3-47 col: halos per bin in each realization   48-92 col: dndlnM per bin in each realization\n')
fil3 = pd.DataFrame(data=np.c_[rebinned_data3])
fil3.to_csv(outreb3,float_format='%.5g')
outreb3.close()

outre4 = 'rebinned_data/'+snap+'_Xoff4.csv'
outreb4 = open(outre4, 'w+')
outreb4.write('#1st col: mass bin     2nd col: merged_bins 3-47 col: halos per bin in each realization   48-92 col: dndlnM per bin in each realization\n')
fil4 = pd.DataFrame(data=np.c_[rebinned_data4])
fil4.to_csv(outreb4,float_format='%.5g')
outreb4.close()

outre_real = 'rebinned_data/'+snap+'_full_sample.csv'
outreb_real = open(outre_real, 'w+')
outreb_real.write('#1st col: mass bin     2nd col: merged_bins 3-47 col: halos per bin in each realization   48-92 col: dndlnM per bin in each realization\n')
fil_real = pd.DataFrame(data=np.c_[rebinned_data_real])
fil_real.to_csv(outreb_real,float_format='%.5g')
outreb_real.close()

#create covariance matrix
mass_func_25 = pd.DataFrame(mass_functions_25)
mass_func_50 = pd.DataFrame(mass_functions_50)
mass_func_75 = pd.DataFrame(mass_functions_75)
mass_func_100 = pd.DataFrame(mass_functions_100)
covariance_matrix_25 = mass_func_25.cov().to_numpy()
covariance_matrix_50 = mass_func_50.cov().to_numpy()
covariance_matrix_75 = mass_func_75.cov().to_numpy()
covariance_matrix_100 = mass_func_100.cov().to_numpy()
#and save them
np.savetxt(outfile+'covariance_25.txt',covariance_matrix_25, fmt='%.5g')
np.savetxt(outfile+'covariance_50.txt',covariance_matrix_50, fmt='%.5g')
np.savetxt(outfile+'covariance_75.txt',covariance_matrix_75, fmt='%.5g')
np.savetxt(outfile+'covariance_100.txt',covariance_matrix_100, fmt='%.5g')

#plot covariance matrix
extent1 = [newbins25[0],newbins25[ncols1-1],newbins25[0],newbins25[ncols1-1]]
extent2 = [newbins50[0],newbins50[ncols2-1],newbins50[0],newbins50[ncols2-1]]
extent3 = [newbins75[0],newbins75[ncols3-1],newbins75[0],newbins75[ncols3-1]]
extent4 = [newbins100[0],max(newbins100),newbins100[0],max(newbins100)]
vmin, vmax = -1.0, 1.0
#X1,Y1 = np.meshgrid(np.array(newbins25),np.array(newbins25))
#X2,Y2 = np.meshgrid(np.array(newbins50),np.array(newbins50))
#X3,Y3 = np.meshgrid(np.array(newbins75),np.array(newbins75))
#X4,Y4 = np.meshgrid(np.array(newbins100),np.array(newbins100))
f, axes = plt.subplots(2,2,figsize=(19, 15))
im=axes[0,0].imshow(covariance_matrix_25, extent = extent1)
#im=axes[0,0].pcolormesh(X1,Y1, covariance_matrix_25)
#plt.sca(axes[0,0])
#plt.xticks(np.array(newbins25))
#plt.yticks(np.array(newbins25))
clim=im.properties()['clim']
axes[0,1].imshow(covariance_matrix_50, clim=clim, extent = extent2)
axes[1,0].imshow(covariance_matrix_75, clim=clim, extent = extent3)
axes[1,1].imshow(covariance_matrix_100, clim=clim, extent = extent4)
#axes[0,1].pcolormesh(X2,Y2, covariance_matrix_50)
#axes[1,0].pcolormesh(X3,Y3, covariance_matrix_75)
#axes[1,1].pcolormesh(X4,Y4, covariance_matrix_100)
#plt.sca(axes[0,1])
#plt.xticks(np.array(newbins50))
#plt.yticks(np.array(newbins50))
#plt.sca(axes[1,0])
#plt.xticks(np.array(newbins75))
#plt.yticks(np.array(newbins75))
#plt.sca(axes[1,1])
#plt.xticks(np.array(newbins100))
#plt.yticks(np.array(newbins100))
cb = f.colorbar(im, ax=axes.ravel().tolist())
cb.ax.tick_params(labelsize=14)
f.suptitle('Covariance Matrix', fontsize=16)
plt.savefig(outfigure+'covariance_matrix.pdf')
plt.show()

#create correlation matrix, save and plot

correlation_matrix_25=mass_func_25.corr()
correlation_matrix_50=mass_func_50.corr()
correlation_matrix_75=mass_func_75.corr()
correlation_matrix_100=mass_func_100.corr()
correlation_matrix_25.to_csv(outfile+'correlation_25.csv', float_format='%.5g')
correlation_matrix_50.to_csv(outfile+'correlation_50.csv', float_format='%.5g')
correlation_matrix_75.to_csv(outfile+'correlation_75.csv', float_format='%.5g')
correlation_matrix_100.to_csv(outfile+'correlation_100.csv', float_format='%.5g')
f, axes = plt.subplots(2,2,figsize=(19, 15))
im=axes[0,0].imshow(correlation_matrix_25, extent = extent1)
clim=im.properties()['clim']
axes[0,1].imshow(correlation_matrix_50, clim=clim, extent = extent2)
axes[1,0].imshow(correlation_matrix_75, clim=clim, extent = extent3)
axes[1,1].imshow(correlation_matrix_100, clim=clim, extent = extent4)
#plt.xticks(range(df.shape[1]), df.columns, fontsize=14, rotation=45)
#plt.yticks(range(df.shape[1]), df.columns, fontsize=14)
cb = f.colorbar(im, ax=axes.ravel().tolist())
cb.ax.tick_params(labelsize=14)
f.suptitle('Correlation Matrix', fontsize=16)
plt.savefig(outfigure+'correl_matrix.pdf')
plt.show()





