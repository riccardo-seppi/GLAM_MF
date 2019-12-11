import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
from scipy import stats
from scipy import histogram
from scipy import integrate
import camb
import sys
import os



def main():
    plt.figure()
    for mass_hist in glob.glob('mass_histogram0*.txt'):
        cosmo_params = np.loadtxt(mass_hist, skiprows=1, max_rows=1, dtype=float)
        z, Omega0, hubble = cosmo_params
        data_ = pd.read_csv(mass_hist,skiprows=[0,1,2,3,4,51,97,143,189,235], dtype=np.float64, sep=',')
        data = data_.to_numpy()
        mass_bins = data[225]
        MF_tot = data[226]
        plt.loglog(mass_bins, MF_tot, label='z=%.2f'%z)
        plt.xlabel(r'M $[M_\odot/h]$', fontsize = 20)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.ylabel(r'dn\dlnM $[(Mpc/h)^{-3}]$', fontsize = 20)
        plt.tight_layout()
        plt.grid(True)
   #     plt.legend()
        plt.savefig('total_MFs_together.pdf')
    plt.show()    
    snap = sys.argv[1]
    catalog = '/data26s/comparat/simulations/GLAM/1Gpc2000x4000/CATALOGS/CatshortV.0'+snap+'.0001.DAT'
    '''read here the cosmological parameters'''
    a = np.loadtxt(catalog, usecols=[2], skiprows=1, max_rows=1, dtype=float)
    z = 1/a - 1

    indir = 'bias/'
    cat = indir+'bias_'+snap+'_001.txt'
    data = pd.read_csv(cat, dtype = float, usecols =[1,2], skiprows = 1)
    data = np.transpose(data.values)
    mass_bins = data[0]            
    ncols = len(data[1])
    nlines = 44
    all_bias = np.zeros((nlines,ncols))
    bias_average2 = np.array([])
    bias_stdev = np.array([])
    a=0
    for catalog in glob.glob(indir+'bias_'+snap+'_0*.txt'):
        data = pd.read_csv(catalog, dtype = float, skiprows=1)
        data_tr = np.transpose(data.values)
        all_bias[a,:] = data_tr[2]
        a=a+1
    print(len(all_bias[0,:]))    
    for i in range(len(all_bias[0,:])):
        bias_average2 = np.append(bias_average2,np.mean(all_bias[:,i]))
        bias_stdev = np.append(bias_stdev,np.std(all_bias[:,i]))
    bias_average = np.sqrt(bias_average2)    
    bias_stdev = np.sqrt(bias_stdev)/np.sqrt(bias_average)
    bias_av_data = (list(mass_bins),list(bias_average),list(bias_stdev))
    outfile = 'bias/average/bias_mean_'+snap+'.csv'
    outf = open(outfile, 'w+')
    outf.write('Mass bin  mean bias  bias stdev\n')
    fil = pd.DataFrame(data = np.c_[bias_av_data])
    fil.to_csv(outf,float_format='%.5g')
    outf.close()

    plt.figure()
    plt.errorbar(mass_bins, bias_average, yerr = bias_stdev)
    plt.xscale('log')
    plt.title('bias at z = %.3g'%z)
    plt.ylabel('b', fontsize=16)
    plt.xlabel(r'M [M$_{\odot}$]',fontsize=16)
    plt.tight_layout()
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig('bias/average/bias_mean_'+snap+'_plot.pdf')
        
if __name__ == '__main__':
    main()
