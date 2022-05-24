from one.api import ONE
import ibllib.atlas as atlas
from brainbox.io.one import SpikeSortingLoader
from ibllib.atlas import AllenAtlas
from ibllib.atlas.regions import BrainRegions

import os
import numpy as np
from scipy.io import savemat, loadmat
from pathlib import Path
from copy import deepcopy
import random
from itertools import combinations
import subprocess
import gc
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from collections import Counter

# matlab function for spectral granger
#https://github.com/oliche/spectral-granger/blob/master/matlab/spectral_granger.m

one_online = ONE()
ba = AllenAtlas()
br = BrainRegions()

T_BIN = 0.005  # time bin size in seconds (5 ms)

align = {'block':'stim on',
         'stim':'stim on',
         'choice':'motion on',
         'action':'motion on',
         'fback':'feedback'}


def get_average_PETH_per_region_per_condition(mapping='Swanson'):

    DD = {}
      
    
    for split in align:
    
        R = np.load('/home/mic/paper-brain-wide-map/'
                    f'manifold_analysis/bwm_psths_{split}.npy',
                    allow_pickle=True).flat[0]

        nt, nclus, nobs = R['ws'][0].shape

        # possible that certain psths are nan if there were not enough trials
        # remove those insertions
        
        nanins = [k for k in range(len(R['ws'])) if np.isnan(R['ws'][k]).any()]   
        
        ws = [np.concatenate([R['ws'][k][i] for k in range(len(R['ws'])) 
              if k not in nanins]) for i in range(nt)]
              
        ids = np.concatenate([R['ids'][k] for k in range(len(R['ws'])) 
                              if k not in nanins])  

        acs = br.id2acronym(ids,mapping=mapping)    
                     
        acs = np.array(acs)
        
        # get psths per region, then reduce to 3 dims via PCA
        regs = Counter(acs)
        #print(regs)
        
        D = {}
        
        for reg in regs:
            if reg in ['void','root']:
                continue
            if sum(acs == reg) < 200:
                continue
            
            print(split, reg)
            
            ws_ = [y[acs == reg] for y in ws]               
            wsc = np.concatenate(ws_,axis=1)
            D[reg] = np.mean(wsc,axis=0)
            
        DD[split] = D
        
    return DD
    
    
    
def save_data_for_matlab():

    '''
    For each region average PETHs across cells; 
    compute Granger for the 
    
    # time x trial x channel
    # two channels are region pairs
    # trials are PETH types
    '''

    DD = np.load('/mnt/8cfe1683-d974-40f3-a20b-b217cad4722a/'
                'granger/PETH/avPETH.npy',allow_pickle=True).flat[0]
                       
    regs_ = Counter(np.concatenate([list(DD[x].keys()) for x in DD]))
    regs = [x for x in regs_ if regs_[x] == 5]
    
    R = {}
    for reg in regs:
        R[reg] = []
        for split in align:
            R[reg].append(DD[split][reg])       
        R[reg] = np.concatenate(R[reg])

    # time x trial x channel

    Xs, ps = [], []
    for p in combinations(range(len(regs)),2):

        x = np.array([R[regs[p[0]]], R[regs[p[1]]]]).T
        x = np.expand_dims(x,axis=1)
        
        Xs.append(x)
        ps.append(p)
     
    
    s = (f'/mnt/8cfe1683-d974-40f3-a20b-b217cad4722a/'
        f'granger/PETH/')
       
    np.save(s+'regs.npy',regs)
     
    k = 0    
    for p in ps:
    
        s2 = s +'data/'+ str(p)
        Path(s2).mkdir(parents=True, exist_ok=True)
        savemat(s2+'/'+'data.mat',{'data':Xs[k]})
        
        s2 = s +'results/'+ str(p)
        Path(s2).mkdir(parents=True, exist_ok=True)         
        k += 1


def run_GC_matlab():

    
    matlab_path = '/mnt/8cfe1683-d974-40f3-a20b-b217cad4722a/MATLAB/bin/matlab'
    function_path = ('/mnt/8cfe1683-d974-40f3-a20b-b217cad4722a/granger/'
                     'spectral-granger-master/matlab/')

    # run for all regions of eid  
    b = f'/mnt/8cfe1683-d974-40f3-a20b-b217cad4722a/granger/PETH/'
    #eid = '15f742e1-1043-45c9-9504-f1e8a53c1744'  
    
    regs = np.load('/mnt/8cfe1683-d974-40f3-a20b-b217cad4722a/'
                   f'granger/PETH/regs.npy')
                   
    for p in combinations(range(len(regs)),2):

        path_to_data = b+'/data/'+str(p)+'/'+'data.mat'
        path_to_res =  b+'/results/'+str(p)+'/'
                 
        os.chdir(Path(function_path).parent)

        command = [matlab_path,
         '-nodisplay',
         '-nosplash',
         '-nodesktop',
         '-r',
         (f'cd("{function_path}");spectral_granger("{path_to_data}",'
         f'"{path_to_res}",{int(1/T_BIN)},1);exit')]

        output = subprocess.check_output(command)
        #print(p,tr_type) 



def get_max_GC_for_all_region_pairs():

    D = {}

    b = f'/mnt/8cfe1683-d974-40f3-a20b-b217cad4722a/granger/PETH'
    regs = np.load(b+'/regs.npy')
    for p in combinations(range(len(regs)),2):
    
        res_path = (b+f'/results/{str(p)}/M.mat')
               
        M = loadmat(res_path)['M'][0][0]        
        grc = M[3]
        
        D[regs[p[0]]+'_'+regs[p[1]]] = [max(grc[:,0,1]), max(grc[:,1,0])]
    


'''
#############
plotting
#############
'''


def plot_res(combi):

    '''
    # compare GC, coherence and PSD across engaged/disengaged trials
    for region pair combi
    '''

    #combi = (0, 1)
    b = f'/mnt/8cfe1683-d974-40f3-a20b-b217cad4722a/granger/PETH'
    regs = np.load(b+'/regs.npy')

    Path(b+'/figs/').mkdir(parents=True, exist_ok=True) 
    plt.figure(figsize=(11,5))
    
    row = 0
    ncols = 3
    
    axs = []
    k = 0
        
    res_path = (b+f'/results/{str(combi)}/M.mat')
           
    M=loadmat(res_path)['M'][0][0]
           
    freqs = M[0][0] # frequencies
    pspec = M[1] # power spectral density per channel
    coherence = M[2] # coherence
    grc = M[3] # conditional Granger causality (i to j on k)
    fs = M[4] # frequency of sampling    
    

    # also plot time series
    DD = np.load('/mnt/8cfe1683-d974-40f3-a20b-b217cad4722a/'
                'granger/PETH/avPETH.npy',allow_pickle=True).flat[0]
                       
    regs_ = Counter(np.concatenate([list(DD[x].keys()) for x in DD]))
    regs = [x for x in regs_ if regs_[x] == 5]
    
    R = {}
    for reg in regs:
        R[reg] = []
        for split in align:
            R[reg].append(DD[split][reg])       
        R[reg] = np.concatenate(R[reg])

    
    # power spectral density of channel 1
    axs.append(plt.subplot(1,4,1))
    axs[k].plot(freqs, pspec[:,0],label=regs[combi[0]])
    axs[k].plot(freqs, pspec[:,1],label=regs[combi[1]])
    axs[k].set_xlabel('frequency [Hz]')
    axs[k].set_ylabel('firing rate psd')
    axs[k].set_title('power spectral density')
    axs[k].legend()
    k+=1
   
    # coherence of channel 1 with 2
    axs.append(plt.subplot(1,4,2))
    axs[k].plot(freqs, coherence[:,0,1],
        label=f'{regs[combi[0]]} with {regs[combi[1]]}',alpha=0.5)
    #axs[k].set_title(tr_type+' trials')
    axs[k].set_ylabel('coherence')
    axs[k].set_xlabel('frequency [Hz]')
    axs[k].legend()
    k+=1

    # GC, channel 1 to channel 2
    axs.append(plt.subplot(1,4,3))
    axs[k].plot(freqs, grc[:,0,1],
        label=f'{regs[combi[0]]} to {regs[combi[1]]}',alpha=0.5)    
    axs[k].plot(freqs, grc[:,1,0],label=f'{regs[combi[1]]} to {regs[combi[0]]}',
             alpha=0.5)
    axs[k].set_ylabel('GC')
    axs[k].set_xlabel('frequency [Hz]')
    axs[k].legend()
    axs[k].set_title('Granger per freq')
    k+=1    
    
    # time series
    axs.append(plt.subplot(1,4,4))
    for reg in [regs[combi[0]],regs[combi[1]]]:
        xs = np.arange(len(R[reg]))* T_BIN
        axs[k].plot(xs, R[reg],
            label=reg,alpha=0.5)    

    axs[k].set_ylabel('firing rate')
    axs[k].set_xlabel('time [sec]')
    axs[k].legend()
    axs[k].set_title('concatenated PETHs')    


    plt.suptitle(f'regs: {regs[combi[0]]}, {regs[combi[1]]}')
    plt.tight_layout() 
    plt.savefig(b+'/figs/'+
                f'/{str(combi)}_{regs[combi[0]]}_{regs[combi[1]]}.png')
    plt.close()



