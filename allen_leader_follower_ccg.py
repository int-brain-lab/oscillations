from one.api import ONE
from reproducible_ephys_processing import bin_spikes2D
from ibllib.atlas import AllenAtlas
from ibllib.atlas.regions import BrainRegions
from brainbox.io.one import SpikeSortingLoader

from ibllib.atlas.plots import prepare_lr_data
from ibllib.atlas.plots import plot_scalar_on_flatmap, plot_scalar_on_slice
from ibllib.atlas import FlatMap
from ibllib.atlas.flatmaps import plot_swanson

from scipy import optimize,signal
import pandas as pd
import numpy as np
from collections import Counter, ChainMap
from sklearn.decomposition import PCA
import gc
from scipy.stats import percentileofscore, zscore
import umap
import os
from PIL import Image
from pathlib import Path
import glob
from dateutil import parser

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from adjustText import adjust_text
import random

'''
script to reproduce leader_follower analysis of 
visual areas, Allen institute paper:
https://www.nature.com/articles/s41586-020-03171-x
'''

T_BIN = 0.001  # time bin size in seconds (1 ms)
minFR = 0.9  # minimum firing rate for units to be included in ccg


pre_post = {'choice':[0.1,0],'stim':[0.025,0.1],
            'fback':[0.025,0.2],'block':[0.325,0],
            'action':[0.025,0.3]}  #[pre_time, post_time]
            
one = ONE() #ONE(mode='local')
ba = AllenAtlas()
br = BrainRegions()


def get_acronyms_per_eid(eid, mapping='Beryl'):

    T_BIN = 1
    
    As = {}

    dsets = one.list_datasets(eid)
    r = [x.split('/') for x in dsets if 'probe' in x]
    rr = [item for sublist in r for item in sublist
          if 'probe' in item and '.' not in item]
    probes = list(Counter(rr))         
    
    for probe in probes:
        sl = SpikeSortingLoader(eid=eid, pname=probe, one=one, atlas=ba)
        spikes, clusters, channels = sl.load_spike_sorting()
        clusters = sl.merge_clusters(spikes, clusters, channels)
        As[probe] = Counter(br.id2acronym(clusters['atlas_id'],mapping='Beryl'))    

    return As


def xs(split):
    return np.arange((pre_post[split][0] + pre_post[split][1])/T_BIN)*T_BIN


def jitter(data, l):
    """
    Jittering multidimentional logical data where 
    0 means no spikes in that time bin and 1 indicates a spike in that time bin.
    """
    if len(np.shape(data))>3:
        flag = 1
        sd = np.shape(data)
        data = np.reshape(data,(np.shape(data)[0],np.shape(data)[1],
                          len(data.flatten())/(np.shape(data)[0]*np.shape(data)[1])),
                          order='F')
    else:
        flag = 0

    psth = np.mean(data,axis=1)
    length = np.shape(data)[0]

    if np.mod(np.shape(data)[0],l):
        data[length:(length+np.mod(-np.shape(data)[0],l)),:,:] = 0
        psth[length:(length+np.mod(-np.shape(data)[0],l)),:]   = 0

    if np.shape(psth)[1]>1:
        dataj = np.squeeze(np.sum(np.reshape(data,[l,np.shape(data)[0]//l,np.shape(data)[1],np.shape(data)[2]], order='F'), axis=0))
        psthj = np.squeeze(np.sum(np.reshape(psth,[l,np.shape(psth)[0]//l,np.shape(psth)[1]], order='F'), axis=0))
    else:
        dataj = np.squeeze(np.sum(np.reshape(
                    data,l,np.shape(data)[0]//l,np.shape(data)[1], order='F')))
        psthj = np.sum(np.reshape(psth,l,np.shape(psth)[0]//l, order='F'))


    if np.shape(data)[0] == l:
        dataj = np.reshape(dataj,[1,np.shape(dataj)[0],np.shape(dataj)[1]], order='F');
        psthj = np.reshape(psthj,[1,np.shape(psthj[0])], order='F');

    psthj = np.reshape(psthj,[np.shape(psthj)[0],1,np.shape(psthj)[1]], order='F')
    psthj[psthj==0] = 10e-10

    corr = dataj/np.tile(psthj,[1, np.shape(dataj)[1], 1]);
    corr = np.reshape(corr,[1,np.shape(corr)[0],np.shape(corr)[1],np.shape(corr)[2]], order='F')
    corr = np.tile(corr,[l, 1, 1, 1])
    corr = np.reshape(corr,[np.shape(corr)[0]*np.shape(corr)[1],np.shape(corr)[2],np.shape(corr)[3]], order='F');

    psth = np.reshape(psth,[np.shape(psth)[0],1,np.shape(psth)[1]], order='F');
    output = np.tile(psth,[1, np.shape(corr)[1], 1])*corr

    output = output[:length,:,:]
    return output


def xcorrfft(a,b,NFFT):
    CCG = np.fft.fftshift(np.fft.ifft(np.multiply(np.fft.fft(a,NFFT), np.conj(np.fft.fft(b,NFFT)))))
    return CCG


def nextpow2(n):
    """get the next power of 2 that's greater than n"""
    m_f = np.log2(n)
    m_i = np.ceil(m_f)
    return 2**m_i


def get_ccgjitter(spikes, FR, jitterwindow=25):

    

    #https://github.com/jiaxx/ccg_jitter
    # spikes: neuron*ori*trial*time
    assert np.shape(spikes)[0]==len(FR)

    n_unit=np.shape(spikes)[0]
    n_t = np.shape(spikes)[-1]
    # triangle function
    t = np.arange(-(n_t-1),(n_t-1))
    theta = n_t-np.abs(t)
    del t
    NFFT = int(nextpow2(2*n_t))
    target = np.array([int(i) for i in NFFT/2+np.arange((-n_t+2),n_t)])

    ccgjitter = []
    pair=0
    for i in np.arange(n_unit-1): # V1 cell
        for m in np.arange(i+1,n_unit):  # V2 cell
            if FR[i] > minFR and FR[m] > minFR:
                temp1 = spikes[i,:,:,:] #np.squeeze(spikes[i,:,:,:])
                temp2 = spikes[m,:,:,:] #np.squeeze(spikes[m,:,:,:])
                FR1 = np.squeeze(np.mean(np.sum(temp1,axis=-1), axis=1))
                FR2 = np.squeeze(np.mean(np.sum(temp2,axis=-1), axis=1))
                tempccg = xcorrfft(temp1,temp2,NFFT)
                tempccg = np.squeeze(np.nanmean(tempccg[:,:,target],axis=1))

                temp1 = np.rollaxis(np.rollaxis(temp1,2,0), 2,1)
                temp2 = np.rollaxis(np.rollaxis(temp2,2,0), 2,1)
                ttemp1 = jitter(temp1,jitterwindow)  
                ttemp2 = jitter(temp2,jitterwindow);
                tempjitter = xcorrfft(np.rollaxis(np.rollaxis(ttemp1,2,0),
                    2,1),np.rollaxis(np.rollaxis(ttemp2,2,0), 2,1),NFFT);  
                tempjitter = np.squeeze(np.nanmean(tempjitter[:,:,target],axis=1))
                ccgjitter.append((tempccg - tempjitter).T/np.multiply(np.tile(np.sqrt(FR[i]*FR[m]), (len(target), 1)), 
                    np.tile(theta.T.reshape(len(theta),1),(1,len(FR1)))))

    ccgjitter = np.array(ccgjitter)
    return ccgjitter      


def get_psths_atlasids(split,eid, probe):

    '''
    for a given session, probe, bin neural activity
    cut into trials, reduced to PSTHs, reduced via PCA per region,
    reduce to 2 D curves describing the PCA trajectories
    '''    
       
    toolong = 2  # discard trials were feedback - stim is above that [sec]

    # Load in spikesorting
    sl = SpikeSortingLoader(eid=eid, pname=probe, one=one, atlas=ba)
    spikes, clusters, channels = sl.load_spike_sorting()
    clusters = sl.merge_clusters(spikes, clusters, channels)

    # Find spikes that are from the clusterIDs
    spike_idx = np.isin(spikes['clusters'], clusters['cluster_id'])

    # Load in trials data
    trials = one.load_object(eid, 'trials', collection='alf')
       
    # remove certain trials    
    stim_diff = trials['feedback_times'] - trials['stimOn_times']     
    rm_trials = np.bitwise_or.reduce([np.isnan(trials['stimOn_times']),
                               np.isnan(trials['choice']),
                               np.isnan(trials['feedback_times']),
                               np.isnan(trials['probabilityLeft']),
                               np.isnan(trials['firstMovement_times']),
                               np.isnan(trials['feedbackType']),
                               stim_diff > toolong])
    events = []

    if split == 'choice':
        for choice in [1,-1]:
            events.append(trials['firstMovement_times'][np.bitwise_and.reduce([
                       ~rm_trials,trials['choice'] == choice])])   

    elif split == 'action':
        for choice in [1,-1]:
            events.append(trials['firstMovement_times'][np.bitwise_and.reduce([
                       ~rm_trials,trials['choice'] == choice])])  
                       
    elif split == 'stim':    
        for side in ['Left', 'Right']:
            for contrast in [0.0, 0.0625, 0.125, 0.25, 1.0]:
                events.append(trials['stimOn_times'][np.bitwise_and.reduce([
                    ~rm_trials,trials[f'contrast{side}'] == contrast])])
       
    elif split == 'fback':    
        for fb in [1,-1]:
            events.append(trials['feedback_times'][np.bitwise_and.reduce([
                ~rm_trials,trials[f'feedbackType'] == fb])])       
              
    elif split == 'block':
        for pleft in [0.8, 0.2]:
            events.append(trials['stimOn_times'][np.bitwise_and.reduce([
                ~rm_trials,trials['probabilityLeft'] == pleft])])

    else:
        print('what is the split?', split)
        return

    # bin and cut into trials    
    bins = []
    for event in events:
        bi, _ = bin_spikes2D(spikes['times'][spike_idx], 
                           spikes['clusters'][spike_idx],
                           clusters['cluster_id'],event, 
                           pre_post[split][0], pre_post[split][1], T_BIN)
        bins.append(bi)                   
                                                     
    b = np.concatenate(bins)        
    ntr, nclus, nbins = b.shape

    acs = br.id2acronym(clusters['atlas_id'],mapping='Beryl')
    return bins, acs



def plot_trials(eid, probe):

#    eid = 'e535fb62-e245-4a48-b119-88ce62a6fe67'
#    probe = 'probe00'
#    regs = ['VISpm','VISp']

    bins, acs = get_psths_atlasids('stim',eid, probe)

    fig, axs = plt.subplots(nrows =2, ncols=5)
    axs = axs.flatten()
    k = 0
    for side in ['Left', 'Right']:
        for contrast in [0.0, 0.0625, 0.125, 0.25, 1.0]:
            # only keep neurons in the two visual areas
            spikes = bins[k][:,np.bitwise_or.reduce([acs == reg for reg in regs])]
            # average trials to visualise response
            axs[k].imshow(np.sum(spikes,axis=0), cmap='Greys',
                          interpolation=None)
            axs[k].set_title(f'{side}, {contrast}')              
            k += 1


def compute_ccg(eid, probe, regs):

#    eid = 'e535fb62-e245-4a48-b119-88ce62a6fe67'
#    probe = 'probe00'
#    regs = ['VISpm','VISp']

#   probe = 'probe01'
#   eid = '07dc4b76-5b93-4a03-82a0-b3d9cc73f412'
#   regs = ['VISp']


    bins, acs = get_psths_atlasids('stim',eid, probe)

    res = {}
    
    k = 0
    for contrast in [0.0, 0.0625, 0.125, 0.25, 1.0]:
        R = []
        for side in ['Left', 'Right']:
            
            # only keep neurons in the two visual areas
            R.append(bins[k][:,np.bitwise_or.reduce([acs == reg for reg in regs])])
            k +=1
        
        mins = min([x.shape[0] for x in R])
        # discard trials to match size to put in format for ccgjitter
        # reshape data: [neuron, trial type, n_trials, time]   
        spikes = np.array([R[0][:mins],R[1][:mins]])
        spikes = np.swapaxes(spikes,0,1)
        spikes = np.swapaxes(spikes,0,2)
        
        FR = spikes.sum(-1).mean(1).mean(1)
        res[str(contrast)] = [FR, get_ccgjitter(spikes, FR, jitterwindow=25)]
        
    plot_result(res)
    plt.suptitle(f'{eid}, {probe}, {str(regs)}')    
    return res


def plot_result(res):


    fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(14,3))
    
    k = 0 
    for c in res:
    
        # find responsive units
        rus = len(np.where(res[c][0] > minFR)[0])
        if rus > 1:
            s = 0
            for side in ['Left', 'Right']:
                x = np.concatenate([list(reversed(-xs('stim')[1:])),
                                    xs('stim')[:-1]])
                axs[k].plot(x,res[c][1][0,:,s], label = side)
                s+=1
        axs[k].axvline(x = 0, linestyle = '--', c='k')       
        axs[k].set_title(f'contrast {c}, {rus} active cells')
        axs[k].set_xlabel('lag [sec]')
        axs[k].set_ylabel('coincidence per spike')
        if k == 0:
            axs[k].legend()   
        k +=1
    fig.tight_layout()    
    
    
    
    
    









