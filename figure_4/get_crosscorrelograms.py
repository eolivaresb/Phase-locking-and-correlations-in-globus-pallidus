import numpy as np
import pandas as pd
import time as clocktime
import pickle
################################################################
################################################################
## Load spike times for every cell, for every fstimulation
ncells = 16
spikes_set = []
for c in range(ncells):
    Nfreq = len(np.loadtxt('../Data/Cells/cell%d/fRates.txt'%(c+1)))
    spikes_set.append([np.loadtxt('../Data/Cells/cell%d/spk_%d.txt'%(c+1, f)) for f in range(1, Nfreq+1)])

################################################################
## Get crosscorrelations from spike trains
ttot = 10  ######  Time control for spike train crosscorrelations
twindow, tbin = 1.0, 0.001
timecc = np.arange(-twindow, twindow+tbin, tbin)
def spk_train_crosscorrelation(ttot, twindow, tbin, spk1, spk2):
    ''' Get crosscorrelation of two spike trains
    result id normalized by neurons rates '''
    r1, r2 = 1.0*len(spk1)/ttot, 1.0*len(spk2)/ttot
    bined2 = np.zeros(int(ttot/tbin) + 1)
    for s in spk2:
        bined2[int(s/tbin)] +=1
    cc = np.zeros(int(2*twindow/tbin) +1)
    for s in spk1:
        if ((s>twindow) and (s< ttot - twindow)):
            cc += bined2[int((s-twindow)/tbin): int((s-twindow)/tbin) +int(2*twindow/tbin) +1]
    cc = cc/((ttot - 2*twindow)*tbin)/(r1*r2) ## geometric mean normalization
    return cc

################################################################

datbins = 30
def pdens_crosscorrelation(spk1, spk2, freq):
    '''Get crosscorrelation from phase densities'''
    ## phase densities histograms
    h1dat = np.histogram((spk1*freq)%1, range = [0, 1], bins = datbins, density=True)[0]
    h2dat = np.histogram((spk2*freq)%1, range = [0, 1], bins = datbins, density=True)[0]
    ### Calculate ccorrelations from spike phases density
    this_crosscorrelation = np.convolve(np.concatenate([h2dat, h2dat, h2dat]), h1dat[::-1], 'valid')/np.sum(h1dat)
    return this_crosscorrelation

################################################################
def fourier_component(cc, freq):
    '''Get crosscorrelation amplitude and angle at the stimulation frequency using fft'''
    findx = int(2*freq)  ### as the CC is 2 seconds long, the fft freq step is 0.5 Hz starting from 0
    rfft = np.fft.rfft(cc)
    ypower = (np.absolute(rfft)/len(rfft))
    amp, ang = ypower[findx], np.angle(rfft[findx])
    return amp, ang

################################################################
##
def interp_spk_train_cc(cc, freq):
    '''interpolate the spike train CC on the range [-T, T] to the lenght of pdens CC'''
    if freq == 1:
        tseg, ccseg = timecc, cc
    else:
        edge = int(1./(freq*tbin))  ## this edge guaranties [-T, T] in the slice
        tseg, ccseg = timecc[1000-edge-1:1000+edge+2], cc[1000-edge-1:1000+edge+2]
    interpcc = np.interp(np.linspace((-1./freq), (1./freq), 20*(2*datbins +1)), tseg, ccseg) ## 1300 = 20* lenght(pdens_cc) = 20*(2*datbins +1)
    return np.array([np.mean(interpcc[i*20:(i+1)*20]) for i in range(2*datbins +1)])

################################################################
###
def correlation_distance(cc1, cc2):
    '''get pearson correlation between crosscorrelations'''
    return np.corrcoef(cc1, cc2)[0, 1]

################################################################
## Get phase densities from spike trains
phase_densities =  []
for i in range(ncells):
    pdens = []
    for f, spk in enumerate(spikes_set[i]):
        freq = f+1.
        pdens.append(np.histogram((spk*freq)%1, range = [0, 1], bins = datbins, density=True)[0])
    phase_densities.append(pdens)
np.save('data/phase_densities.npy', phase_densities)


cc_spk_train = {}  ## this dictionary will store spike train crosscorrelations
cc_len_pdens = {}  ## this dictionary will store both CC from pdens and spk train interpolated between [-T, T]

cc_pearson = []  ## this array of arrays will store the correlation between spk train CC and pdens CC
cc_pdens_amp = []  ## this array of arrays will store the amplitude of the spk train CC at the stim freq
cc_amp = []  ## this array of arrays will store the amplitude of the spk train CC at the stim freq
cc_ang = []  ## this array of arrays will store the angle of the spk train CC at the stim freq
cc_pairs = []  ## this array of arrays will store the neuron pairs indices to map on the other arrays

for f in range(100): # CC on the 100 frequencies of stimulation 1 -> 100 Hz
    freq = f + 1.0
    freq_cc_spk_train = {}
    freq_cc_len_pdens = {}
    freq_cc_pearson = []
    freq_cc_pdens_amp = []
    freq_cc_amp = []
    freq_cc_ang = []
    freq_cc_pairs = []
    # print(f)
    for i in range(ncells):
        if f>=len(spikes_set[i]): continue
        for j in range(i+1, ncells):
            if f>=len(spikes_set[j]): continue
            spk1 = spikes_set[i][f]
            spk2 = spikes_set[j][f]
            ### Spike train crosscorrelation
            cc_spktrain = spk_train_crosscorrelation(ttot, twindow, tbin, spk1, spk2)
            ### Spike train crosscorrelation between -T, T and lenght(pdens_cc)
            cc_spktrain_len_pdens = interp_spk_train_cc(cc_spktrain, freq)
            ## phase densities crosscorrelation
            cc_pdens = pdens_crosscorrelation(spk1, spk2, freq)
            ## spk train CC amplitude and angle at the stimulation frequency
            amp, ang = fourier_component(cc_spktrain, freq)
            ### save crosscorrelations in dictionary
            freq_cc_spk_train[(i, j)] = cc_spktrain
            pdens_cc = {}
            pdens_cc['spks'] = cc_spktrain_len_pdens
            pdens_cc['amp'] = amp
            pdens_cc['ang'] = ang
            pdens_cc['pdens'] = cc_pdens
            freq_cc_len_pdens[(i, j)] = pdens_cc
            ### add pearson correlation
            freq_cc_pearson.append(correlation_distance(cc_pdens, cc_spktrain_len_pdens))
            ### add CC amplitude and angle at the stimulation frequency
            freq_cc_pdens_amp.append(np.max(cc_pdens) - np.min(cc_pdens))
            freq_cc_amp.append(amp)
            freq_cc_ang.append(ang)
            freq_cc_pairs.append((i, j))
    ######################################
    cc_spk_train[f] = freq_cc_spk_train
    cc_len_pdens[f] = freq_cc_len_pdens
    cc_pearson.append(np.array(freq_cc_pearson))
    cc_pdens_amp.append(np.array(freq_cc_pdens_amp))
    cc_amp.append(np.array(freq_cc_amp))
    cc_ang.append(np.array(freq_cc_ang))
    cc_pairs.append(np.array(freq_cc_pairs))

#### save dictionaries to files
with open('data/spk_train_crosscorrelations.pkl', 'wb') as f:
    pickle.dump(cc_spk_train, f)

with open('data/pdens_crosscorrelations.pkl', 'wb') as f:
    pickle.dump(cc_len_pdens, f)

np.save('data/cc_pearson.npy', cc_pearson)
np.save('data/cc_pdens_amp.npy', cc_pdens_amp)
np.save('data/cc_amp.npy', cc_amp)
np.save('data/cc_ang.npy', cc_ang)
np.save('data/cc_pairs.npy', cc_pairs)
