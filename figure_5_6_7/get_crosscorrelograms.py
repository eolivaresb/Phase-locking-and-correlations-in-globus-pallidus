import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
import time as clocktime
################################################################
################################################################
N = 1000
frequencies = np.arange(1, 101)
ttot = 1000.
twindow, tbin = 1.0, 0.001
timecc = np.arange(-twindow, twindow+tbin, tbin)

pi = np.pi
################################################################
################################################################
def crosscorrelation(ttot, twindow, tbin, spk1, spk2):
    ''' Get crosscorrelation of two spike trains
    result id normalized by neurons rates '''
    r1, r2 = 1.0*len(spk1)/ttot, 1.0*len(spk2)/ttot
    bined2 = np.zeros(int(ttot/tbin) + 1)

    for s in spk2:
        bined2[int(s/tbin)] = 1

    cc_len = int(2*twindow/tbin) + 1
    cc = np.zeros(cc_len)

    spk1 = spk1[np.searchsorted(spk1, twindow):np.searchsorted(spk1, ttot-twindow)]
    spk1 = ((spk1-twindow)/tbin).astype(int)
    for s in spk1:
        cc += bined2[s: s + cc_len]

    cc = cc/((ttot - 2*twindow)*tbin)/(r1*r2) ## geometric mean normalization
    return cc
###############################################################
# ## Get crosscorrelation amplitude and angle at the stimulation frequency using fft
def cc_analysis(cc, freq):
    ''' Get the amplitude and phase shift for the CC at the frequency of Stimulation
    using Fourier transformation. The lenght of the cc is set to 2 seconds'''
    findx = int(2*freq)  ### as the CC is 2 seconds long, the fft freq step is 0.5 Hz starting from 0
    rfft = np.fft.rfft(cc)
    ypower = (np.absolute(rfft)/len(rfft))
    amp, ang = ypower[findx], np.angle(rfft[findx])
    return amp, np.abs(ang)/pi
###############################################################
def geom_mean(r1, r2):
    return np.sqrt(r1*r2)
################################################################
################################################################
def analysis_spikes(spikes, freq, folder):
    rates = np.array([len(s)/ttot for s in spikes])
    analysis = []
    for i in range(N):
        for j in range(i, N):
            cc = crosscorrelation(ttot, twindow, tbin, spikes[i], spikes[j])
            amp, ang = cc_analysis(cc, freq)
            analysis.append(np.array([rates[i], rates[j], geom_mean(rates[i],rates[j]), amp, ang]))
    np.save('data/analysis_%s_%d.npy'%(folder, freq), analysis)
################################################################
################################################################

################################################################
def load_spikes(freq, folder):
    return np.load('data/spikes_%s_%d.npy'%(folder, freq), allow_pickle = True)
################################################################
frequencies = [8, 35, 80]#np.arange(1, 101)
################################################################
for k, freq in enumerate(frequencies):
    if (rank==3*k + 0):
        spikes = load_spikes(freq, 'control')
        analysis_spikes(spikes, freq, 'control')
    if (rank==3*k + 1):
        spikes = load_spikes(freq, 'barrage')
        analysis_spikes(spikes, freq, 'barrage')
    if (rank==3*k + 2):
        spikes = load_spikes(freq, 'sworld')
        analysis_spikes(spikes, freq, 'sworld')
