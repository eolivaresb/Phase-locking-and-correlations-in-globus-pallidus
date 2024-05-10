import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
pi = np.pi

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
#################################################################
def get_rmap(z, fstim, w):
    ''' For a given prc it runs the phase model starting from differents initial phases in the stimuli
    it saves the phase stimuli at wich the cell firea again'''
    ph = np.zeros((nphases))
    rmap = np.zeros((nphases))
    check_finish = np.ones(nphases)
    for t in time: ## simulate for 1 second, should finish before, but there's no a priori way to know when
        zindx = ((ph*nbins)%nbins).astype(int) ## indeces for z(phi) in every simulation
        ######### Euler integration step
        dph = dt * (w + z[zindx] * A * np.sin(2*pi*(fstim * t + p_init)))
        ph += dph
        ##### Get indices for simulations done
        finished = np.where(ph>=1)
        ##### Save the phase of the stimuli for the done simulations
        rmap[finished] += check_finish[finished]*((fstim * t + p_init[finished])%1)
        #### Check finished set to zeros so no modification further
        check_finish[finished] = 0
        ### akk  all initial phases are finished criteria
        if np.sum(check_finish) == 0:
            break
    return rmap

#################################################################
#################################################################
dt = 0.000025  ## 40 kHz
ttot = 500
time = np.arange(0, ttot, dt)
#################################################################
nphases = 200
p_init = np.arange(0, 1, 1./nphases)
#################################################################
A = 20
ncells = 16
nbins = 10000
xprc = np.linspace(0, 1, nbins)
prcs = np.loadtxt('../../Experimental_iPRCs/prcs_data.txt')
rates_set = [np.loadtxt('../../Data/Cells/cell%d/fRates.txt'%(c+1)) for c in range(ncells)]
#################################################################
#################################################################

for cindx in range(ncells):
    if cindx == rank:
        rates = rates_set[cindx]
        nfreq = len(rates)
        rmap = np.zeros((nfreq, nphases))
        z = prcs[cindx]
        for f in range(nfreq):
            freq = f + 1.0
            w = rates[f]
            rmap[f] = get_rmap(z, freq, w)
        np.savetxt('Rmaps/Rmap_cell_%d.txt'%(cindx), rmap, fmt = '%.6f')

#################################################################
if rank == 0:
    ncells = 16
    spikes_set = []
    rates_set = []
    for c in range(ncells):
        cell = c + 1
        rates_set.append(np.loadtxt('../../Data/Cells/cell%d/fRates.txt'%(cell)))
        nfreq = len(rates_set[-1])
        spikes_set.append([np.array(pd.read_csv('../../Data/Cells/cell%d/spk_%d.txt'%(cell, k+1), sep='\s+', header=None))for k in range(nfreq)])
    np.save('spikes_set.npy', spikes_set)
    np.save('rates_set.npy', rates_set)
