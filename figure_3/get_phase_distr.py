import numpy as np
from scipy.interpolate import interp1d
import time as clocktime
################################################################
## Get phase densities from experimental data
#################################################################
ncells = 16
## Spike set contain spike times, we need to map them to stimulus phase

phase_stim_set =[] ## this array will contain the corresponding phase at the stimuli for every spike

for c in range(ncells):
    ph_cell = []
    Nfreq = len(np.loadtxt('../Data/Cells/cell%d/fRates.txt'%(c+1)))
    spikes = [np.loadtxt('../Data/Cells/cell%d/spk_%d.txt'%(c+1, f)) for f in range(1, Nfreq+1)]
    for f, spks in enumerate(spikes):
        freq = f+1 ## from f index to frequency of stimulation
        ph_cell.append((spks * freq)%1) ## mapping spiketimes to period in every stimulation frequency
    phase_stim_set.append(ph_cell)


#################################################################
### Get probability density of phases from experimental data
#################################################################

datbins = 30 ## Phase density for experimental data, that's has around 10 s * 20 spk/s = 200 spikes per cell-freq
xpddat = np.arange(0.5/datbins, 1, 1.0/datbins)
## get PDF for every cell and every fstimulation
Pdat = [[np.histogram(ph, range = [0, 1], bins = datbins, density=True)[0] for ph in ph_cell] for ph_cell in phase_stim_set]

#################################################################
### resample the PDF as to have the same number of points than the simulated data
simbins = 150
xpdsim = np.arange(0.5/simbins, 1, 1.0/simbins)
Pdat_interp = [[np.interp(xpdsim, xpddat, F) for F in phase_cell] for phase_cell in Pdat]

#################################################################
######### Get probability density of phases from Return maps iterations
#################################################################
## Stimulus amplitude and rates already taked in to account in the return map evaluation
## We will load return maps and spikes calculated previously for the experimental phase locking evaluation
## Load return maps, they were calculated using 200 bins for phases

nphases = 200
p_init = np.arange(0, 1, 1./nphases)
Rmaps = [np.loadtxt('../figure_2/Exp_Cells_data/Rmaps/Rmap_cell_%d.txt'%c) for c in range(ncells)]

#### Noise level calculated at locking to best match distrubution spread, for every Cells
noise_levels = [0.017, 0.008, 0.01, 0.01, 0.007, 0.009, 0.015, 0.008, 0.012, 0.018, 0.015, 0.009, 0.012, 0.02, 0.015, 0.018]

nsamples = 4000000 # this number of samples will get a smooth PDF with simbins #bins

################################################################
### function to retrieve a sequency of phase from a return map. Noise level was estimated empirically
### to reduce calculation time, it sample in parallel all return maps (one per each fstimulation)

def get_sample(rmaps, basal_noise):
    nfreq = len(rmaps)
    xfreq = np.arange(1, nfreq + 1)
    nlevel = basal_noise * np.sqrt(xfreq) ### Empirical noise level dependence of stim frequency
    pi, samples = 0.5 * np.ones(nfreq), np.zeros((nsamples, nfreq))
    shift_eval = np.arange(nfreq)
    xinterp = np.concatenate([np.concatenate([p_init + i, [i+0.999999999]]) for i in range(nfreq)])
    yinterp = np.concatenate([np.concatenate([r, [r[0]]]) for r in rmaps])
    predMap = interp1d(xinterp, yinterp, kind='nearest')
    for adapt in range(100): ### move from initial phase to steady-state map iteration
        pi = predMap(pi + shift_eval)
    for k in range(nsamples):
        pi = (predMap(pi + shift_eval) + np.random.normal(0, nlevel)) %1.
        samples[k] = pi
    return(samples.T)
##############################
Psim = []
for cindx in range(ncells):
    print(cindx)
    samples = get_sample(Rmaps[cindx], noise_levels[cindx])
    # ## Getting probability densities for rmaps nsamples
    Psim.append([np.histogram(s, range = [0, 1], bins = simbins, density=True)[0] for s in samples])
np.save('./Pdat.npy', Pdat_interp)
np.save('./Psim.npy', Psim)
