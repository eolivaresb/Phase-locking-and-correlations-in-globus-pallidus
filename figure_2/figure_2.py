################################################################
### Figure 2
###############################################################
################################################################
import numpy as np
import pickle
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
import matplotlib
from scipy.interpolate import interp1d
import scipy.stats as stats
from scipy.stats import pearsonr
matplotlib.rcParams.update({'text.usetex': False, 'font.family': 'stixgeneral', 'mathtext.fontset': 'stix',})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
pi = np.pi

#################################################################
####################  Toy model, triangular PRCs
#################################################################
### Triangular PRC
ncells = 21
nbins = 500
x_prc = np.arange(0.5/nbins, 1, 1.0/nbins)
thetas = np.linspace(0, 1, ncells)#np.arange(0.5/ncells, 1, 1./ncells)
################
def get_prc(theta):
    z = np.zeros(nbins)
    peak_index = int(nbins*theta)
    z[:peak_index] = x_prc[:peak_index]/theta
    z[peak_index:] = (1-x_prc[peak_index:])/(1-theta)
    return z
prcs = [get_prc(t) for t in thetas]
################
mode = 1
def get_fourier_decomposition(z):
    nbins = len(z) #number of bins to calculate PRC in data
    F = np.fft.rfft(z)
    rad, ang = np.abs(F)[:]/nbins, np.angle(F)
    # z1 = rad[0] + 2*np.sum([rad[i]*np.cos(i*2*pi*x_fcomp + ang[i]) for i in range(1, 1+ncomp)], axis = 0)
    return [rad, ang]
#################### Get PRCs
prcs = [get_prc(t) for t in thetas]
####################
### Get first angle from FFT for every prcs
zcomponents = [get_fourier_decomposition(z) for z in prcs]
alphas = np.array([z[1][mode] for z in zcomponents])
Amp = np.array([z[0][mode] for z in zcomponents])
Deltas = (alphas/(2*pi))%1

#################################################################
### Return maps calculations
#################################################################
### Time control simulation
dt = 0.000025  ## 40 kHz
ttot = 5.5
time = np.arange(0, ttot, dt)
#### Stimuli and cell propierties
w, fstim = 30, 30*mode
A = 15
#### sample array for return map
nphases = 250
p_init = np.arange(0, 1, 1./nphases)
####################
def get_rmap(z):
    ''' For a given prc it runs the phase model starting from differents initial phases in the stimuli
    it saves the phase stimuli at wich the cell fires again'''
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
####################

def line_rmap(rmap):
    rmap[np.where(np.abs(np.diff(rmap))>0.5)] = np.nan
    return rmap
####################

### Samples of the return map at locking
nsamples = 1000
def get_sample(Rmaps):
    nparalel = len(Rmaps)
    ph, samples = 0.5 * np.ones(nparalel), np.zeros((nsamples, nparalel))
    shift_eval = np.arange(nparalel)
    xinterp = np.concatenate([np.concatenate([p_init + i, [i+0.9999999999999]]) for i in range(nparalel)])
    yinterp = np.concatenate([np.concatenate([r, [r[0]]]) for r in Rmaps])
    predMap = interp1d(xinterp, yinterp, kind='nearest')
    for adapt in range(200):
        ph = predMap(ph + shift_eval)
    for k in range(nsamples):
        ph = predMap(ph + shift_eval)
        samples[k] = ph
    return(samples.T)

####################
####################
### Get return maps for every prc
Rmaps = [get_rmap(z) for z in prcs]
### Get locking phase for every prc
samples = get_sample(Rmaps)
## Experimenal locking phase (from simulations)
psis = np.mean(samples, axis = 1) ##
std_psis = np.std(samples, axis = 1) ### This should be <<<< 1 to be in look
#################################################################
#################################################################


#################################################################
### Figure layout
#################################################################
plt.close('all')
fig = plt.figure(figsize = [12,14])
gm = gridspec.GridSpec(280, 280, figure = fig)


ax20 = plt.subplot(gm[80:140, :90])
ax21 = plt.subplot(gm[80:140, 110:183])
ax22 = plt.subplot(gm[80:140, 207:280])

ax3 = plt.subplot(gm[160:200, :])

ax30 = plt.subplot(gm[220:280, :90])
ax31 = plt.subplot(gm[220:280, 110:183])
ax32 = plt.subplot(gm[220:280, 207:280])

#################################################################
#### Plot return map and observed locking
ax21.plot([0, 1], [0, 1], '--k', lw = 1)

#### Plot relationship observed locking and predicted by first mode of the PRC
ax22.plot([0, 1], [0, 1], '--k', lw = 1)
ax22.plot(Deltas, psis, '.k')

dxprc = np.concatenate([x_prc, 1+x_prc])
### Plot expected first mode given the stimuli
ax3.plot(dxprc, 0.5 + 0.5*np.cos(2*pi*mode*dxprc), ':k', alpha = 0.5, label = r'$cos(2 \pi ft)$' )
### Plot stimuli
ax3.plot(dxprc, 0.5 + 0.5*np.sin(2*pi*mode*dxprc), 'k', label = r'$sin(2 \pi ft)$')

ax3.legend(bbox_to_anchor =(0.02, 0.98, 0.2, 0.1), ncol = 2, frameon = False, fontsize = 15)

def plot_theta(pindx, color, i):
    ax21.plot(p_init, line_rmap(Rmaps[pindx]), color = color)                   ## Rmap
    ax21.plot(samples[pindx][:-1], samples[pindx][1:],  'd', color = color)     ## locking phase
    ax22.plot([Deltas[pindx]], [psis[pindx]],  'd', color = color)                ## 1st mode v/s PRC Locking phase
    ax21.plot([psis[pindx], 1], [psis[pindx], psis[pindx]],  '--', color = color)                ## 1st mode v/s PRC Locking phase
    ax22.plot([0, Deltas[pindx]], [psis[pindx], psis[pindx]],  '--', color = color)                ## 1st mode v/s PRC Locking phase

    ax20.plot(x_prc, prcs[pindx], color = color)
    ax20.plot(x_prc, 0.5 + Amp[pindx] * np.cos(2*pi*mode*(x_prc + Deltas[pindx])), '--', color = color)
    ax20.text(thetas[pindx] - 0.15, 0.98, r'$\theta_{%d}$'%(i) + ' = %.1f'%(thetas[pindx]), color = color, fontsize = 14, ha= 'center', va = 'center')
    ax21.text(1.05, psis[pindx], r'$\psi_{%d}$'%(i), color = color, fontsize = 14, ha = 'center', va = 'center')
    ### line dipicting locking phase
    if i == 1:
        for d in [0, 1]:
            ax3.axvline(x = psis[pindx]+d, color = color, linewidth = 1., linestyle = '--', alpha = 0.5)
            if d == 0: ax3.text(psis[pindx]+d, 1.02, r'$\psi_{%d}$'%(i), color = color, fontsize = 14, ha = 'center', va = 'bottom')
            if d == 1: ax3.text(psis[pindx]+d, 1.02, r'$1+\psi_{%d}$'%(i), color = color, fontsize = 14, ha = 'center', va = 'bottom')
        # plot PRC and first mode
        ax3.plot(x_prc + psis[pindx], prcs[pindx], color = color)
        ax3.plot(x_prc + psis[pindx], 0.5 + 0.5 * np.cos(2*pi*mode*(x_prc + Deltas[pindx])), '--', color = color) # *Amp[pindx]

plot_theta(18, 'r', 1)
plot_theta(10, 'b', 2)
#################################
fz = 16
for ax in [ax20, ax21, ax22, ax30, ax31, ax32]:
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'])
    ax.set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'])

for ax in [ax3, ax20, ax21, ax22, ax30, ax31, ax32]:
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

ax3.spines['left'].set_visible(False)
ax3.set_yticks([])
ax3.set_xlabel('Stimulus phase (ft)', fontsize = fz)
ax3.set_ylim(-0.01, 1.01)

for ax in [ax20, ax30]:
    ax.set_xlabel('Phase', fontsize = fz)
ax20.set_ylabel('iPRC', fontsize = fz)
ax20.set_yticks([])
ax30.set_ylim(-0.17, 2.54)
ax30.set_yticks([0, 0.5, 1, 1.5, 2, 2.5])
ax30.set_yticklabels(['0', '0.5', '1', '1.5', '2', '2.5'])
ax30.set_ylabel('iPRC (pA s)'+r'$^{-1}$', fontsize = fz)

for ax in [ax21, ax31]:
    ax.set_xlabel(r'$\phi_{prev}$', fontsize = fz+4)
    ax.set_ylabel(r'$\phi_{next}$', fontsize = fz+4)

for ax in [ax22, ax32]:
    ax.set_xlabel('PRC first mode angle  '  + r'$\Delta_{1}$', fontsize = fz)
    ax.set_ylabel('Locking phase  ' + r'$\psi$', fontsize = fz)


#################################################################
####################  Experimenal PRCs
#################################################################
ncells = 16
nbins = 10000
x_prc = np.arange(0.5/nbins, 1, 1.0/nbins)
prcs = np.loadtxt('./../Experimental_iPRCs/prcs_data.txt')
prcs_raw = np.loadtxt('./../Experimental_iPRCs/prcs_lr.txt')
prcs_raw_errors = np.loadtxt('./../Experimental_iPRCs/prcs_lr_error.txt')
################
mode = 1
#######################
### Get first angle from FFT for every prcs
zcomponents = [get_fourier_decomposition(z) for z in prcs]
Amp = np.array([z[0][mode] for z in zcomponents])
mprcs = np.mean(prcs, axis = 1)
alphas = np.array([z[1][1] for z in zcomponents])
Deltas = (alphas/(2*pi))%1


#################################################################
### Plotting a exmaple PRC to exemplify the Fourier mode get_fourier_decomposition
#################################################################
ax10 = plt.subplot(gm[:60, :90])
ax11 = [plt.subplot(gm[i*15:i*15+13, 110:173]) for i in range(4)]
ax12 = plt.subplot(gm[:60, 200:267])
ax13 = ax12.twinx()
theta_test = 0.9
z = get_prc(theta_test)
rad, ang = get_fourier_decomposition(z)
ncomp = 4

z_cumm = rad[0] * np.ones(len(x_prc))
ax10.plot(x_prc, z_cumm, '0.78')

for i in range(4):
    zcomp = 2* rad[i+1]*np.cos((i+1)*2*pi*x_prc + ang[i+1])
    z_cumm += zcomp
    ax10.plot(x_prc, z_cumm, '%.2f'%(0.74-i*0.07), lw = 1.3)
    ang_i = 1 - ((ang[i+1]/(2*pi))%1)/(i+1)
    ax11[i].plot(x_prc, zcomp, '%.2f'%(0.74-i*0.07))
    ax11[i].plot([1-1/(1+i), 1], [0, 0], '-m', lw = 1.3)
    ax11[i].plot([ang_i, 1], [0, 0], '-r', lw = 1.5)
    ax11[i].plot([ang_i, ang_i], [0, 2*rad[i+1]], 'k', lw = 1.5)

for k, ax in enumerate(ax11):
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.36, 0.36)
    ax.set_yticks([-0.2, 0, 0.2])
    ax.set_xticks([])
    ax.axhline(y = 0, color = 'k', linewidth = 0.3, linestyle = '--')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.text(0.012, 0.23, 'Mode %d'%(k+1), fontsize = fz-1)

ax11[-1].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
ax11[-1].set_xticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'])
ax11[-1].spines['bottom'].set_visible(True)
ax11[-1].spines['bottom'].set_position(('data', -0.42))
ax11[-1].set_xlabel('Phase', fontsize = fz)

ax10.plot(x_prc, z, 'k')

ax10.set_xlim(-0.01, 1.01)
ax10.set_xticks(np.arange(0, 1.01, 0.2))
ax10.set_xticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'])
ax10.spines['right'].set_visible(False)
ax10.spines['top'].set_visible(False)
ax10.set_xlabel('Phase', fontsize = fz)
ax10.set_ylabel('iPRC (pA s)'+r'$^{-1}$', fontsize = fz)

nmodes =11
ax12.plot(np.arange(1, nmodes), 2*rad[1:nmodes], 'xk')
ax13.plot(np.arange(1, nmodes), (ang[1:nmodes]/(2*pi))%1, 'dr')
ax12.set_ylim(bottom = -0.005)


ax13.set_ylim(-0.01, 1.01)

ax13.tick_params(axis='y', labelcolor='red')
ax13.set_ylabel('Fourier mode Angle', color = 'r', fontsize = fz, rotation = -90, labelpad = 17)
ax12.set_ylabel('Fourier mode Amplitude', color = 'k', fontsize = fz)
ax12.set_xlabel('Fourier mode', color = 'k', fontsize = fz)

ax12.spines['right'].set_visible(False)
ax12.spines['top'].set_visible(False)
ax13.spines['top'].set_visible(False)

#################################################################
### Load Return maps
#################################################################
### Return maps
nphases = 200
p_init = np.arange(0, 1, 1./nphases)
Rmaps = [np.loadtxt('Exp_Cells_data/Rmaps/Rmap_cell_%d.txt'%(cindx)) for cindx in range(ncells)]
#################################################################
####  Load spikes phases and calculate VS
#################################################################
####################
def get_vs(spks):
    nfreq = len(spks)
    VS, mVS = np.zeros(nfreq),np.zeros(nfreq)
    Sph_set = []
    for k, s in enumerate(spks):
        freq = k+1.0
        nspk = len(s)
        ### Vector strenght
        Sph = (s%(1./freq))/(1./freq) # spikes phase aligned to stim freq [0, 1]
        Sph_set.append(Sph)
        xv, yv = np.sum(np.cos(2*pi*Sph)), np.sum(np.sin(2*pi*Sph))
        # mVS[k] = np.mean(Sph)
        mVS[k] = (np.arctan2(yv, xv)/(2*pi))%1
        VS[k] = np.sqrt(xv**2 + yv**2)/nspk
    return VS, mVS, Sph_set
#####################
spikes_set = np.load('Exp_Cells_data/spikes_set.npy', allow_pickle = True)
rates_set = np.load('Exp_Cells_data/rates_set.npy', allow_pickle = True)
### VS and spikes phases
VS_set, mVS_set, Spikes_set = [], [], []
for c in range(ncells):
    VS, mVS, Spikes = get_vs(spikes_set[c])
    VS_set.append(VS)
    mVS_set.append(mVS)
    Spikes_set.append(Spikes)
#################################################################
### frequency closest to locking for all cells in the sample
### From visual inspection of phase distributions and rate
f_locks = [33, 38, 26, 26, 37, 47, 34, 53, 45, 20, 26, 22, 32, 27, 22, 46]
####################
####################
## Experimenal locking phase (from simulations)
psis = np.array([mVS_set[c][int(f_locks[c]-1)] for c in range(ncells)])
#################################################################
#################################################################
#### Plot return map and observed locking
ax31.plot([0, 1], [0, 1], '--k', lw = 1)

#### Plot relationship observed locking and predicted by first mode of the PRC
ax32.plot([0, 1], [0, 1], '--k', lw = 1)
ax32.plot(Deltas, psis, '.k')

dxprc = np.concatenate([x_prc, 1+x_prc])
xraw_prc = np.arange(0.5/50, 1, 1/50)

def plot_theta(pindx, color):
    flock = f_locks[pindx]
    findx = int(flock-1)
    samples = Spikes_set[pindx][findx]
    ax31.plot(p_init, line_rmap(Rmaps[pindx][findx]), color = color)                   ## Rmap
    ax31.plot(samples[:-1], samples[1:],  'd', color = color, ms = 0.1)     ## locking phase
    ax32.plot([Deltas[pindx]], [psis[pindx]],  'd', color = color)                ## 1st mode v/s PRC Locking phase
    ax31.plot([psis[pindx], 1], [psis[pindx], psis[pindx]],  '--', color = color)                ## 1st mode v/s PRC Locking phase
    ax32.plot([0, Deltas[pindx]], [psis[pindx], psis[pindx]],  '--', color = color)                ## 1st mode v/s PRC Locking phase

    ax30.plot(x_prc, prcs[pindx], color = color)
    ax30.plot(x_prc, mprcs[pindx] + Amp[pindx] * np.cos(2*pi*mode*(x_prc + Deltas[pindx])), '--', color = color)
    ax30.errorbar(xraw_prc, prcs_raw[pindx], yerr = prcs_raw_errors[pindx], color = color, ls = 'none', alpha = 0.64)
    ax30.plot(xraw_prc, prcs_raw[pindx], 'o', color = color, ls = 'none', alpha = 0.64, ms = 4)

plot_theta(10, 'r')
plot_theta(7, 'b')

a= stats.linregress(Deltas, psis)
corr_coeff, p_value = pearsonr(Deltas, psis)
ax32.text(0.1, 0.9, 'r = %.3f'%corr_coeff, fontsize = fz-3)
ax32.text(0.1, 0.8, 'p = %.1e'%p_value, fontsize = fz-3)

#################################################################
x1, x2, x3, y1, y2, y3, y4, fz = 0.01, 0.37, 0.7, 0.98, 0.73, 0.46, 0.27, 26
plt.figtext(x1, y1, 'A', ha = 'left', va = 'center', fontsize = fz)
plt.figtext(x2, y1, 'B', ha = 'left', va = 'center', fontsize = fz)
plt.figtext(0.67, y1, 'C', ha = 'left', va = 'center', fontsize = fz)
plt.figtext(x1, y2, 'D', ha = 'left', va = 'center', fontsize = fz)
plt.figtext(x2, y2, 'E', ha = 'left', va = 'center', fontsize = fz)
plt.figtext(x3, y2, 'F', ha = 'left', va = 'center', fontsize = fz)
plt.figtext(x1, y3, 'G', ha = 'left', va = 'center', fontsize = fz)
plt.figtext(x1, y4, 'H', ha = 'left', va = 'center', fontsize = fz)
plt.figtext(x2, y4, 'I', ha = 'left', va = 'center', fontsize = fz)
plt.figtext(x3, y4, 'J', ha = 'left', va = 'center', fontsize = fz)
# Add a color bar which maps values to colors
fig.subplots_adjust(left = 0.06, bottom = 0.05, right = 0.98, top = 0.97)
plt.savefig('figure_2.png', dpi = 300)
###############################################################
