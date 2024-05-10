################################################################
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
import matplotlib
matplotlib.rcParams.update({'text.usetex': False, 'font.family': 'stixgeneral', 'mathtext.fontset': 'stix',})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
################################################################
### Load data
###############################################################
'''
There are five cells that were recorded all the way up to 100 hz and have a relative stable firing rate along the recordings.
The hdf files for those recordings are located in ../Data/raw_data
Their spike times are stored in ../Data/Cells and their cell index are as follow:
cell 1 : 13Jan2021.4.4.hdf
cell 6 : 09Feb2021.2.10.hdf
cell 8 : 06Apr2021.4.5.hdf
cell 9 : 13Apr2021.2.5.hdf
cell 11 : 01Jun2021.4.12.hdf
'''
#### Noise level calculated at locking to best match distrubution spread, for every Cells
noise_levels = [0.017, 0.008, 0.01, 0.01, 0.007, 0.008, 0.015, 0.008, 0.012, 0.018, 0.015, 0.009, 0.012, 0.02, 0.015, 0.018]
ncells = 16
dt = 0.00005
#################################################################
################################################################
### Load phase densities
################################################################
Pdat = np.load('./Pdat.npy', allow_pickle = True)
Psim = np.load('./Psim.npy', allow_pickle = True)
################################################################
### get phase densities distances
################################################################
def pearson(pdf1, pdf2):
    return np.corrcoef(pdf1, pdf2)[0, 1]
##############################
corr_pearson = [[] for _ in range(100)]
for cindx in range(ncells):
    nfreq = len(Pdat[cindx])
    for f in range(nfreq):
        corr_pearson[f].append(pearson(Pdat[cindx][f], Psim[cindx][f]))
######### Getting mean and standard error on each frequency
mean_corr_pearson = np.array([np.mean(c) for c in corr_pearson])
std_corr_pearson = np.array([np.std(c)/np.sqrt(len(c)) for c in corr_pearson])
#################################################################
### Load voltage traces
for cell in [8]: #[1, 6, 8, 9, 11]:
    basal_noise = noise_levels[cindx]
    freq_to_plot = [8, 52, 80]  ## there will be three frequencies depicted in the figure
    pdens_lim = 5.65

    cindx = int(cell-1)

    ### Load return map
    nphases = 200
    p_init = np.arange(0.5/nphases, 1, 1./nphases)
    Rmaps = np.loadtxt('../figure_2/Exp_Cells_data/Rmaps/Rmap_cell_%d.txt'%(cell-1))

    ### load experimental spikes
    spikes = [np.loadtxt('../Data/Cells/cell%d/spk_%d.txt'%(cell, f)) for f in freq_to_plot]
    spikes_phase = []
    for f, freq in enumerate(freq_to_plot):
        spikes_phase.append((spikes[f]%(1./freq))/(1./freq))

    ### get simulated spike phases
    sim_spk_phases = []
    nsamples = 600
    for f, freq in enumerate(freq_to_plot):
        samples = np.zeros(nsamples)
        r = Rmaps[int(freq-1)]  ### return map calculated for this cell and this stimulation frequency
        nlevel = basal_noise * np.sqrt(freq) ### Empirical noise level dependence of stim frequency
        xinterp = np.concatenate([[0], p_init, [1]])
        yinterp = np.concatenate([[r[0]], r, [r[-1]]])
        predMap = interp1d(xinterp, yinterp, kind='nearest')
        pi = 0.5
        for adapt in range(100): ### move from initial phase to steady-state map iteration
            pi = predMap(pi)
        for k in range(nsamples):
            pi = (predMap(pi) + np.random.normal(0, nlevel)) %1.
            samples[k] = pi
        sim_spk_phases.append(samples)

    ### Get phase densities to be plotted
    datbins = 30
    xpddat = np.arange(0.5/datbins, 1, 1.0/datbins)
    ## this one (used for odens distance) are interpolated to have the same bins are Psim
    ### this ones are calculated with less bins to have a smooth histogram (no zero values)
    exampl_Pdat = [np.histogram(Sph, range = [0, 1], bins = datbins, density=True)[0] for Sph in spikes_phase]
    exampl_Psim = [Psim[cindx][int(f-1)] for f in freq_to_plot]

    ###############################################################
    ### Plot
    ################################################################
    def line_rmap(rmap):
        rmap[np.where(np.abs(np.diff(rmap))>0.5)] = np.nan
        return rmap
    ###############################################################
    plt.close('all')
    fig = plt.figure(figsize = [8, 6.5])
    gm = gridspec.GridSpec(320, 280)
    axes_rmaps = [plt.subplot(gm[:98, i*100: i*100 + 80]) for i in range(3)]
    axes_pdens = [plt.subplot(gm[130:186, i*100: i*100 + 80]) for i in range(3)]

    # ax_prc = plt.subplot(gm[232: 320, :106])
    ax_dist = plt.subplot(gm[232: 320, 20:-20])
    ### Plot examples
    ###############################################################
    fz = 13
    nbins = 150
    xpd = np.arange(0.5/nbins, 1, 1./nbins)
    dcolor = '#e41a1c'#'#1c9099'#'r'#
    scolor = 'b'

    for k, freq in enumerate(freq_to_plot):
        findx = int(freq-1)
        axes_rmaps[k].plot(p_init, line_rmap(Rmaps[findx]), color = scolor, lw = 1.2, zorder = 100, alpha = 0.95)
        axes_rmaps[k].plot(spikes_phase[k][:-1], spikes_phase[k][1:], '.', mec = dcolor, mfc = dcolor, ms = 0.8, zorder = 2)
        # axes_rmaps[k].plot(sim_spk_phases[k][:-1], sim_spk_phases[k][1:], '.', color = scolor, ms = 0.65, zorder = 1)
        axes_rmaps[k].plot([0, 1], [0, 1], '--', color = 'k',  zorder = 1, lw = 0.7)

        axes_rmaps[k].text(0.05, 1.1, 'Stimulus frequency = %d Hz'%(freq),
        ha='left', va='center', transform=axes_rmaps[k].transAxes, fontsize = 12)

        axes_pdens[k].bar(xpddat, exampl_Pdat[k], width=1./(datbins), color = dcolor, zorder = 0)
        axes_pdens[k].plot(xpd[2::5], exampl_Psim[k][2::5], '_', ms = 5, color = 'k', zorder = 1)
        axes_pdens[k].plot(xpd, exampl_Psim[k], color = 'k', lw = 1.2, zorder = 1)
    ###############################################################
    #### Plot person correlations
    xfreq = np.arange(1, 101)
    ax_dist.errorbar(xfreq, mean_corr_pearson, yerr=std_corr_pearson, fmt='-k')

    ax_dist.set_xlim(0.5, 100.5)
    ax_dist.set_ylim(-0.01, 1.01)
    ax_dist.set_yticks([0, 0.25, 0.5, 0.75, 1])

    ax_dist.set_ylabel('Pearson\ncorrelation', fontsize = fz)
    ax_dist.set_xlabel('Stimulation frequency (Hz)', fontsize = fz)
    ###############################################################
    ###############################################################
    for ax in axes_rmaps:
        ax.set_ylim(-0.01, 1.01)
        ax.set_xlim(-0.01, 1.01)
        ax.set_xlabel(r'$\phi_{prev}$', fontsize = fz+2)
        ax.set_ylabel(r'$\phi_{next}$', fontsize = fz+2)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(['0', '0.5', '1'])
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels(['0', '0.5', '1'])

    for ax in axes_pdens:
        ax.set_ylim(-0.02, pdens_lim)
        ax.set_xlim(-0.002, 1.002)
        ax.set_ylabel('PDF', fontsize = fz)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(['0', '0.5', '1'])
        ax.set_xlabel('Stimulus phase', fontsize = fz)

    ###############################################################
    for ax in axes_rmaps + axes_pdens + [ax_dist]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    ###### Save plot
    ##################################################################
    x1, x2, y1, y2, y3, fz = 0.01, 0.445, 0.97, 0.6, 0.32, 18
    plt.figtext(x1, y1, 'A', ha = 'left', va = 'center', fontsize = fz)
    plt.figtext(x1, y2, 'B', ha = 'left', va = 'center', fontsize = fz)
    plt.figtext(x1+0.05, y3, 'C', ha = 'left', va = 'center', fontsize = fz)
    # plt.figtext(x2, y3, 'D', ha = 'left', va = 'center', fontsize = fz)
    ############################################################
    ####
    fig.subplots_adjust(left = 0.09, bottom = 0.07, right = 0.98, top = 0.94)
    plt.savefig('./figure_3.png', dpi = 300)
