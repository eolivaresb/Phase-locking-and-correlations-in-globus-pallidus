################################################################
### Figure 1
###############################################################
################################################################
import numpy as np
from pyhdf.SD import SD, SDC ## to load experimental hd4 files
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
import matplotlib
matplotlib.rcParams.update({'text.usetex': False, 'font.family': 'stixgeneral', 'mathtext.fontset': 'stix',})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
pi = np.pi
################################################################
#################################################################
'''
cell 6 : 09Feb2021.2.10.hdf
'''
Nfreq = 100
cells = {6 : '09Feb2021.2.10.hdf'}
dt = 0.00005
### define cell to be plotted

for cell in [6]:

    freq_to_plot = [8, 49, 80]  ## there will be three frequencies depicted in the figure
    pdens_lim = 4.25
    frlim = [36, 59]

    ### Load voltage traces
    Vset = SD('./data/%s'%(cells[cell]), SDC.READ).select('Amplitude').get()#[ind_sine] * 1000 ###(mV)
    traces_to_plot = [Vset[int(f-1)][int(0.5/dt):int(0.75/dt)] * 1000 for f in freq_to_plot] ## 250 ms of recording in mV
    time = np.arange(0, 250, dt*1000)   ## 250 ms time for x axis

    ### Load spike times
    spikes = [np.loadtxt('../Data/Cells/cell%d/spk_%d.txt'%(cell, f)) for f in range(1, Nfreq+1)]
    spk_to_plot = [spikes[int(f-1)] for f in freq_to_plot]

    ### Get phase densities to be plotted
    datbins = 30 ## Phase density for experimental data, that has around 10 s * 20 spk/s = 200 spikes per cell-freq
    xpddat = np.arange(0.5/datbins, 1, 1.0/datbins)
    def get_pdens(spk, freq):
        Sph = (spk%(1./freq))/(1./freq) # spikes phase aligned to stim freq [0, 1]
        return np.histogram(Sph, range = [0, 1], bins = datbins, density=True)[0]
    Pdensities = [get_pdens(spk_to_plot[i], freq_to_plot[i]) for i in range(3)]

    ### Load rates over frequencies of stimulation
    rates = np.loadtxt('../Data/Cells/cell%d/fRates.txt'%(cell))

    ### Get VS magnitude and angle
    magVS, angVS = np.zeros(Nfreq),np.zeros(Nfreq)
    for k, s in enumerate(spikes):
        freq = k+1.0
        nspk = len(s)
        Sph = (s%(1./freq))/(1./freq) # spikes phase aligned to stim freq [0, 1]
        xv, yv = np.sum(np.cos(2*pi*Sph)), np.sum(np.sin(2*pi*Sph))
        magVS[k] = np.sqrt(xv**2 + yv**2)/nspk
        angVS[k] = (np.arctan2(yv, xv)/(2*pi))%1
    xfreq = np.arange(1, 101)
    ###############################################################
    ###############################################################
    ### Plot layout
    ###############################################################
    fz = 16
    plt.close('all')
    fig = plt.figure(figsize = [9.5, 10.5])
    gm = gridspec.GridSpec(510, 100)

    axes_traces = [plt.subplot(gm[i*100: i*100 + 72, :58]) for i in range(3)]
    axes_stim = [ax.twinx() for ax in axes_traces]

    axes_pdens = [plt.subplot(gm[i*100: i*100 + 72, 72:]) for i in range(3)]
    a, b, c = 310, 59, 70
    axes_summary = [plt.subplot(gm[i*c + a: i*c + a + b, :]) for i in range(3)]

    ###############################################################
    ### Plot examples
    ###############################################################
    scolor = '#e41a1c'#'#1c9099'
    for k, freq in enumerate(freq_to_plot):
        findx = int(freq - 1)
        ax = axes_traces[k]
        ax.plot(time, traces_to_plot[k], 'k')
        ax.text(0.05, 1.1, 'Stimulus frequency = %d Hz'%(freq), ha='left', va='center', transform=ax.transAxes, fontsize = 12)
        ax.text(0.52, 1.1, 'Firing rate = %.1f spk/s'%(rates[findx]), ha='left', va='center', transform=ax.transAxes, fontsize = 12)
        axes_stim[k].plot(time, 20 * np.sin(2 * pi * time* freq/1000), scolor, zorder = 0) # stimulus amplitude = 20 pA
        axes_pdens[k].bar(xpddat, Pdensities[k], width=1./(datbins), color = scolor, zorder = 12)


    for ax in axes_traces + axes_stim + axes_pdens:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    for ax in axes_traces:
        ax.set_ylim(-85, 25)
        ax.set_xlim(-1, 251)
        ax.set_ylabel('Vm (mV)', fontsize = fz, labelpad = 12)
    axes_traces[-1].set_xlabel('Time (ms)', fontsize = fz)

    for ax in axes_stim:
        ax.spines['right'].set_visible(True)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_ylim(-48, 48)
        ax.set_yticks([-20, 0, 20])
        # ax.set_xticks([])
        ax.axhline(y = 0, linestyle = '--', linewidth = 0.3, color = scolor)
        ax.set_ylabel(r'$\mathrm{I_{app}}$'+' (pA)', fontsize = fz, labelpad = 17, rotation = -90, color = scolor)

    for ax in axes_pdens:
        ax.set_ylim(-0.02, pdens_lim)
        ax.set_xlim(-0.002, 1.002)
        ax.set_ylabel('PDF', fontsize = fz)
        # ax.set_yticks([-20, 0, 20])
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xticklabels(['0', '0.25', '0.5', '0.75', '1'])
    axes_pdens[-1].set_xlabel('Stimulus phase', fontsize = fz)

    ###############################################################
    ### Plot summary
    ###############################################################
    axes_summary[0].plot(xfreq, rates, '.-k')
    axes_summary[1].plot(xfreq, magVS, '.-k')
    axes_summary[2].plot(xfreq, angVS, '.-k')

    for k, ax in enumerate(axes_summary):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(0.5, 100.5)

    acolor = scolor#'b'
    for k, f in enumerate(freq_to_plot):
        dy = np.diff(axes_summary[0].get_ylim())[0]*0.03
        axes_summary[0].text(f, dy + rates[int(f-1)], r'$\downarrow$', fontsize = 4 + fz, color = acolor, ha = 'center', va = 'bottom')
        dy = np.diff(axes_summary[1].get_ylim())[0]*0.03
        axes_summary[1].text(f, dy + magVS[int(f-1)], r'$\downarrow$', fontsize = 4 + fz, color = acolor, ha = 'center', va = 'bottom')
        dy = np.diff(axes_summary[2].get_ylim())[0]*0.03
        axes_summary[2].text(f, dy + angVS[int(f-1)], r'$\downarrow$', fontsize = 4 + fz, color = acolor, ha = 'center', va = 'bottom')

    axes_summary[0].set_ylabel('Spk/s', fontsize = fz)
    axes_summary[0].set_ylim(frlim)
    axes_summary[0].plot(xfreq, xfreq, '--k', lw = 1)
    axes_summary[0].plot(2*xfreq, xfreq, '--k', lw = 1)

    axes_summary[1].set_ylabel('VS\nmagnitude', fontsize = fz)
    axes_summary[1].set_ylim(0, 1)
    axes_summary[1].set_yticks([0, 0.5, 1])

    axes_summary[2].set_xlabel('Stimulus frequency (Hz)', fontsize = fz)
    axes_summary[2].set_ylabel('VS\nAngle', fontsize = fz)
    axes_summary[2].set_ylim(0, 1)
    axes_summary[2].set_yticks([0, 0.5, 1])

    ###### Save plot
    ##################################################################
    x1, x2, y1, y2, y3, y4, fz = 0.005, 0.68, 0.98, 0.43, 0.31, 0.18, 22
    plt.figtext(x1, y1, 'A', ha = 'left', va = 'center', fontsize = fz)
    plt.figtext(x2, y1, 'B', ha = 'left', va = 'center', fontsize = fz)
    plt.figtext(x1, y2, 'C', ha = 'left', va = 'center', fontsize = fz)
    plt.figtext(x1, y3, 'D', ha = 'left', va = 'center', fontsize = fz)
    plt.figtext(x1, y4, 'E', ha = 'left', va = 'center', fontsize = fz)
    ################################################################
    fig.subplots_adjust(left = 0.09, bottom = 0.07, right = 0.99, top = 0.96)
    plt.savefig('figures/figure_1_cell_%d.png'%(cell), dpi = 300)
    plt.savefig('./figure_1.png', dpi = 300)
