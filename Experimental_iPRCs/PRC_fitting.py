import numpy as np
import matplotlib.pylab as plt
from scipy.optimize import curve_fit
import matplotlib.gridspec as gridspec
import os
import glob
##################################################################
import matplotlib
matplotlib.rcParams.update({'text.usetex': False, 'font.family': 'stixgeneral', 'mathtext.fontset': 'stix',})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
##################################################################
def func(x, c1, c2, c3, e1, e2, e3, p_thre):
    comp1 = (x**e1*(p_thre**e1 - x**e1))/(e1*p_thre**(1+2*e1)/(1 + 3*e1+2*e1**2))
    comp2 = (x**e2*(p_thre**e2 - x**e2))/(e2*p_thre**(1+2*e2)/(1 + 3*e2+2*e2**2))
    comp3 = (x**e3*(p_thre**e3 - x**e3))/(e3*p_thre**(1+2*e3)/(1 + 3*e3+2*e3**2))
    y = c1*comp1 + c2*comp2 + c3*comp3
    y[np.where(y<0)] = 0.0
    return y
##################################################################
##################################################################

prcs = np.loadtxt('prcs_ea_noise.txt')
nbins = 250
xprc = np.arange(0.5/nbins, 1, 1./nbins)
##################################################################
##################################################################
##################################################################
ncells = 16
nphases = 10000
xphase = np.arange(0.5/nphases, 1, 1./nphases)
params_file = np.zeros((ncells, 7))
prcs_fitted = np.zeros((ncells, nphases))

for k in range(ncells):
    #Fit for the parameters a, b, c of the function func:
    popt, pcov = curve_fit(func, xprc, prcs[k], p0 = [1, 1, 1, 1, 10, 30, 1], bounds=(0.0001, [50., 50., 50., 50., 50., 50., 1.0]) )
    params_file[k] = popt
    prcs_fitted[k] = func(xphase, *popt)


for k in range(ncells):
    plt.close('all')
    fig = plt.figure(figsize = [8,4])
    gm = gridspec.GridSpec(20, 24)
    ax = plt.subplot(gm[1:18, 0:22])
    ax.plot(xprc, prcs[k], '.r')
    ax.axhline(y=0, linestyle = '--', linewidth = 1.6)
    ax.plot(xphase, prcs_fitted[k], '-b', lw = 1.6)

    ax.set_xlim(-0.01, 1.01)
    fig.subplots_adjust(left = 0.1, bottom = 0.05, right = 0.95, top = 0.90)
    ax.set_xlabel('Phase')
    ax.set_ylabel('PRC  cycles/(mA*s)')

    ax.set_ylim(bottom = -0.2)
    plt.savefig('./prc_figures/prc_%s.png'%(k+1), dpi = 250)

np.savetxt('./prcs_data.txt', np.array(prcs_fitted))
np.savetxt('./params_prc.txt', params_file.T)
