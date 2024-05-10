import numpy as np
import time as clocktime
import os
################################################################
################################################################
N = 1000
pi = np.pi
##########################################################
ttot = 100.0
################################################################
##############      Simulation     #############################
################################################################
frequencies = np.arange(1, 101)
################################################################
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
for k, freq in enumerate(frequencies):
    if (rank==k):
        os.mkdir('sworld_%d'%freq)
        os.system('cp sworld/main sworld_%d'%freq)
        os.chdir('./sworld_%d'%freq)
        os.system('./main %.1f %.1f'%(freq, ttot))
        os.system('cp Spikes_times.dat ../Spikes_times_sworld_%d.dat'%freq)
        os.system('cp Spiking_neurons.dat ../Spiking_neurons_sworld_%d.dat'%freq)
        os.chdir('../')
        os.mkdir('control_%d'%freq)
        os.system('cp control/main control_%d'%freq)
        os.chdir('./control_%d'%freq)
        os.system('./main %.1f %.1f'%(freq, ttot))
        os.system('cp Spikes_times.dat ../Spikes_times_control_%d.dat'%freq)
        os.system('cp Spiking_neurons.dat ../Spiking_neurons_control_%d.dat'%freq)
    if (rank==len(frequencies) + k):
        os.mkdir('barrage_%d'%freq)
        os.system('cp barrage/main barrage_%d'%freq)
        os.chdir('./barrage_%d'%freq)
        os.system('./main %.1f %.1f'%(freq, ttot))
        os.system('cp Spikes_times.dat ../Spikes_times_barrage_%d.dat'%freq)
        os.system('cp Spiking_neurons.dat ../Spiking_neurons_barrage_%d.dat'%freq)
