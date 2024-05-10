// =============================================================
// =============================================================
#include <iostream>
#include <iomanip>  // cout precision
#include <math.h>
#include "VectorMatrix.h"
#include "NervousSystem.h"
using namespace std;

//////////////////////////////////////////////////////////
///////////////// Simulation and network configuration
// Integration Parameters
const double StepSize = 0.0001;
double ttot = 3;
// Network parameters
const int N = 1000;
const int n = 10; // numer of synapses going out and in of a neuron = 10 out of 1000
// Files for neurons and network configuration
char path_to_files[] = "./../simulation_files/";
ifstream neurons_file(std::string(path_to_files) + "neurons.dat");
ifstream Connected_file(std::string(path_to_files) + "network_small.dat");
ifstream prc_file(std::string(path_to_files) + "diff_prc.dat");
ifstream volt_file(std::string(path_to_files) + "volt.dat");
ifstream Icomp_file(std::string(path_to_files) + "I_rate_compensation.txt");
// Stimuli parameters
double freq = 10.0;  // Hz
const double Iampl = 20; // pA
const double pi = 3.14159265359;
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
int main (int argc, const char* argv[]){
/////////////////      read gmax and Erev from terminal  ////////////////////////////
    freq = atof(argv[1]);
    ttot = atof(argv[2]);
//////////////////////////////////////////////////////////
    RandomState rs;
    long randomseed = static_cast<long>(time(NULL));
    cout << "randomseed  " << randomseed << endl;
    rs.SetRandomSeed(randomseed);
    cout.precision(10);
//////////////////////////////////////////////////////////
    TVector<double> NeuronsProp(1, 2*N);
    neurons_file >> NeuronsProp;
    neurons_file.close();

    TVector<double> Connected(1, 2*n*N);
    Connected_file >> Connected;
    Connected_file.close();

    TVector<double> prc_params(1, 7*N);
    prc_file >> prc_params;
    prc_file.close();

///////////////// Load compensation currents to get similar rates as in disconnected
    TVector<double> I_compensation(1, N);
    Icomp_file >> I_compensation;

///////////////// Construct connected network
    NervousSystem gp;           // Connected GPe
    gp.SetCircuitSize(N, 3*n);

    // Load mean Voltage trace
    gp.LoadVoltage(volt_file);
    // Set conductance amplitud variability (Std normal distribution pS)
    gp.IpscStd = 1750;
    // load neurons propierties
    for (int i=1; i<=N; i++){
        gp.SetNeuronW(i,NeuronsProp[2*i-1]);
        gp.SetNeuronCV(i, NeuronsProp[2*i]);
    }
    // load network connectivity
    for (int i=1; i<=n*N; i++){
        gp.SetChemicalSynapseWeight(Connected[2*i-1], Connected[2*i], 1);
    }
    // load neurons PRCs
    for (int i=1; i<=N; i++){
        for (int p=1; p<=7; p++){
            gp.PrcNeuron[i].PrcParam[p] = prc_params[7*(i-1)+p];
        }
    }
//////////////////////////////////////////////
//////////////////////////////////////////////
    // Inicialization
    gp.RandomizeCircuitPhase(0.0, 1.0, rs);
    //////////////////////////////////////////////
    //////////////////////////////////////////////
    double Iext = 0;
    /////////////////// Adapting the network for 1 second
    gp.SetSaveSpikes(false);
    for (double t = 0; t <1.0; t += StepSize){
        Iext = Iampl*sin(2*pi*t*freq);
        for (int i=1; i<=N; i++)  {
            gp.externalinputs[i] = Iext + I_compensation[i];
        }
        gp.EulerStep(StepSize, t, rs);
    }
    /////////////////// Recording spikes
    gp.SetSaveSpikes(true);
    for (double t = 0; t <ttot; t += StepSize){
        Iext = Iampl*sin(2*pi*t*freq);
        for (int i=1; i<=N; i++)  {
            gp.externalinputs[i] = Iext + I_compensation[i];
        }
        gp.EulerStep(StepSize, t, rs);
    }
}
