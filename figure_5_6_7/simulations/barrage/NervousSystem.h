// ************************************************************
// A neural network of phase model neurons 
// (based on the  Randall Beer CTRNN class)
//
// modified to run phase model neurons (EOlivares)
//  11/19 Created
// ************************************************************
#include "VectorMatrix.h"
#include "random.h"
#include <iostream>
#include <math.h>

#pragma once

// An entry in a sparse weight matrix
struct weightentry {int to; double weight; double faliure; double delay;};

struct PrcEntry {TVector<double> PrcParam;};

inline double PrcComp(double x, double e, double p){
    return pow(x,e)*(pow(p,e) - pow(x,e)) * (1+3*e+2*pow(e,2))/(e*pow(p,1+2*e));
}

inline double PrcEval(double x, TVector<double> P){
    double z = P[1]*PrcComp(x, P[4], P[7]) + P[2]*PrcComp(x, P[5], P[7]) + P[3]*PrcComp(x, P[6], P[7]);
    return z>0?z:0;
}

//inline double PrcEvaluation(Tvector)

// The NervousSystem class declaration

class NervousSystem {
    public:
        // The constructor
        NervousSystem(int size = 0, int maxchemconns = -1);
        // The destructor
        ~NervousSystem();
        
        // Accessors
        int CircuitSize(void) {return size;};
        void SetCircuitSize(int newsize, int maxchemconns);
        
        double NeuronPhase(int i) {return Phases[i];};
        void SetNeuronPhase(int i, double value) {Phases[i] = value;};
        
        double NeuronW(int i) {return W[i];};
        void SetNeuronW(int i, double value) {W[i] = value;};
        double NeuronCV(int i) {return CV[i];};
        void SetNeuronCV(int i, double value) {CV[i] = value;};
        
        double NeuronExternalInput(int i) {return externalinputs[i];};
        void SetNeuronExternalInput(int i, double value) {externalinputs[i] = value;};
        
        double ChemicalSynapseWeight(int from, int to);
        void SetChemicalSynapseWeight(int from, int to, double value);
        
        bool SaveSpikes;
        void SetSaveSpikes(bool b) {SaveSpikes = b;};
        //// Voltage file and interpolation function
        void LoadVoltage(ifstream &ifs);
        double VoltEval(double x);
                            
        // Control
        void RandomizeCircuitPhase(double lb, double ub);
        void RandomizeCircuitPhase(double lb, double ub, RandomState &rs);
        void EulerStep(double stepsize, double t, RandomState &rs);

        int size, maxchemconns;

        int PRCsize;
        TVector<PrcEntry> PrcNeuron;
        TVector<double> Sensitivity;
        
        TVector<double> Phases, Phases_pre, W, CV, externalinputs;
        
        
        double Erev, IpscRef, IpscStd; // IPSG reve potential, mean amplitude and STD (normal distr)
        TVector<double> Conductance, Conductance_pre;
        double tauG;
        
        TVector<int> NumChemicalConns;
        TMatrix<weightentry> chemicalweights;

};

