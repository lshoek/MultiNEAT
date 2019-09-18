/*
 * Main.cpp
 *
 *  Created on: Sep 20, 2012
 *      Author: peter
 */

 /*
  * Ignore this file. I use it to test stuff.
  *
  */

#include "Genome.h"
#include "Population.h"
#include "NeuralNetwork.h"
#include "Parameters.h"
#include "Substrate.h"

#include <iostream>
#include <limits>
#include <boost/asio.hpp>
#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
using namespace NEAT;

double xortest(Genome& g)
{
    double f = 0;

    NeuralNetwork net;
    g.BuildPhenotype(net);

    static const std::vector< std::vector< double > > inputs {
            {0.0,0.0,1.0},
            {0.0,1.0,1.0},
            {1.0,0.0,1.0},
            {1.0,1.0,1.0},
    };
    static const std::vector< double > outputs {
        0.0,
        1.0,
        1.0,
        0.0
    };

    for (long unsigned int i=0; i < inputs.size(); i++)
    {
        net.Input(inputs[i]);
        net.Activate();
        double test_output = outputs[i];
        double output = net.Output()[0];

        double diff = std::abs(test_output - output);
        f -= diff;
    }

    return f;
}


int main()
{
    Parameters *_params = new Parameters();
    Parameters &params = *_params;

    params.PopulationSize = 100;
    params.DynamicCompatibility = true;
    params.NormalizeGenomeSize = true;
    params.WeightDiffCoeff = 0.1;
    params.CompatTreshold = 2.0;
    params.YoungAgeTreshold = 15;
    params.SpeciesMaxStagnation = 15;
    params.OldAgeTreshold = 35;
    params.MinSpecies = 2;
    params.MaxSpecies = 10;
    params.RouletteWheelSelection = false;
    params.RecurrentProb = 0.0;
    params.OverallMutationRate = 1.0;

    params.ArchiveEnforcement = false;

    params.MutateWeightsProb = 0.05;

    params.WeightMutationMaxPower = 0.5;
    params.WeightReplacementMaxPower = 8.0;
    params.MutateWeightsSevereProb = 0.0;
    params.WeightMutationRate = 0.25;
    params.WeightReplacementRate = 0.9;

    params.MaxWeight = 8;

    params.MutateAddNeuronProb = 0.001;
    params.MutateAddLinkProb = 0.3;
    params.MutateRemLinkProb = 0.0;

    params.MinActivationA  = 4.9;
    params.MaxActivationA  = 4.9;

    params.ActivationFunction_SignedSigmoid_Prob = 0.0;
    params.ActivationFunction_UnsignedSigmoid_Prob = 1.0;
    params.ActivationFunction_Tanh_Prob = 0.0;
    params.ActivationFunction_SignedStep_Prob = 0.0;

    params.CrossoverRate = 0.0 ;
    params.MultipointCrossoverRate = 0.0;
    params.SurvivalRate = 0.2;

    params.AllowClones = true;
    params.AllowLoops = true;

    params.MutateNeuronTraitsProb = 0.0;
    params.MutateLinkTraitsProb = 0.0;

    Genome s(0, 3, 0, 1,
             false,
             UNSIGNED_SIGMOID,
             UNSIGNED_SIGMOID,
             0,
             params,
             0);

    int seed = 0; // time(nullptr)
    Population pop(s, params, true, 1.0, seed);

    for(int k=1; k<=21; k++)
    {
        double bestf = -std::numeric_limits<double>::infinity();
        for(unsigned int i=0; i < pop.m_Species.size(); i++)
        {
            for(unsigned int j=0; j < pop.m_Species[i].m_Individuals.size(); j++)
            {
                double f = xortest(pop.m_Species[i].m_Individuals[j]);
                pop.m_Species[i].m_Individuals[j].SetFitness(f);
                pop.m_Species[i].m_Individuals[j].SetEvaluated();
                
//                if (pop.m_Species[i].m_Individuals[j].HasLoops())
//                {
//                    std::cout << "loops found in individual\n";
//                }

                if (f > bestf)
                {
                    bestf = f;
                }
            }
        }

        Genome g = pop.GetBestGenome();
//        g.PrintAllTraits();

        std::cout << "Generation: " << k << ", best fitness: " << bestf << std::endl;
//        std::cout << "Species: " << pop.m_Species.size() << std::endl;
        pop.Epoch();
    }

//    double best_fitness = pop.GetBestGenome().GetFitness();
    double best_fitness = pop.GetBestFitnessEver();
    std::cout << "best fitness: " << best_fitness << std::endl;

    //TODO make these two a unit test
    assert(best_fitness > -1e-8);
    assert(best_fitness < 0);

    return 0;
}
