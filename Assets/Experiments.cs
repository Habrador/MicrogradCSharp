using Micrograd;
using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;


//Run experiments here!
public class Experiments : MonoBehaviour
{
    private void Start()
    {
        //ValueExperiments();

        NeuralNetworkExperiments();
    }


    private void ValueExperiments()
    {
        //Value examples
        ValueExperiments valueTest = new();

        //valueTest.TestGradients();

        //valueTest.HowDerivativesWork();

        //valueTest.BasicNeuron();
    }

    private void NeuralNetworkExperiments()
    {
        //Neural Network examples
        NeuralNetworksExperiments nn = new();

        //nn.XOR_Gate_Just_Values();

        //nn.XOR_Gate_NN();

        //nn.XOR_Gate_Minimal();

        nn.YouTube_Example();
    }
}
