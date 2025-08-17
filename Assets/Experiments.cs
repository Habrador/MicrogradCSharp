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
        //NN learning the XOR gate in as few line sof code as possible
        NN_XOR_Minimal nn_xor_minimal = new();
    

        //NN learning the XOR by using just Values - not the NN library
        //XOR_Gate_Just_Values nn = new();


        //Various NN doing regression
        NN_Regression nn_regression = new();

        //Example from Andrej Karpathy's video
        //nn_regression.YouTube_Example();

        //XOR gate using tanh 
        //nn_regression.XOR_Gate_NN();

        //XOR gate using Relu 
        //nn_regression.XOR_Gate_NN_Relu();
    }
}
