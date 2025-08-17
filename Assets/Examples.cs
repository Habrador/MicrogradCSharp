using Micrograd;
using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;


//Run the different examples here!
public class Examples : MonoBehaviour
{
    private void Start()
    {
        //ValueExperiments();

        NeuralNetworkExamples();
    }


    private void ValueExperiments()
    {
        //Value examples
        ValueExamples valueTest = new();

        //valueTest.TestGradients();

        //valueTest.HowDerivativesWork();

        //valueTest.BasicNeuron();
    }

    private void NeuralNetworkExamples()
    {
        //NN learning the XOR gate in as few line sof code as possible
        //NN_XOR_Minimal nn_xor_minimal = new();
    

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


        //NN learning to predict the next character in the alphabet
        NN_Alphabet alphabet = new();
    }
}
