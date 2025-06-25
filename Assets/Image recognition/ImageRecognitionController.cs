using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;
using Micrograd;

//Train a neural network to recognize handwritten digits using autograd
public class ImageRecognitionController : MonoBehaviour
{
    public GameObject displayQuadObj;



    private void Start()
    {
        //
        // Preprocess data
        //

        //Use mnist data (minimized version with 100 training and 10 test)
        //csv file, comma-separated
        //Each row consists of 785 values:
        //- the first value is the label (0 -> 9)
        //- the remaining 784 (28x28) values are the pixel values (0 -> 255)
        string[] trainingDataStringArray = ProcessData.GetRawData("mnist-100", "mnist_train_100.csv");

        Debug.Log(trainingDataStringArray[0]);
        Debug.Log(trainingDataStringArray.Length);

        ProcessData.GetData(trainingDataStringArray, out float[][] inputTrainingDataFloat, out int[] labels);

        //Display the data on a quad
        int displayIndex = 6;

        Display.DisplayDigitOnQuad(inputTrainingDataFloat[displayIndex], displayQuadObj);

        Debug.Log(labels[displayIndex]);

        Value[][] preprocessedTrainingInputData = ProcessData.PreprocessData(inputTrainingDataFloat);



        //
        // Neural Network
        //

        //Init seed so we get the same random numbers
        MicroMath.Random.Seed(0);

        //How fast/slow the network will learn
        float learningRate = 0.1f;

        //How many times to go through all data when learning
        int epochs = 10;



        //Create the NN
        //Output layer needs to have sigmoid af - linear doesnt work
        MLP nn = new(784, new int[] { 8, 10 }, new Value.AF[] { Value.AF.Relu, Value.AF.Sigmoid });

        NeuralNetworkTrainer nnTrainer = new();

        nnTrainer.Train(nn, preprocessedTrainingInputData, labels, epochs, learningRate);

        

        //Testing

        string[] testingDataStringArray = ProcessData.GetRawData("mnist-100", "mnist_train_100.csv");

        //Debug.Log(trainingDataStringArray[0]);
        //Debug.Log(trainingDataStringArray.Length);

        ProcessData.GetData(trainingDataStringArray, out float[][] testInputDataFloat, out int[] testLabels);

        Value[][] preprocessedTestInputData = ProcessData.PreprocessData(testInputDataFloat);

        nnTrainer.Test(nn, preprocessedTestInputData, testLabels);
    }
}
