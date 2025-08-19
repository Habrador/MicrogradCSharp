using Micrograd;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

//Neural Networks doing regression
public class NN_Regression : MonoBehaviour
{
    private void Start()
    {
        //YouTube_Example();

        //XOR_Gate_NN();

        XOR_Gate_NN_Relu();
    }



    //Test network example from the YT video "The spelled-out intro to neural networks and backpropagation: building micrograd"
    public void YouTube_Example()
    {
        //Init seed so we get the same random numbers
        MicroMath.Random.Seed(1);

        //Training data 
        float[][] inputDataFloat = {
            new[] { 2f, 3f, -1f },
            new[] { 3f, -1f, 0.5f },
            new[] { 0.5f, 1f, 1f },
            new[] { 1f, 1f, -1f } };

        float[][] outputDataFloat = new[] { new[] { 1f }, new[] { -1f }, new[] { -1f }, new[] { 1f } };

        //Convert training data from float to Value
        Value[][] inputData = Value.Convert(inputDataFloat);
        Value[][] outputData = Value.Convert(outputDataFloat);

        //How fast/slow the network will learn
        float learningRate = 0.1f;
        //How many times to go through all data when learning
        int epochs = 100;

        //Create the NN
        //3 inputs, 4 neurons in two middle layers, 1 output
        MLP nn = new();

        nn.AddLayer(nn.Linear(3, 4));
        nn.AddLayer(nn.Tanh());
        nn.AddLayer(nn.Linear(4, 4));
        nn.AddLayer(nn.Tanh());
        nn.AddLayer(nn.Linear(4, 1));
        nn.AddLayer(nn.Tanh());

        TrainNN(nn, learningRate, epochs, inputData, outputData);
        TestNN(nn, inputData, outputData);
    }



    //Train a Neural Network to understand the XOR gate using tanh
    public void XOR_Gate_NN()
    {
        //Init seed so we get the same random numbers
        MicroMath.Random.Seed(0);

        //Training data XOR
        float[][] inputDataFloat = { new[] { 0f, 0f }, new[] { 0f, 1f }, new[] { 1f, 0f }, new[] { 1f, 1f } };
        float[][] outputDataFloat = new[] { new[] { 0f }, new[] { 1f }, new[] { 1f }, new[] { 0f } };

        //Convert training data from float to Value
        Value[][] inputData = Value.Convert(inputDataFloat);
        Value[][] outputData = Value.Convert(outputDataFloat);

        //How fast/slow the network will learn
        float learningRate = 0.1f;
        //How many times to go through all data when learning
        int epochs = 100;

        //Create the NN
        MLP nn = new();

        //Add layers
        //2 inputs, 3 neurons in the middle layer, 1 output
        //middle layer uses tanh transfer function, last layer uses no transfer function
        nn.AddLayer(nn.Linear(2, 3, useBias: true));
        nn.AddLayer(nn.Tanh());
        nn.AddLayer(nn.Linear(3, 1, useBias: true));

        TrainNN(nn, learningRate, epochs, inputData, outputData);
        TestNN(nn, inputData, outputData);
    }



    //Train a Neural Network to understand the XOR gate using Relu
    public void XOR_Gate_NN_Relu()
    {
        //Init seed so we get the same random numbers
        MicroMath.Random.Seed(1);

        //Training data XOR
        float[][] inputDataFloat = { new[] { 0f, 0f }, new[] { 0f, 1f }, new[] { 1f, 0f }, new[] { 1f, 1f } };
        float[][] outputDataFloat = new[] { new[] { 0f }, new[] { 1f }, new[] { 1f }, new[] { 0f } };

        //Convert training data from float to Value
        Value[][] inputData = Value.Convert(inputDataFloat);
        Value[][] outputData = Value.Convert(outputDataFloat);

        //How fast/slow the network will learn
        float learningRate = 0.1f;

        //How many times to go through all data when learning
        int epochs = 500;

        //Create the NN
        MLP nn = new();

        //Add layers
        nn.AddLayer(nn.Linear(2, 8));
        nn.AddLayer(nn.ReLU());
        nn.AddLayer(nn.Linear(8, 1));
        //Output layer needs to have sigmoid af -linear doesnt work
        nn.AddLayer(nn.Sigmoid());

        TrainNN(nn, learningRate, epochs, inputData, outputData);
        TestNN(nn, inputData, outputData);
    }



    //Method for training a Neural Network
    private void TrainNN(MLP nn, float learningRate, int epochs, Value[][] inputData, Value[][] outputData)
    {
        //Train
        Debug.Log("Training!");

        for (int i = 0; i <= epochs; i++)
        {
            //Forward pass

            //Catch all outputs for this batch
            Value[][] networkOutputs = new Value[outputData.Length][];

            for (int inputDataIndex = 0; inputDataIndex < outputData.Length; inputDataIndex++)
            {
                //Run input through the network
                Value[] outputArray = nn.Activate(inputData[inputDataIndex]);

                networkOutputs[inputDataIndex] = outputArray;
            }

            //Error calculations using MSE
            Value loss = nn.MSE(networkOutputs, outputData);

            //Divide loss with batch size which is only needed if we have batches of different sizes?

            if (i % 10 == 0)
            {
                Debug.Log($"Iteration: {i}, Network error: {loss.data}");
            }

            //Backward pass
            //ZERO GRAD (remember we do A.grad += in Value class) so they will accumulate 4ever if we dont reset
            nn.ZeroGrad();

            //Calculate the gradients
            loss.Backward();

            //Optimize the weights and biases by using gradient descent
            Value[] parameters = nn.GetParameters();

            foreach (Value param in parameters)
            {
                param.data -= learningRate * param.grad;
            }
        }
    }



    //Method for testing a neural network by comparing actual and wanted outputs 
    private void TestNN(MLP nn, Value[][] inputData, Value[][] outputData)
    {
        //Test the network
        Debug.Log("Testing!");

        for (int inputDataIndex = 0; inputDataIndex < outputData.Length; inputDataIndex++)
        {
            Value[] outputArray = nn.Activate(inputData[inputDataIndex]);

            float wantedData = outputData[inputDataIndex][0].data;
            float actualData = outputArray[0].data;

            Debug.Log("Wanted: " + wantedData + ", Actual: " + actualData);
        }
    }

}
