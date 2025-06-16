using Micrograd;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public class NeuralNetworksExperiments
{
    //Test network example from the YT video "The spelled-out intro to neural networks and backpropagation: building micrograd"
    public void YouTube_Example()
    {
        //Init seed so we get the same random numbers
        MicroMath.Random.Seed(0);

        //Training data 
        float[][] inputDataFloat = {
            new[] { 2f, 3f, -1f },
            new[] { 3f, -1f, 0.5f },
            new[] { 0.5f, 1f, 1f },
            new[] { 1f, 1f, -1f } };

        float[] outputDataFloat = new[] { 1f, -1f, -1f, 1f };

        //Convert training data from float to Value
        Value[][] inputData = Value.Convert(inputDataFloat);
        Value[] outputData = Value.Convert(outputDataFloat);

        //How fast/slow the network will learn
        float learningRate = 0.1f;

        //Create the NN
        //3 inputs, 4 neurons in two middle layers, 1 output
        MLP nn = new(3, new int[] { 4, 4, 1 }, new Value.AF[] { Value.AF.Tanh, Value.AF.Tanh, Value.AF.Tanh });

        TrainNN(nn, learningRate, inputData, outputData);
        TestNN(nn, inputData, outputData);
    }



    //Train a Neural Network to understand the XOR gate
    public void XOR_Gate_NN()
    {
        //Init seed so we get the same random numbers
        MicroMath.Random.Seed(0);

        //Training data XOR
        float[][] inputDataFloat = { new[] { 0f, 0f }, new[] { 0f, 1f }, new[] { 1f, 0f }, new[] { 1f, 1f } };
        float[] outputDataFloat = new[] { 0f, 1f, 1f, 0f };

        //Convert training data from float to Value
        Value[][] inputData = Value.Convert(inputDataFloat);
        Value[] outputData = Value.Convert(outputDataFloat);

        //How fast/slow the network will learn
        float learningRate = 0.1f;

        //Create the NN
        //2 inputs, 3 neurons in the middle layer, 1 output
        //middle layer uses tanh transfer function, last layer uses linear transfer function
        MLP nn = new(2, new int[] { 3, 1 }, new Value.AF[] { Value.AF.Tanh, Value.AF.Linear });

        TrainNN(nn, learningRate, inputData, outputData);
        TestNN(nn, inputData, outputData);
    }



    //Method for training a Neural Network
    private void TrainNN(MLP nn, float learningRate, Value[][] inputData, Value[] outputData)
    {
        //Train
        Debug.Log("Training!");

        for (int i = 0; i <= 100; i++)
        {
            //Forward pass

            //Catch all outputs for this batch
            Value[] networkOutputs = new Value[outputData.Length];

            for (int inputDataIndex = 0; inputDataIndex < outputData.Length; inputDataIndex++)
            {
                //Run input through the network
                Value[] outputArray = nn.Activate(inputData[inputDataIndex]);

                //We know we have just a single output
                Value output = outputArray[0];

                networkOutputs[inputDataIndex] = output;
            }

            //Error calculations using MSE
            Value loss = new(0f);

            for (int j = 0; j < networkOutputs.Length; j++)
            {
                Value wantedOutput = outputData[j];
                Value actualOutput = networkOutputs[j];

                Value errorSquare = Value.Pow(actualOutput - wantedOutput, 2f);

                loss += errorSquare;
            }

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
                //Learning rate decay
                //learningRate = 1f - 0.9f * i / (float)100;

                param.data += -learningRate * param.grad;
            }
        }
    }



    //Method for testing a neural network by comparing actual and wanted outputs 
    private void TestNN(MLP nn, Value[][] inputData, Value[] outputData)
    {
        //Test the network
        Debug.Log("Testing!");

        for (int inputDataIndex = 0; inputDataIndex < outputData.Length; inputDataIndex++)
        {
            Value[] outputArray = nn.Activate(inputData[inputDataIndex]);

            float wantedData = outputData[inputDataIndex].data;
            float actualData = outputArray[0].data;

            Debug.Log("Wanted: " + wantedData + ", Actual: " + actualData);
        }
    }



    //Train a Neural Network to understand the XOR gate in as few lines of code as possible
    public void XOR_Gate_Minimal()
    {
        MicroMath.Random.Seed(0);

        Value[][] inputData = Value.Convert(new [] { new[] { 0f, 0f }, new[] { 0f, 1f }, new[] { 1f, 0f }, new[] { 1f, 1f } });
        Value[] outputData = Value.Convert(new[] { 0f, 1f, 1f, 0f });

        MLP nn = new(2, new int[] { 3, 1 }, new Value.AF[] { Value.AF.Tanh, Value.AF.Linear }); //2 inputs, 3 neurons in the middle layer, 1 output, tanh transfer function

        //Train
        for (int i = 0; i <= 100; i++)
        {
            Value loss = new(0f);

            for (int j = 0; j < inputData.Length; j++) 
            {
                loss += Value.Pow(nn.Activate(inputData[j])[0] - outputData[j], 2f); //MSE loss function
            }

            Debug.Log($"Iteration: {i}, Network error: {loss.data}");

            nn.ZeroGrad();
            loss.Backward(); //The notorious backpropagation

            foreach (Value param in nn.GetParameters())
            {
                param.data += -0.1f * param.grad; //Gradient descent with 0.1 learning rate
            }
        }

        //Test
        for (int j = 0; j < inputData.Length; j++)
        {
            Debug.Log("Wanted: " + outputData[j].data + ", Actual: " + nn.Activate(inputData[j])[0].data);
        }
    }



    //Minimal code for the Autograd to learn the XOR gate
    public void XOR_Gate_Just_Values()
    {
        //Init seed so we get the same random numbers
        MicroMath.Random.Seed(0);

        //2 inputs, 1 output, 3 neurons in the middle layer

        //Training data XOR
        float[][] inputDataFloat = { new[] { 0f, 0f }, new[] { 0f, 1f }, new[] { 1f, 0f }, new[] { 1f, 1f } };
        float[] outputDataFloat = new[] { 0f, 1f, 1f, 0f };

        Value[][] inputData = Value.Convert(inputDataFloat);
        Value[] outputData = Value.Convert(outputDataFloat);

        //Weights
        //Init the weights with mean 0 and standardDeviation 1

        //Weights input -> middle layer
        Value w_x1_m1 = new(MicroMath.Random.Normal());
        Value w_x1_m2 = new(MicroMath.Random.Normal());
        Value w_x1_m3 = new(MicroMath.Random.Normal());

        Value w_x2_m1 = new(MicroMath.Random.Normal());
        Value w_x2_m2 = new(MicroMath.Random.Normal());
        Value w_x2_m3 = new(MicroMath.Random.Normal());

        //Weights middle -> output
        Value w_m1_o = new(MicroMath.Random.Normal());
        Value w_m2_o = new(MicroMath.Random.Normal());
        Value w_m3_o = new(MicroMath.Random.Normal());

        //Biases
        Value b_m1 = new(0f);
        Value b_m2 = new(0f);
        Value b_m3 = new(0f);

        Value b_o = new(0f);

        //Put the weights in an array to easier do operations later
        Value[] allWeights = {
            w_x1_m1, w_x1_m2, w_x1_m3,
            w_x2_m1, w_x2_m2, w_x2_m3,
            w_m1_o, w_m2_o, w_m3_o };

        Value[] allBiases = { b_m1, b_m2, b_m3, b_o };



        //Training
        Debug.Log("Training");

        for (int i = 0; i < 101; i++)
        {
            //Train the batch and cache the output from each sample
            Value[] networkOutputs = new Value[outputData.Length];

            for (int trainingDataIndex = 0; trainingDataIndex < outputData.Length; trainingDataIndex++)
            {
                //Input to the network
                Value x1 = inputData[trainingDataIndex][0];
                Value x2 = inputData[trainingDataIndex][1];

                //Forward pass

                //Input to neurons in middle layer
                Value m1_input = x1 * w_x1_m1 + x2 * w_x2_m1;
                Value m2_input = x1 * w_x1_m2 + x2 * w_x2_m2;
                Value m3_input = x1 * w_x1_m3 + x2 * w_x2_m3;

                m1_input += b_m1;
                m2_input += b_m2;
                m2_input += b_m2;

                //Middle layer activation function tanh
                Value m1_output = m1_input.Tanh();
                Value m2_output = m2_input.Tanh();
                Value m3_output = m3_input.Tanh();

                //Input to neurons in output layer
                Value o_input = m1_output * w_m1_o + m2_output * w_m2_o + m3_output * w_m3_o;

                o_input += b_o;

                //Output layer has linear activation function -> input = output
                networkOutputs[trainingDataIndex] = o_input;
            }

            //Network error calculations

            //Use MSE loss function: SUM((wanted - actual)^2)
            Value loss = new(0f);

            for (int trainingDataIndex = 0; trainingDataIndex < outputData.Length; trainingDataIndex++)
            {
                Value wantedOutput = outputData[trainingDataIndex];
                Value actualOutput = networkOutputs[trainingDataIndex];

                Value error = actualOutput - wantedOutput;

                Value errorSquare = Value.Pow(error, 2f);

                loss += errorSquare;
            }

            if (i % 10 == 0)
            {
                Debug.Log($"Iteration: {i}, Network error: {loss.data}");
            }

            //Backward pass

            //ZERO GRAD (remember we do A.grad += in Value class) so they will accumulate 4ever
            foreach (Value weight in allWeights)
            {
                weight.grad = 0f;
            }
            foreach (Value bias in allBiases)
            {
                bias.grad = 0f;
            }

            //Calculate the gradients
            loss.Backward();

            //Debug.Log(w_x1_m1.grad);

            //Optimize the weights by using gradient descent
            float learningRate = 0.01f;

            foreach (Value weight in allWeights)
            {
                weight.data += -learningRate * weight.grad;
            }
            foreach (Value bias in allBiases)
            {
                bias.data += -learningRate * bias.grad;
            }
        }



        //Test the network
        Debug.Log("Testing");

        for (int trainingDataIndex = 0; trainingDataIndex < outputData.Length; trainingDataIndex++)
        {
            //Input to the network
            Value x1 = inputData[trainingDataIndex][0];
            Value x2 = inputData[trainingDataIndex][1];

            //Input to neurons in middle layer
            Value m1_input = x1 * w_x1_m1 + x2 * w_x2_m1;
            Value m2_input = x1 * w_x1_m2 + x2 * w_x2_m2;
            Value m3_input = x1 * w_x1_m3 + x2 * w_x2_m3;

            m1_input += b_m1;
            m2_input += b_m2;
            m2_input += b_m2;

            //Middle layer activation function tanh
            Value m1_output = m1_input.Tanh();
            Value m2_output = m2_input.Tanh();
            Value m3_output = m3_input.Tanh();

            //Input to neurons in output layer
            Value o_input = m1_output * w_m1_o + m2_output * w_m2_o + m3_output * w_m3_o;

            o_input += b_o;

            //Output layer has linear activation function -> input = output

            float wantedData = outputData[trainingDataIndex].data;
            float actualData = o_input.data;

            Debug.Log("Wanted: " + wantedData + ", Actual: " + actualData);
        }
    }

}
