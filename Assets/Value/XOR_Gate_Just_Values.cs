using Micrograd;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;


//Minimal code for the Autograd to learn the XOR gate without using the Neural Network library
public class XOR_Gate_Just_Values
{
    public XOR_Gate_Just_Values()
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
                weight.data -= learningRate * weight.grad;
            }
            foreach (Value bias in allBiases)
            {
                bias.data -= learningRate * bias.grad;
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
