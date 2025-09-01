using Micrograd;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

//Neural Network that learns the XOR gate in as few lines of code as possible
public class NN_XOR_Minimal : MonoBehaviour
{
    private void Start()
    {
        Generate();
    }



    private void Generate()
    {
        MicroMath.Random.Seed(0);

        Value[][] inputData = Value.Convert(new[] { new[] { 0f, 0f }, new[] { 0f, 1f }, new[] { 1f, 0f }, new[] { 1f, 1f } });
        Value[] outputData = Value.Convert(new[] { 0f, 1f, 1f, 0f });

        //Create the NN
        MLP nn = new();

        //Add layer
        //2 inputs, 3 neurons in the middle layer with tanh activation function, 1 output with no activation function
        nn.AddLayers(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 1));

        //Optimizer that will do gradient descent for us
        Adam optimizer = nn.Adam_Optimizer(nn.GetParameters(), learningRate: 0.1f);

        //Train
        for (int i = 0; i <= 100; i++)
        {
            Value loss = new(0f);

            for (int j = 0; j < inputData.Length; j++)
            {
                loss += Value.Pow(nn.Activate(inputData[j])[0] - outputData[j], 2f); //MSE loss function without the M
            }

            Debug.Log($"Iteration: {i}, Network error: {loss.data}");

            optimizer.ZeroGrad();
            
            loss.Backward(); //The notorious backpropagation

            //Update weights and biases
            optimizer.Step();
            //foreach (Value param in nn.GetParameters()) //Update weights and biases
            //{
            //    param.data -= 0.1f * param.grad; //Gradient descent with 0.1 learning rate
            //}
        }

        //Test
        for (int j = 0; j < inputData.Length; j++)
        {
            Debug.Log("Wanted: " + outputData[j].data + ", Actual: " + nn.Activate(inputData[j])[0].data);
        }
    }
}
