using Micrograd;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class NeuralNetworkTrainer
{   
    public void Train(MLP nn, Value[][] input, int[] labels, int epochs, float learningRate)
    {
        int batchSize = input.Length;

        //Train
        Debug.Log("Training!");

        for (int i = 0; i <= epochs; i++)
        {
            //For each batch...
            
            //
            // Forward pass
            //

            //Catch all outputs for this batch
            Value[][] networkOutputs = new Value[batchSize][];

            for (int j = 0; j < batchSize; j++)
            {
                //Run input through the network
                Value[] outputArray = nn.Activate(input[j]);

                networkOutputs[j] = outputArray; //Do we need to clone this one?
            }


            //
            // Error calculations using MSE
            //

            Value loss = new(0f);

            for (int j = 0; j < networkOutputs.Length; j++)
            {
                for (int k = 0; k < networkOutputs[j].Length; k++)
                {
                    //One-hot-encoded
                    //We want the output to be 1 at the index of current number, otherwise 0
                    Value wantedOutput = labels[j] == k ? new Value(1f) : new Value(0f);
                    Value actualOutput = networkOutputs[j][k];

                    Value errorSquare = Value.Pow(actualOutput - wantedOutput, 2f);

                    loss += errorSquare;
                }
            }

            //Divide loss with batch size which is only needed if we have batches of different sizes?

            //Accumulate the loss if we split data into batches


            //
            // Backward pass
            //

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


            //Display
            if (i % 10 == 0)
            {
                Debug.Log($"Iteration: {i}, Network error: {loss.data}");
            }
        }
    }



    public void Test(MLP nn, Value[][] input, int[] wantedOutput)
    {
        float networkAccuracy = 0f;
    
        for (int i = 0; i < input.Length; i++)
        {
            //Run input through the network
            Value[] outputArray = nn.Activate(input[i]);

            //Find the index of the maximum value in this array (argmax)
            int maxIndex = -1;
            float maxValue = float.NegativeInfinity;
            
            for (int j = 0; j < outputArray.Length; j++)
            {
                float thisValue = outputArray[j].data;

                if (outputArray[j].data > maxValue)
                {
                    maxIndex = j;
                    maxValue = thisValue;
                }
            }

            Debug.Log($"Wanted: {wantedOutput[i]}, Actual: {maxIndex}, Value: {maxValue}");
        }

        Debug.Log($"Accuracy: {networkAccuracy}");
    }
}
