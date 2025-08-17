using System.Collections;
using System.Collections.Generic;


namespace Micrograd
{
    //Loss function: Categorical Crossentropy (aka negative log likelihood)
    public class CategoricalCrossentropy
    {
        //networkOutputs are output from NN after Softmax
        //oneHotIndex are the indices we want the Softmax output to be 1
        public static Value Forward(Value[][] networkOutputs, int[] oneHotIndices)
        {
            Value batchLoss = new(0f);

            //For each output in the batch
            for (int i = 0; i < networkOutputs.Length; i++)
            {
                Value[] actualOutput = networkOutputs[i];

                //We are only interested in the probability where the one-hot encoded output should be 1
                //If we make sure that goes to 1 the other outputs should go to 0 because its a probability distribution
                int oneHotIndex = oneHotIndices[i];

                Value outputElement = actualOutput[oneHotIndex];

                //People are here clipping the value to 1e-7, 1 - 1e-7 because log(0) = undefined

                Value outputElementLog = outputElement.Log();

                //Sum the logs for the entire batch
                batchLoss += outputElementLog;
            }

            //Loss is the negative mean of the sum of logs
            batchLoss = -batchLoss / networkOutputs.Length;

            return batchLoss;
        }

    }
}