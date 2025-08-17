using System.Collections;
using System.Collections.Generic;


namespace Micrograd
{
    //Loss function Mean Squared Error (MSE)
    public class MeanSquaredError
    {
        public static Value Forward(Value[] networkOutputs, Value[] wantedOutputs)
        {
            Value loss = new(0f);

            for (int j = 0; j < networkOutputs.Length; j++)
            {
                Value wantedOutput = wantedOutputs[j];
                Value actualOutput = networkOutputs[j];

                Value errorSquare = Value.Pow(actualOutput - wantedOutput, 2f);

                loss += errorSquare;
            }

            return loss;
        }

        public static Value Forward(Value[][] networkOutputs, Value[][] wantedOutputs)
        {
            Value loss = new(0f);

            for (int j = 0; j < networkOutputs.Length; j++)
            {
                Value[] wantedOutput = wantedOutputs[j];
                Value[] actualOutput = networkOutputs[j];

                loss += Forward(wantedOutput, actualOutput);
            }

            return loss;
        }
    }
}