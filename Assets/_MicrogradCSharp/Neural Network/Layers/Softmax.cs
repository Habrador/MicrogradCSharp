using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Windows;

namespace Micrograd
{
    public class Softmax 
    {
        public static Value[] Activate(Value[] x)
        {
            //We say the output from the network are the logits (log counts)
            Value[] output = x;

            //Find the largest value in the array
            //subtract it from all other values to avoid exploding values because of exp
            //The final result is the same
            float largest = 0f;

            for (int i = 0; i < output.Length; i++)
            {
                if (output[i].data > largest)
                {
                    largest = output[i].data;
                }
            }

            Value largestValue = new(largest);

            //Exponentiate all outputs
            //All values are now 0 -> ...
            for (int i = 0; i < output.Length; i++)
            {
                Value toExp = output[i] - largestValue;

                output[i] = toExp.Exp();
            }

            //Get the sum of all items (has to be value not float)
            Value sum = new(0f);

            for (int i = 0; i < output.Length; i++)
            {
                sum += output[i];
            }

            //Divide each element by the sum to the probabilities
            for (int i = 0; i < output.Length; i++)
            {
                output[i] /= sum;
            }

            return output;
        }

    }
}