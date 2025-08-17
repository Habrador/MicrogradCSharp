using System.Collections;
using System.Collections.Generic;


namespace Micrograd
{
    public class ReLU : Layer
    {
        public override Value[] Activate(Value[] x)
        {
            for (int neuron = 0; neuron < x.Length; neuron++)
            {
                Value input = x[neuron];

                x[neuron] = input.Relu();
            }

            return x;
        }
    }
}