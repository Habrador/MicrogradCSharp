using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace Micrograd
{
    public class Tanh : Layer
    {
        public override Value[] Activate(Value[] x)
        {
            for (int neuron = 0; neuron < x.Length; neuron++)
            {
                Value input = x[neuron];

                x[neuron] = input.Tanh();
            }

            return x;
        }
    }
}