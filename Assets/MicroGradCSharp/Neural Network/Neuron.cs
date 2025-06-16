using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;


namespace Micrograd
{
    //A single neuron in a neural network
    //Is using tanh activation function
    public class Neuron : Module
    {
        //All weights going to this neuron
        private readonly Value[] w;
        //The bias
        private readonly Value b;



        public Neuron(int number_of_inputs)
        {
            //Init weights
            w = new Value[number_of_inputs];

            //Random normal distributed weights which is common in Neural Networks
            w = w.Select(item => new Value(MicroMath.Random.Normal())).ToArray();

            //Init the bias with zero which is common in Neural Networks
            b = new(0f);
        }



        //Run input data x through the neuron
        //output = activation_function(w * x + b)
        public Value Activate(Value[] x)
        {
            //w * x is a dot product of weights and input
            //x1 * w_x1 + x2 * w_x2;
            Value wx = new(0f);

            for (int i = 0; i < x.Length; i++)
            {
                wx += (x[i] * w[i]);
            }

            Value input = wx + b;

            //Tanh activation function
            Value output = input.Tanh();
            //Value output = input.Sigmoid();
            //Value output = input.Relu();

            return output;
        }



        //Get an array with all weights and the bias
        public override Value[] Parameters()
        {
            return new List<Value> { b }.Concat(w).ToArray();
        }
    }
}