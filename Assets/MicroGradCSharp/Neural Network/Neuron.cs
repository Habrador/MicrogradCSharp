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
        //Which activation function to use
        private readonly Value.AF af;



        public Neuron(int number_of_inputs, Value.AF af)
        {
            //Init weights
            w = new Value[number_of_inputs];

            //Random normal distributed weights which is common in Neural Networks
            w = w.Select(item => new Value(MicroMath.Random.Normal())).ToArray();

            //Init the bias with zero which is common in Neural Networks
            b = new(0f);

            this.af = af;
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

            //return input.Tanh();

            //output = activation_function(input)
            if (af == Value.AF.Tanh)
            {
                return input.Tanh();
            }
            else if (af == Value.AF.Relu)
            {
                return input.Relu();
            }
            else if (af == Value.AF.Sigmoid)
            {
                return input.Sigmoid();
            }
            else
            {
                return input;
            }
        }



        //Get an array with all weights and the bias
        public override Value[] Parameters()
        {
            return new List<Value> { b }.Concat(w).ToArray();
        }
    }
}