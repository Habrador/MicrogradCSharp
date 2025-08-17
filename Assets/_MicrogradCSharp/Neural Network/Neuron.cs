using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;


namespace Micrograd
{
    //A single neuron in a neural network
    public class Neuron : Module
    {
        //All weights going to this neuron
        private readonly Value[] w;
        //The bias
        private readonly Value b;
        //If we should use the bias?
        private readonly bool useBias;



        public Neuron(int number_of_inputs, bool useBias = true)
        {
            //Init weights
            w = new Value[number_of_inputs];

            //Random normal distributed weights which is common in Neural Networks
            w = w.Select(item => new Value(MicroMath.Random.Normal())).ToArray();

            if (useBias)
            {
                //Init the bias with zero which is common in Neural Networks
                b = new(0f);
            }

            this.useBias = useBias;
        }



        //Run input data x through the neuron
        public Value Activate(Value[] x)
        {
            //w * x is a dot product of weights and input
            //x1 * w_x1 + x2 * w_x2;
            Value wx = new(0f);

            for (int i = 0; i < x.Length; i++)
            {
                wx += (x[i] * w[i]);
            }

            if (useBias)
            {
                return wx + b;
            }
            else
            {
                return wx;
            }
        }



        //Get an array with all weights and the bias
        public override Value[] Parameters()
        {
            if (useBias)
            {
                return new List<Value> { b }.Concat(w).ToArray();
            }
            else
            {
                return w;
            }
        }
    }
}