using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace Micrograd
{
    //The linear layer is a module that applies a linear transformation on the input using its stored weights and biases.
    //Each neuron has its own object with an array of weights and a bias
    public class Linear : Layer
    {
        public readonly Neuron[] neurons;



        public Linear(int neuronsPrev, int neuronsThis, bool useBias = true)
        {
            neurons = new Neuron[neuronsThis];
        
            neurons = neurons.Select(item => new Neuron(neuronsPrev, useBias)).ToArray();
        }



        //Activate each neuron in this layer and return their outputs
        public override Value[] Activate(Value[] x)
        {
            Value[] outputs = neurons.Select(neuron => neuron.Activate(x)).ToArray();

            return outputs;
        }



        //Get an array with all weights and biases belonging to all neurons in this layer
        public override Value[] Parameters()
        {
            return neurons.SelectMany(n => n.Parameters()).ToArray();
        }
    }
}