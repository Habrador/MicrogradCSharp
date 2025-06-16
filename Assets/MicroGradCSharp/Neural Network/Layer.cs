using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace Micrograd
{
    //A single layer of neurons in a neural network
    public class Layer : Module
    {
        public readonly Neuron[] neurons;



        public Layer(int neurons_prev, int neurons_this, Value.AF af)
        {
            neurons = new Neuron[neurons_this];
        
            neurons = neurons.Select(item => new Neuron(neurons_prev, af)).ToArray();

            //UnityEngine.Debug.Log(af);
        }



        //Activate each neuron in this layer and return their outputs
        public Value[] Activate(Value[] x)
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