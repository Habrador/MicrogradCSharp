using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;


namespace Micrograd
{
    //MLP = MultiLayer perceptron which is a Feedforward Neural Network
    public class MLP : Module
    {
        //How many neurons are there in a layer?
        //Do we need size array?
        private readonly List<int> size;

        private readonly List<Layer> layers;



        //Ex. 2-3-1 -> nin = 2 nouts = {3, 1}
        public MLP(int nin, int[] nouts, Value.AF[] afs) 
        { 
            //{2, 3, 1}
            size = new List<int> { nin }.Concat(nouts).ToList();

            //nouts.Length = 2
            //i = 0 -> size[0] = 2, nout = 3
            //i = 1 -> size[1] = 3, nout = 1 
            layers = nouts.Select((nout, i) => new Layer(size[i], nout, afs[i])).ToList();
        }

        //Init with how many inputs there are to rthe nn, then add layers using AddLayer()
        public MLP(int nin)
        {
            size = new() { nin };
            layers = new();
        }



        //Add a layer to the NN
        public void AddLayer(int neurons, Value.AF af)
        {
            layers.Add(new Layer(size[^1], neurons, af));

            size.Add(neurons);
        }



        //Run the entire neural network with some input x which will become output x
        public Value[] Activate(Value[] x)
        {
            foreach (Layer layer in layers)
            {
                x = layer.Activate(x);
            }

            //This is now the output from the network
            return x;
        }



        //Get an array with all weights and biases belonging to all neurons in the network
        public override Value[] Parameters()
        {
            List<Value> parametersList = new();

            foreach (Layer layer in layers)
            {
                parametersList.AddRange(layer.Parameters());
            }

            Value[] parameters = parametersList.ToArray();
            
            return parameters;
        }
    }
}