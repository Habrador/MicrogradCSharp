using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;


namespace Micrograd
{
    //MLP = MultiLayer perceptron which is a Feedforward Neural Network
    public class MLP : Module
    {
        private readonly List<Layer> layers;

        //The layers we can add

        //Linear applies a linear transformation on the input using its stored weights and biases
        public Linear Linear(int neuronsPrev, int neuronsThis, bool useBias = true) => new(neuronsPrev, neuronsThis, useBias);

        //Layers that are activation functions
        public Sigmoid Sigmoid() => new();
        public Tanh Tanh() => new();
        public ReLU ReLU() => new();
        public Softmax Softmax() => new();



        public MLP()
        {
            layers = new();
        }



        //Add multiple layers to the NN
        public void AddLayers(params Layer[] layersToAdd)
        {
            foreach (Layer layer in layersToAdd)
            {
                layers.Add(layer);
            }
        }

        //Add a layer to the NN
        public void AddLayer(Layer layer)
        {
            layers.Add(layer);
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
                //All layers dont have weights or biases
                if (layer.Parameters() == null)
                {
                    continue;
                }

                parametersList.AddRange(layer.Parameters());
            }

            Value[] parameters = parametersList.ToArray();

            return parameters;
        }
    }
}