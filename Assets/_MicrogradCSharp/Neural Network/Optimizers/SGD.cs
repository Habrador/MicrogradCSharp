using System.Collections;
using System.Collections.Generic;

namespace Micrograd
{
    //Stochastic Gradient Descent
    //Optionally with momentum which helps escape local minima and speeds up convergence
    //One can also add learning rate decay, but is generally constant (at least in PyTorch)
    public class SGD : Optimizer
    {
        private readonly float[] momentums;

        private readonly float momentum;



        //Same default values as PyTorch
        //momentum is parameters range 0 -> 1
        public SGD (Value[] parameters, float learningRate = 0.001f, float momentum = 0f) : base(parameters, learningRate)
        {
            if (momentum > 0f)
            {
                //Init with zeroes
                this.momentums = new float[parameters.Length];

                this.momentum = momentum;
            }
        }



        //Do gradient descent
        public override void Step()
        {
            for (int i = 0; i < parameters.Length; i++)
            {
                Value parameter = parameters[i];

                //SGD with momentum
                if (momentum > 0f)
                {
                    //If momentum = 0 this becomes vanilla SGD
                    float paramUpdate = momentum * momentums[i] - learningRate * parameter.grad;

                    momentums[i] = paramUpdate;

                    parameter.data += paramUpdate;
                }
                //Vanilla SGD
                else
                {
                    float paramUpdate = -learningRate * parameter.grad;

                    parameter.data += paramUpdate;
                }
            }
        }
    }
}