using System.Collections;
using System.Collections.Generic;



namespace Micrograd
{
    //Base class for optimizer
    public class Optimizer
    {
        protected float learningRate;
        //The weights and biases
        protected Value[] parameters;



        public Optimizer(Value[] parameters, float learningRate)
        {
            this.learningRate = learningRate;
            this.parameters = parameters;
        }



        public virtual void Step() { }



        //Reset gradients
        public void ZeroGrad() 
        {
            foreach (Value parameter in parameters)
            {
                parameter.grad = 0f;
            }
        }
    }
}