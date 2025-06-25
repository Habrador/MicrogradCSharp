using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;

namespace Micrograd
{
    //Parent class to easier get parameters and reset gradients
    public class Module
    {
        private Value[] parameters;
        


        //Reset all gradients to zero
        public void ZeroGrad()
        {
            if (parameters == null)
            {
                parameters = Parameters();
            }

            foreach (Value param in parameters)
            {
                param.grad = 0f;
            }
        }



        public Value[] GetParameters()
        {
            if (parameters == null)
            {
                parameters = Parameters();
            }

            return parameters;
        }



        public virtual Value[] Parameters()
        {
            UnityEngine.Debug.Log("Got parameters from Module");
        
            return null;
        }

    }
}