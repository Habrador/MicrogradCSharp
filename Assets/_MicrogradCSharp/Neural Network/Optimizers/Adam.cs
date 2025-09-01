using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;



namespace Micrograd
{
    //Adam (Adaptive Momentum) optimizer
    //Commonly used as alternative to SGD with momentum
    //Uses a counter t so use it only for batch or entire episodes or move the counter to its own method and update somewhere else
    public class Adam : Optimizer
    {
        //betas are coefficients used for computing running averages
        private readonly float beta1;
        private readonly float beta2;
        //epsilon is a term added to the denominator to improve numerical stability (avoid zero division)
        private readonly float epsilon;

        //First moment vector
        private readonly float[] m;
        //Second moment vector
        private readonly float[] v;
        //Time step
        private int t = 1;



        //Same default parameters as PyTorch
        public Adam(
            Value[] parameters, 
            float learningRate = 0.001f, 
            float beta1 = 0.9f, 
            float beta2 = 0.999f,  
            float epsilon = (float)1e-8) : base (parameters, learningRate)
        {
            this.beta1 = beta1;
            this.beta2 = beta2;

            this.epsilon = epsilon;

            //Init with 0s
            this.m = new float[parameters.Length];
            this.v = new float[parameters.Length];
        }



        //Do gradient descent
        public override void Step()
        {
            for (int i = 0; i < parameters.Length; i++)
            {
                float gradient = parameters[i].grad;

                //Update biased first moment estimate
                m[i] = beta1 * m[i] + (1f - beta1) * gradient;
                
                //Update biased second moment estimate
                v[i] = beta2 * v[i] + (1f - beta2) * (gradient * gradient);

                //Bias correction
                float m_hat = m[i] / (1f - MicroMath.Pow(beta1, t));

                float v_hat = v[i] / (1f - MicroMath.Pow(beta2, t));

                parameters[i].data += -learningRate * m_hat / (MicroMath.Sqrt(v_hat) + epsilon); 
            }

            //Some are updating this after each batch or after each episode
            t += 1; 
        }
    }
}