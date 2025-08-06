using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace Micrograd
{
    //Value class for autograd that stores a single scalar and its gradient
    //which automates backpropagation 
    //PyTorch is using something similar
    public class Value
    {
        public float data;
        //Derivative wrt output
        public float grad;
        //To easier recognize the value, if we name is b1 or something for bias 1
        public string label;

        //The two Values that generated this node, such as thisValue = childA + childB
        //Can be null if one wasn't used
        public Value childA;
        public Value childB;

        //The method doing the gradient calculations
        public delegate void BackwardDelegate();

        public BackwardDelegate backwardOneStep;



        public Value(float value, Value childA = null, Value childB = null, string label = "")
        {
            this.data = value;
            this.grad = 0f;

            this.childA = childA;
            this.childB = childB;

            this.label = label;

            //Anonymous method using lambda
            this.backwardOneStep = () =>
            {
                //Debug.Log("You've reached a leaf node, backward not possible!");
            };
        }



        //
        // Automatic Backward pass using topological sort
        //

        public void Backward()
        {
            List<Value> topo = new();
            HashSet<Value> visited = new();

            BuildTopo(this, topo, visited);

            //Reverse the topo list so we start at the end
            topo.Reverse();

            this.grad = 1f;

            foreach (Value node in topo)
            {
                node.backwardOneStep();
            }

            //Debug.Log("Added " + topo.Count + " nodes");
        }

        //Topological sort
        private void BuildTopo(Value v, List<Value> topo, HashSet<Value> visited)
        {
            if (v == null)
            {
                return;
            }

            if (!visited.Contains(v))
            {
                visited.Add(v);

                //Add children before you add yourself 
                BuildTopo(v.childA, topo, visited);
                BuildTopo(v.childB, topo, visited);

                topo.Add(v);
            }
        }



        //
        // Addition
        //

        public static Value AutogradAdd(Value A, Value B)
        {
            Value output = new(A.data + B.data, A, B);

            output.backwardOneStep = () =>
            {
                //Accumulate the gradient if we are using the same Value object multiple times in the network
                A.grad += 1f * output.grad;
                B.grad += 1f * output.grad;
            };

            return output;
        }

        //A + B
        public static Value operator +(Value A, Value B) => AutogradAdd(A, B);

        //A + b
        public static Value operator +(Value A, float b) => AutogradAdd(A, new(b));

        //a + B
        public static Value operator +(float a, Value B) => AutogradAdd(new(a), B);



        //
        // Multiplication
        //

        public static Value AutogradMultiply(Value A, Value B)
        {
            Value output = new(A.data * B.data, A, B);

            output.backwardOneStep = () =>
            {
                A.grad += B.data * output.grad;
                B.grad += A.data * output.grad;
            };

            return output;
        }

        //A * B
        public static Value operator *(Value A, Value B) => AutogradMultiply(A, B);

        //A * b
        public static Value operator *(Value A, float b) => AutogradMultiply(A, new(b));

        //a * B
        public static Value operator *(float a, Value B) => AutogradMultiply(new(a), B);



        //
        // Division
        //

        public static Value AutogradDivide(Value A, Value B)
        {
            // A / B = A * B^-1
            Value output = A * Pow(B, -1f);

            return output;
        }
         
        // A / B
        public static Value operator /(Value A, Value B) => AutogradDivide(A, B);

        // A / b
        public static Value operator /(Value A, float b) => AutogradDivide(A, new(b));

        // a / B
        public static Value operator /(float a, Value B) => AutogradDivide(new(a), B);



        //
        // Subtraction
        //

        //A - B
        public static Value operator -(Value A, Value B)
        {
            //A - B = A + (-B)
            return A + (-B);
        }

        //-A (negate)
        public static Value operator -(Value A)
        {
            return A * -1f;
        }



        //
        // Activation functions
        //

        public enum AF
        {
            Tanh, Relu, Sigmoid, Linear
        }

        //Tanh
        public Value Tanh()
        {
            float x = this.data;

            float Exp_2x = MicroMath.Exp(2f * x);

            float tanh = (Exp_2x - 1f) / (Exp_2x + 1f);

            Value output = new(tanh, this, null);

            output.backwardOneStep = () =>
            {
                this.grad += (1f - tanh * tanh) * output.grad;
            };

            return output;
        }

        //Relu
        public Value Relu()
        {
            float x = this.data;
        
            //Relu(x) = max(0, x)
            float relu = x < 0f ? 0f : x;

            Value output = new(relu, this, null);

            output.backwardOneStep = () =>
            {
                //Derivative is 1 if data > 0, otherwhise 0
                this.grad += output.data > 0f ? output.grad : 0f;
            };

            return output;
        }

        //Sigmoid (logistic function)
        public Value Sigmoid()
        {
            float x = this.data;

            float sigmoid = 1f / (1f + MicroMath.Exp(-x));

            Value output = new(sigmoid, this, null);

            output.backwardOneStep = () =>
            {
                this.grad += (sigmoid * (1f - sigmoid)) * output.grad;
            };

            return output;
        }



        //
        // Math
        //

        //Exp e^x
        public Value Exp()
        {
            float x = this.data;

            float exp = MicroMath.Exp(x);

            Value output = new(exp, this, null);

            output.backwardOneStep = () =>
            {
                this.grad += exp * output.grad;
            };

            return output;
        }

        //Pow A^other
        public static Value Pow(Value A, float other)
        {
            float pow = MicroMath.Pow(A.data, other);

            Value output = new(pow, A, null);

            output.backwardOneStep = () =>
            {
                A.grad += (other * MicroMath.Pow(A.data, other - 1f)) * output.grad;
            };

            return output;
        }

        //Log(x) - natural base
        public Value Log()
        {
            float x = this.data;

            float log = MicroMath.Log(x);

            Value output = new(log, this, null);

            output.backwardOneStep = () =>
            {
                this.grad += (1 / x) * output.grad;
            };

            return output;
        }



        //
        // Conversions
        //

        //float -> Value
        public static Value[] Convert(float[] input)
        {
            Value[] output = Array.ConvertAll(input, item => new Value(item));

            return output;
        }

        public static Value[][] Convert(float[][] input)
        {
            Value[][] output = Array.ConvertAll(input, subArray => Value.Convert(subArray));

            return output;
        }



        //
        // Argmax
        //

        //The index with highest value
        public static int Argmax(Value[] a)
        {
            int maxIndex = 0;
            float maxValue = -float.MaxValue;

            for (int j = 0; j < a.Length; j++)
            {
                float value = a[j].data;

                if (value > maxValue)
                {
                    maxIndex = j;
                    maxValue = value;
                }
            }

            return maxIndex;
        }
    }
}