using Micrograd;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ValueExamples : MonoBehaviour
{
    private void Start()
    {
        //TestGradients();

        //HowDerivativesWork();

        //BasicNeuron();

        XOR_Gate_Just_Values xor = new();
    }



    //Test example from the micrograd github repo https://github.com/karpathy/micrograd
    public void TestGradients()
    {
        Value a = new(-4.0f);

        Value b = new(2.0f);

        Value c = a + b;

        Value d = a * b + Value.Pow(b, 3f);

        c += c + 1f;

        c += 1f + c + (-a);

        d += d * 2f + (b + a).Relu();

        d += 3f * d + (b - a).Relu();

        Value e = c - d;

        Value f = Value.Pow(e, 2f);

        Value g = f / 2.0f;

        g += 10.0f / f;

        Debug.Log("Expected: 24.7041, Actual: " + g.data);

        g.Backward();

        //dg/da
        Debug.Log("Expected: 138.8338, Actual: " + a.grad);


        //dg/db
        Debug.Log("Expected: 645.5773, Actual: " + b.grad);

        //Works!
    }



    //Basic neuron from the YT video "The spelled-out intro to neural networks and backpropagation: building micrograd"
    public void BasicNeuron()
    {
        //Basic neuron

        //Input
        Value x1 = new(2f);
        Value x2 = new(0f);
        //Weights
        Value w1 = new(-3f);
        Value w2 = new(1f);
        //Bias
        Value b = new(6.8813735870195432f); //To get nice grads


        //Forward pass
        Value x1w1 = x1 * w1;
        Value x2w2 = x2 * w2;

        Value x1w1x2w2 = x1w1 + x2w2;

        Value n = x1w1x2w2 + b;

        Value o = n.Tanh();

        Debug.Log("Output: " + o.data);


        //Backward pass
        /*
        o.grad = 1f; //We have to init because is initialized to 0
  
        o.backward();

        n.backward();

        b.backward(); //Null error because leaf node
        x1w1x2w2.backward();

        x1w1.backward();
        x2w2.backward();
        */

        //Automatic Backward pass using topological sort
        o.Backward();


        Debug.Log("Gradients:");

        Debug.Log("Wanted: 0.5, Actual: " + n.grad);

        Debug.Log("Wanted: 0.5, Actual: " + x1w1x2w2.grad);
        Debug.Log("Wanted: 0.5, Actual: " + b.grad);

        Debug.Log("Wanted: 0.5, Actual: " + x1w1.grad);
        Debug.Log("Wanted: 0.5, Actual: " + x2w2.grad);

        Debug.Log("Wanted: -1.5, Actual: " + x1.grad);
        Debug.Log("Wanted: 1, Actual: " + w1.grad);
        Debug.Log("Wanted: 0.5, Actual: " + x2.grad);
        Debug.Log("Wanted: 0, Actual: " + w2.grad);
    }



    //How derivatives work from the YT video "The spelled-out intro to neural networks and backpropagation: building micrograd"
    public void HowDerivativesWork()
    {
        Value a = new(2f);
        Value b = new(-3f);
        Value c = new(10f);
        Value f = new(-2f);

        Value d = a * b + c;

        Value L = d * f;

        //a
        // \ * -> e       
        // /       \ + -> d 
        //b        /       \ * -> L
        //        c        /
        //                f

        //Values
        //e = a * b = 2 * -3 = -6
        //d = e + c = -6 + 10 = 4
        //L = d * f = 4 * -2 = -8

        Debug.Log(d.data);

        Debug.Log(d.childA.data);
        Debug.Log(d.childB.data);

        Debug.Log(L.data);

        //Gradients

        //dL/dL = 1

        //L = d * f:
        //dL/dd = f = -2
        //dL/df = d = 4

        //dL/dc
        //d = c + e -> dd/dc = 1
        //(f(x+h) - f(x)) / h -> ((c + h + e) - (c + e))/h = (c + h + e - c - e) / h = 1
        //The chain rule: dz/dx = dz/dy * dy/dx -> dL/dc = dL/dd * dd/dc = -2 * 1 = -2

        //dL/de (as above)
        //dL/de = dL/dd * dd/de = -2 * 1 = -2

        //dL/da = dL/de * de/da 
        //e = a * b -> de/da = b = -3 -> dL/da = -2 * -3 = 6

        //dL/db = dL/de * de/db
        //e = a * b -> de/db = a = 2 -> dL/da = -2 * 2 = -4


        Debug.Log("Derivative: " + ApproximateDerivative());
    }


    //Approximate the derivative
    private float ApproximateDerivative()
    {
        float h = 0.0001f;

        float a = 2f;
        float b = -3f;
        float c = 10f;
        float f = -2f;

        float L1 = TheEquation(a, b, c, f);
        //dL/db: float L2 = TheEquation(a, b + h, c, f);
        //dL/dL: float L2 = TheEquation(a, b, c, f) + h;

        float L2 = TheEquation(a, b + h, c, f);

        float derivative = (L2 - L1) / h;

        return derivative;
    }

    private float TheEquation(float a, float b, float c, float f)
    {
        float d = a * b + c;
        float L = d * f;

        return L;
    }

}
