# MicrogradCSharp

MicrogradCSharp is an open-source Artificial Intelligence project that implements a tiny scalar-valued automatic differentiation (autograd) engine and a Neural Network library for C# within the Unity game engine. There's nothing Unity specific in the library so you can use it for other C# projects as well.  

This library provides a lightweight, efficient, and simple way to build and train Neural Networks directly in Unity. Whether you're prototyping a new game AI or experimenting with Neural Networks, MicrogradCSharp offers a straightforward and intuitive code library to get you started.

> [!CAUTION]
> We all saw in Terminator what can happen if you experiment too much with Artifical Intelligence, please be careful.  


## Example usage of Value class

The idea of scalar-valued automatic differentiation (autograd) engine is to make it easy to find derivatives. If you do some math using the Value class you can find derivatives by typing .Backward(); which is useful when you start experimenting with Neural Networks and encounter Backpropagation.  

```csharp
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
```


## Example usage of Neural Network library

A common "Hello World" example when making Neural Networks is the [XOR gate](https://en.wikipedia.org/wiki/XOR_gate). You want to create a Neural Network that understands the following:

| Input 1  | Input  2 | Output   |
| ---------| -------- | -------- |
| 0        | 0        | 0        |
| 0        | 1        | 1        |
| 1        | 0        | 1        |
| 1        | 1        | 0        |

The code for training and test such a neural network can be coded in as few lines as:

```csharp
MicroMath.Random.Seed(0);

Value[][] inputData = Value.Convert(new [] { new[] { 0f, 0f }, new[] { 0f, 1f }, new[] { 1f, 0f }, new[] { 1f, 1f } });
Value[] outputData = Value.Convert(new[] { 0f, 1f, 1f, 0f });

//2 inputs, 3 neurons in the middle layer, 1 output, tanh transfer function
MLP nn = new(2, new int[] { 3, 1 }, new Value.AF[] { Value.AF.Tanh, Value.AF.Linear }); 

//Train
for (int i = 0; i <= 100; i++)
{
    Value loss = new(0f);

    for (int j = 0; j < inputData.Length; j++) 
    {
        loss += Value.Pow(nn.Activate(inputData[j])[0] - outputData[j], 2f); //MSE loss function
    }

    Debug.Log($"Iteration: {i}, Network error: {loss.data}");

    nn.ZeroGrad();
    loss.Backward(); //The notorious backpropagation

    foreach (Value param in nn.GetParameters())
    {
        param.data += -0.1f * param.grad; //Gradient descent with 0.1 learning rate
    }
}

//Test
for (int j = 0; j < inputData.Length; j++)
{
    Debug.Log("Wanted: " + outputData[j].data + ", Actual: " + nn.Activate(inputData[j])[0].data);
}
```

When I ran the neural network I got the following results:

| Input  1 | Input  2 | Output   |
| ---------| -------- | -------- |
| 0        | 0        | -0,01415 |
| 0        | 1        |  0,98866 |
| 1        | 0        | 0,984572 |
| 1        | 1        | -0,01677 |

The outputs are very close to the 0 and 1 we wanted - the output will never be exactly 0 or 1. 


## Learn more

This project was inspired by by Andrej Karpathy's Micrograd for Python GitHub project [micrograd](https://github.com/karpathy/micrograd) and YouTube video [The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0). They are great sources if you want to learn more what's going on behind the scenes. 
