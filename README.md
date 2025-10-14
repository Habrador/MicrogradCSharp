# MicrogradCSharp

MicrogradCSharp is an open-source Artificial Intelligence project. It implements a scalar-valued automatic differentiation (autograd) engine, so you don't have to implement backpropagation on your own, and a Neural Network library for C# within the Unity game engine. There's nothing Unity specific in the library so you can use it for other C# projects as well.  

This library provides a lightweight, efficient, and simple way to build and train Neural Networks directly in Unity. Whether you're prototyping a new game AI or experimenting with Neural Networks, MicrogradCSharp offers a straightforward and intuitive code library to get you started. You can do regression, you can do classification - and you can even do reinforcement learning. 

> [!CAUTION]
> We all saw in Terminator what can happen if you experiment too much with Artifical Intelligence, please be careful.  


## Implementations

Activation functions:

* Tanh
* Relu
* Sigmoid
* Softmax

Loss functions:

* Negative Log-Likelihood (NLL)
* Mean Squared Error (MSE)

Optimizers:

* SGD with momentum
* Adam

Neural Networks:

* Regression:
	* XOR gate
	* Some example Andrej Karpathy used

* Classification:
	* Predict the next character in the alphabet given a character
	* Predict the class of a point organized like a twister

<img src="/_media/twister-classification.jpg" width="200">

Help functions:

* Batch manager that makes it easy to work with batches when training Neural Networks


## How to create a Neural Network?

You can create it layer-by-layer:

```csharp
//Create the NN
MLP nn = new();

//Add layers
//2 inputs, 3 neurons in the middle layer with tanh activation function, 1 output with no activation function
nn.AddLayer(nn.Linear(2, 3));
nn.AddLayer(nn.Tanh());
nn.AddLayer(nn.Linear(3, 1));
```


...or in one go:

```csharp
//Create the NN
MLP nn = new();

//Add layers
//2 inputs, 3 neurons in the middle layer with tanh activation function, 1 output with no activation function
nn.AddLayers(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 1));
```


## Example usage of Neural Network library

A common "Hello World" example when making Neural Networks is the [XOR gate](https://en.wikipedia.org/wiki/XOR_gate). You want to create a Neural Network that understands the following:

| Input 1  | Input  2 | Output   |
| ---------| -------- | -------- |
| 0        | 0        | 0        |
| 0        | 1        | 1        |
| 1        | 0        | 1        |
| 1        | 1        | 0        |

The minimal Neural Network to learn this example has 2 inputs, 3 neurons in the middle layer, and 1 output. It also has 2 biases connected to the middle layer and the ouput layer. A bias always has input 1 and the bias's weight can be trained like the other weights in the network. It looks like this: 

<img src="/_media/neural-network-2-3-1.jpg" width="400">

The code for training and test such a Neural Network can be coded in as few lines as:

```csharp
MicroMath.Random.Seed(0);

Value[][] inputData = Value.Convert(new [] { new[] { 0f, 0f }, new[] { 0f, 1f }, new[] { 1f, 0f }, new[] { 1f, 1f } });
Value[] outputData = Value.Convert(new[] { 0f, 1f, 1f, 0f });

MLP nn = new();

//2 inputs, 3 neurons in the middle layer with tanh activation function, 1 output with no activation function
nn.AddLayers(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 1)); 

//Optimizer that will do gradient descent for us
Adam optimizer = nn.Adam_Optimizer(nn.GetParameters(), learningRate: 0.1f);

//Train
for (int i = 0; i <= 100; i++)
{
    Value loss = new(0f);

    for (int j = 0; j < inputData.Length; j++) 
    {
        loss += Value.Pow(nn.Activate(inputData[j])[0] - outputData[j], 2f); //MSE loss function without the M
    }

    Debug.Log($"Iteration: {i}, Network error: {loss.data}");

    optimizer.ZeroGrad(); //Reset gradients

	loss.Backward(); //The notorious backpropagation

	optimizer.Step(); //Update weights and biases
}

//Test
for (int j = 0; j < inputData.Length; j++)
{
    Debug.Log("Wanted: " + outputData[j].data + ", Actual: " + nn.Activate(inputData[j])[0].data);
}
```

When I ran the Neural Network I got the following results:

| Input 1  | Input  2 | Wanted   | Actual   |
| ---------| -------- | -------- | -------- |
| 0        | 0        | 0        | 0,008705 |
| 0        | 1        | 1        | 0,994957 |
| 1        | 0        | 1        | 0,993833 |
| 1        | 1        | 0        | 0,006619 |

The outputs are very close to the 0 and 1 we wanted - the output will never be exactly 0 or 1. 


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


## Learn more

This project was inspired by by Andrej Karpathy's Micrograd for Python GitHub project [micrograd](https://github.com/karpathy/micrograd) and YouTube video [The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0). They are great sources if you want to learn more what's going on behind the scenes. 
