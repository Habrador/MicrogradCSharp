using Micrograd;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

//Train a Neural Network to classify points 
//The points are organized like in a twister or spiral
public class TwisterController : MonoBehaviour
{
    //Drags
    public GameObject pointObj;
    public GameObject planeObj;


    //Private

    //Green and yellow look too much the same, dont use both
    private readonly Color[] colors = new Color[]
    {
        Color.red, 
        Color.blue,
        Color.yellow,
        Color.white,
        Color.black,
        Color.cyan,
        Color.magenta
    };



    private void Start()
    {
        MicroMath.Random.Seed(0);
    

        //
        // Get and display the data
        //
        
        int classes = 3;

        int points = 40;

        //Twister needs random numbers from the normal distribution
        MyRNG rng = new();

        (float[][] coordinates, int[] labels) = Data.Twister.TwisterData.Generate(points, classes, rng);

        //Display the points
        for (int i = 0; i < coordinates.Length; i++)
        {
            float x = coordinates[i][0];
            float y = coordinates[i][1];

            int label = labels[i];

            AddSphere(x, y, label);
        }



        //
        // Init data
        //

        //The coordinates are [-1,1] so we dont need to normalize those

        //Convert data from float to Value
        Value[][] coordinatesValue = Value.Convert(coordinates);



        //
        // Init the Neural Network
        //

        MLP nn = new();

        //2 inputs (x,y coordinates), 3 ouputs because 3 classes
        nn.AddLayer(nn.Linear(2, 16));
        nn.AddLayer(nn.ReLU());
        nn.AddLayer(nn.Linear(16, 3));
        nn.AddLayer(nn.Softmax());

        

        //
        // Train NN
        //

        float learningRate = 1f;

        int epochs = 100;

        System.Diagnostics.Stopwatch timer = new();

        timer.Start();

        Train(nn, learningRate, epochs, coordinatesValue, labels);

        Debug.Log($"Time to train NN: {timer.ElapsedMilliseconds / 1000f} seconds");


        
        //
        // Test NN
        //

        //Display how good the network is on a plane
        DisplayNNOnQuad(nn, planeObj);

        //Calculate how good the network is
        CalculateAccuracy(nn, coordinatesValue, labels);        
    }



    //Calculate the accuracy of the NN
    //If the one-hot output is predicted correctly it scores 1, otherwise 0
    //Accuracy is average
    private void CalculateAccuracy(MLP nn, Value[][] input, int[] labels)
    {
        int score = 0;

        for (int i = 0; i < input.Length; i++)
        {
            Value[] output = nn.Activate(input[i]);

            int predictedClass = Value.Argmax(output);

            int wantedClass = labels[i];

            if (wantedClass == predictedClass)
            {
                score += 1;
            }
        }

        float accuracy = (float)score / input.Length;

        Debug.Log($"Network accuracy; {accuracy}%");
    }



    //Display how well the NN predicts a category by for each pixel on a plane predict the category
    private void DisplayNNOnQuad(MLP nn, GameObject planeObj)
    {
        //We know the coordinates are in the range [-1, 1]
        float min = -1f;
        float max = 1f;

        int resolution = 100;

        float step = (max - min) / resolution;

        //resolution = 4
        //| | | | |
        //| | | | |
        //| | | | |
        //| | | | |
        //We want the coordinate in the middle of each cell
        //x -> -0.75, -0.25, 0.25, 0.75
        //step = (1 - -1) / 4 = 0.5

        //Create a new texture
        Texture2D texture = new(resolution, resolution)
        {
            //filterMode = FilterMode.Point,
            filterMode = FilterMode.Bilinear,
            wrapMode = TextureWrapMode.Clamp
        };

        //Fill the texture with colors

        //Predict from the center of each cell
        float yCoordinate = min + step * 0.5f;

        for (int y = 0; y < resolution; y++)
        {
            float xCoordinate = min + step * 0.5f;

            for (int x = 0; x < resolution; x++)
            {
                //Run the NN and predict the color
                Value[] coordinates = new Value[] { new(xCoordinate), new(yCoordinate) };

                Value[] output = nn.Activate(coordinates);

                int category = Value.Argmax(output);

                Color thisColor = colors[category];

                texture.SetPixel(x, y, thisColor);

                xCoordinate += step;
            }

            yCoordinate += step;
        }

        texture.Apply();

        //Assign texture to material
        Material quadMaterial = planeObj.GetComponent<MeshRenderer>().material;

        quadMaterial.mainTexture = texture;
    }



    //The training loop
    private void Train(MLP nn, float learningRate, int epochs, Value[][] input, int[] labels)
    {
        for (int i = 0; i <= epochs; i++)
        {
            int batchSize = input.Length;


            //Forward
            Value[][] outputs = new Value[batchSize][];

            for (int j = 0; j < batchSize; j++)
            {
                outputs[j] = nn.Activate(input[j]);
            }


            //Calculate loss
            Value loss = nn.NLL_Loss(outputs, labels);

            //Display the loss
            if (i % 10 == 0)
            {
                Debug.Log($"Iteration: {i}, Network error: {loss.data}");
            }


            //Backward
            
            //Reset gradients
            nn.ZeroGrad();

            //Calculate new gradients
            loss.Backward();

            //Optimize the weights and biases by using gradient descent
            Value[] parameters = nn.GetParameters();

            foreach (Value param in parameters)
            {
                param.data -= learningRate * param.grad;
            }
        }
    }



    //Add a sphere to the game world so we can visualize the points
    private void AddSphere(float x, float y, int label)
    {
        GameObject newPointObj = Instantiate(pointObj);

        Vector3 pos = new(x, y, 0f);

        float scale = 0.05f;
        
        newPointObj.transform.position = pos;
        newPointObj.transform.localScale = Vector3.one * scale;

        newPointObj.GetComponent<MeshRenderer>().material.color = colors[label];
    }
}
