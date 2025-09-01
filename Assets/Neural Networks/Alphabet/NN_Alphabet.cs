using Micrograd;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

//Most basic neural network to predict next character in the alphabet given a single previous character
//Uses classification so Softmax activation function in output layer
//Based on Andrej Karpathy's makemore YouTube series
//Is also using batches
public class NN_Alphabet: MonoBehaviour
{
    private void Start()
    {
        MicroMath.Random.Seed(2);

        Generate();
    }



    private void Generate()
    {
        string alphabet = "abcdefghijklmnopqrstuvwxyz";

        //Generate lookup tables for all characters (plus the . we sacrifice to denote start and end of word)
        //char -> int
        //int -> char
        GenerateLookupTables(alphabet, out Dictionary<char, int> stoi, out Dictionary<int, char> itos);



        //
        // Create the training set of bigrams (.,a),(a,b), etc
        //
        
        //Input
        List<int> xs = new();
        //Wanted output given the input
        List<int> ys = new();

        //Add a special character to know if a character is a start or end
        string word = "." + alphabet + ".";

        //Turn the word we have into an array of chars
        List<char> chs = new(word);

        //For each character pair
        for (int j = 1; j < chs.Count; j++)
        {
            char ch1 = chs[j - 1];
            char ch2 = chs[j + 0];

            //Debug.Log(ch1 + " " + ch2);

            xs.Add(stoi[ch1]);
            ys.Add(stoi[ch2]);
        }


        //We need the data to be one-hot-encoded Values
        Value[][] xec = Value.ToOneHotEncoding(xs.ToArray(), dimensions: 27);

        Debug.Log($"Training samples: {xec.Length}");



        //
        // Init the NN
        //

        System.Random rng = Micrograd.MicroMath.Random.GetGenerator;

        //How fast/slow the network will learn
        float learningRate = 1f;

        //How many times to go through all data when learning
        int epochs = 20;

        //How many training samples do we show the network before we do backprop?
        int batchSize = 5;

        //Create the NN
        MLP nn = new();

        //Add layers
        nn.AddLayers(
            nn.Linear(27, 27, useBias: true),
            nn.Softmax()
        );



        //
        // Test NN before training
        //

        Debug.Log("Before training:");

        TestNN(nn, stoi, itos);



        //
        // Train the NN
        //

        Debug.Log("Training!");

        TrainNN(nn, epochs, batchSize, learningRate, xec, ys, rng);



        //
        // Test NN after training
        //

        Debug.Log("After training:");

        TestNN(nn, stoi, itos);
    }



    //Test how good the network is
    private void TestNN(MLP nn, Dictionary<char, int> stoi, Dictionary<int, char> itos)
    {
        //Start with the dot and end the loop when we see another dot
        char startCharacter = '.';

        int ix = stoi[startCharacter];

        //To avoid getting stuck in infinite loop if we never reaches the end character
        int maxLength = 100;

        List<char> generatedCharacters = new();

        for (int i = 0; i < maxLength; i++)
        {
            //Index to one-hot
            float[] onehot = new float[27];

            onehot[ix] = 1f;

            //float -> value
            Value[] NNinput = Value.Convert(onehot);

            //Run the data through the nn
            Value[] NNoutput = nn.Activate(NNinput);

            //The index with highest probability which becomes the next index
            ix = Value.Argmax(NNoutput);

            char ixChar = itos[ix];

            generatedCharacters.Add(ixChar);

            if (ixChar == '.')
            {
                break;
            }
        }

        string generatedWord = new(generatedCharacters.ToArray());

        Debug.Log(generatedWord);
    }



    //Training loop
    private void TrainNN(MLP nn, int epochs, int batchSize, float learningRate, Value[][] xecValue, List<int> ys, System.Random rng)
    {
        //How many batches?
        int batches = Micrograd.DataManager.GetNumberOfBatches(batchSize, xecValue.Length);

        //Shuffle data each epoch so each batch always includes different examples
        Permutator dataShuffler = new(xecValue.Length, rng);

        //Timer
        System.Diagnostics.Stopwatch timer = new();
        

        //Train
        timer.Start();

        for (int i = 0; i <= epochs; i++)
        {
            float epochLoss = 0f;

            //Shuffle the data
            dataShuffler.Permute();

            for (int batchNumber = 0; batchNumber < batches; batchNumber++)
            {
                //Cache the shuffled data once in an array which makes it less confusing
                List<int> shuffledBatchIndices = new();

                //The indices in dataShuffler array for this batch
                Micrograd.DataManager.GetBatchStartAndEndIndex(batchNumber, batchSize, xecValue.Length, out int batchStartIndex, out int batchEndIndex);

                for (int j = batchStartIndex; j <= batchEndIndex; j++)
                {
                    shuffledBatchIndices.Add(dataShuffler[j]);
                }

                int actualBatchSize = shuffledBatchIndices.Count;


                //Forward pass

                //Catch all outputs for this batch
                Value[][] networkOutputs = new Value[actualBatchSize][];

                for (int j = 0; j < shuffledBatchIndices.Count; j++)
                {
                    int shuffledIndex = shuffledBatchIndices[j];

                    Value[] inputData = xecValue[shuffledIndex];
                    
                    //Run input through the network
                    Value[] output = nn.Activate(inputData);
                    
                    //Cache the output
                    networkOutputs[j] = output;
                }


                //Loss calculations using negative log likelihood aka Categorical Crossentropy
                //The shuffled indices where we want the one-hot to be 1
                int[] oneHotOutputIndices = shuffledBatchIndices.Select(shuffledIndex => ys[shuffledIndex]).ToArray();

                Value batchLoss = CategoricalCrossentropy.Forward(networkOutputs, oneHotOutputIndices);

                epochLoss += batchLoss.data;


                //Backward pass
                //Reset the gradients
                nn.ZeroGrad();
                //Calculate the gradients
                batchLoss.Backward();

                //Optimize the weights and biases by using gradient descent
                Value[] parameters = nn.GetParameters();

                foreach (Value param in parameters)
                {
                    param.data -= learningRate * param.grad;
                }
            }

            //Average loss over all batches
            epochLoss = epochLoss / batches;

            //Display the loss
            if (i % 10 == 0)
            {
                Debug.Log($"Iteration: {i}, Network error: {epochLoss}");
            }
        }

        timer.Stop();

        Debug.Log($"It took: {timer.ElapsedMilliseconds / 1000f} seconds to train the network");
    }



    //Neural networks don't understand characters so we need to associate each character with a number
    public static void GenerateLookupTables(string word, out Dictionary<char, int> stoi, out Dictionary<int, char> itos)
    {
        //Create a list of all individual characters we have in the data set
        char[] chars = word.ToArray();

       
        //Create a lookup tables

        //Mapping char -> int (a -> 1, b -> 2, etc)
        stoi = new();

        //Sacrifice the dot to denote start or end character
        stoi['.'] = 0;

        for (int i = 0; i < chars.Length; i++)
        {
            stoi[chars[i]] = i + 1; //The special character . starts at 0
        }

        //We also need the inverted lookup table: int -> char (1 -> a, 2 -> b, etc)
        itos = new();

        foreach (KeyValuePair<char, int> entry in stoi)
        {
            itos[entry.Value] = entry.Key;
        }
    }
}
