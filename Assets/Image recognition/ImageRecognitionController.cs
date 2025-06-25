using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;
using Micrograd;

//Train a neural network to recognize handwritten digits using autograd
public class ImageRecognitionController : MonoBehaviour
{
    public GameObject displayQuadObj;



    private void Start()
    {
        //Use mnist data (minimized version with 100 training and 10 test)
        //csv file, comma-separated
        //Each row consists of 785 values:
        //- the first value is the label (0 to 9)
        //- the remaining 784 (28x28) values are the pixel values (0 to 255)
        string[] trainingDataStringArray = GetRawData("mnist-100", "mnist_train_100.csv");

        Debug.Log(trainingDataStringArray[0]);
        Debug.Log(trainingDataStringArray.Length);

        GetData(trainingDataStringArray, out float[][] inputDataFloat, out float[] outputDataFloat);

        //Normalize 
        //Cant do that in GetData because we might not know the max and min values
        //We know inputdata is grayscale number 0 -> 255
        MinMax currentRange = new(0f, 255f);
        //The range we want the data to be in
        MinMax wantedRange = new(0f,1f);

        float[][] normalizedInputDataFloat = NormalizeData(inputDataFloat, wantedRange, currentRange);

        //Do we need to normalize the output as well???


        //Display the data
        int displayIndex = 6;

        DisplayDigitOnQuad(inputDataFloat[displayIndex]);

        Debug.Log(outputDataFloat[displayIndex]);
    }



    //Display data on a quad so we can see how the numbers looks like
    private void DisplayDigitOnQuad(float[] grayScaleValues)
    {
        int textureSize = 28;

        //Unity wants color in 0-1 range
        MinMax currentRange = new(0f, 255f);
        MinMax wantedRange = new(0f, 1f);

        //Create a new texture
        Texture2D texture = new(textureSize, textureSize)
        {
            filterMode = FilterMode.Point
        };

        //Convert grayscale data to a texture
        for (int y = 0; y < textureSize; y++)
        {
            for (int x = 0; x < textureSize; x++)
            {
                //Calculate the index in the 1D array
                int index = y * textureSize + x;
                
                //SHould be in 0-1 range, not 0-255
                float value = DataManager.Remap(grayScaleValues[index], currentRange, wantedRange);

                //Invert color which maybe looks better
                value = 1f - value;

                //Invert the y-coordinate to correct the orientation - otherwise it's upside down
                texture.SetPixel(x, textureSize - 1 - y, new Color(value, value, value));
            }
        }

        texture.Apply();

        //Assign texture to material
        Material quadMaterial = displayQuadObj.GetComponent<MeshRenderer>().material;

        quadMaterial.mainTexture = texture;
    }



    //Normalize data array from one range to another
    private float[][] NormalizeData(float[][] inputData, MinMax wantedRange, MinMax currentRange)
    {
        float[][] normalizedData = new float[inputData.Length][];
        
        for (int i = 0; i < inputData.Length; i++)
        {
            float[] normalizedSubData = new float[inputData[i].Length];

            for (int j = 0; j < inputData[i].Length; j++)
            {
                normalizedSubData[j] = DataManager.Remap(inputData[i][j], currentRange, wantedRange);
            }

            normalizedData[i] = normalizedSubData;
        }

        return normalizedData;
    }



    //Standardize the data from the raw file
    private void GetData(string[] dataStringArray, out float[][] inputData, out float[] outputData)
    {
        inputData = new float[dataStringArray.Length][];
        outputData = new float[dataStringArray.Length];
    
        for (int i = 0; i < dataStringArray.Length; i++)
        {
            //The values in the string a comma-separated
            string[] dataString = dataStringArray[i].Split(',');

            //string -> float
            float[] dataCombined = Array.ConvertAll(dataString, float.Parse);

            //Split label and image data
            float label = dataCombined[0];
            float[] pixelValues = dataCombined.Skip(1).ToArray();

            //Debug.Log(label);
            //Debug.Log(pixelValues[0]);
            //Debug.Log(pixelValues.Length); //784

            inputData[i] = pixelValues;
            outputData[i] = label;
        }
    }



    //Get all rows from a file located in Assets/StreamingAssets folder
    public static string[] GetRawData(string folderName, string fileName)
    {
        //File should be in Assets/StreamingAssets
        string[] allRowsString = File.ReadAllLines(Path.Combine(Application.streamingAssetsPath, $"{folderName}/{fileName}"));

        return allRowsString;
    }

}
