using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using UnityEngine;

//Train a neural network to recognize handwritten digits using autograd
public class ImageRecognitionController : MonoBehaviour
{
    public GameObject displayQuadObj;



    private void Start()
    {
        //Use mnist data (minimized version with 100 training and 10 test)
        //csv file, comma-separated
        //Each row consists of 785 values: the first value is the label (0 to 9) and the remaining 784 values are the pixel values (0 to 255)
        string[] trainingDataStringArray = GetRawData("mnist-100", "mnist_train_100.csv");

        Debug.Log(trainingDataStringArray[0]);
        Debug.Log(trainingDataStringArray.Length);

        GetData(trainingDataStringArray, out float[][] inputDataFloat, out float[] outputDataFloat);


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
