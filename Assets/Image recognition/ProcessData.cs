using Micrograd;
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;

public static class ProcessData
{
    public static Value[][] PreprocessData(float[][] data)
    {
        //Normalize input data
        //Cant do that in GetData because we might not know the max and min values
        //We know inputdata is grayscale number 0 -> 255
        MinMax currentRange = new(0f, 255f);
        //The range we want the data to be in
        MinMax wantedRange = new(0f, 1f);

        float[][] normalizedDataFloat = ProcessData.NormalizeData(data, wantedRange, currentRange);

        //No need to normalize the output data because we are doing classification not regression...
        //and then label is one-hot-encoded and not a number
        //float[][] outputDataFloat = new 
        //Do we need to one-hot-encode, better to modify the loss function or we will have arrays with useless 0s
        //float[]

        //float -> Value
        Value[][] preprocessedData = Value.Convert(normalizedDataFloat);

        return preprocessedData;
    }



    //Normalize data array from one range to another
    public static float[][] NormalizeData(float[][] inputData, MinMax wantedRange, MinMax currentRange)
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
    public static void GetData(string[] dataStringArray, out float[][] pixels, out int[] labels)
    {
        pixels = new float[dataStringArray.Length][];
        labels = new int[dataStringArray.Length];

        for (int i = 0; i < dataStringArray.Length; i++)
        {
            //The values in the string a comma-separated
            string[] dataString = dataStringArray[i].Split(',');

            //string -> float
            float[] dataCombined = Array.ConvertAll(dataString, float.Parse);

            //Split label and image data
            int label = (int)dataCombined[0];
            float[] pixelValues = dataCombined.Skip(1).ToArray();

            //Debug.Log(label);
            //Debug.Log(pixelValues[0]);
            //Debug.Log(pixelValues.Length); //784

            pixels[i] = pixelValues;
            labels[i] = label;
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
