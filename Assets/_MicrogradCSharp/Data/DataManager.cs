using System.Collections;
using System.Collections.Generic;
using UnityEditor.Experimental.GraphView;
using UnityEngine;

namespace Micrograd
{
    //Standardized methods that will make it easier to handle data when training and using Neural Networks
    public class DataManager
    {
        //Remap value from current to wanted
        //We often need to remap input data to range -1 -> 1
        public static float Remap(float value, MinMax currentRange, MinMax wantedRange)
        {
            float remappedValue = wantedRange.min + (value - currentRange.min) * ((wantedRange.max - wantedRange.min) / (currentRange.max - currentRange.min));

            return remappedValue;
        }



        //
        // Batches
        //

        public static int GetNumberOfBatches(int batchSize, int dataSize)
        {
            int numberOfBatches = (int)System.Math.Ceiling((double)dataSize / batchSize);

            return numberOfBatches;
        }



        //Get the start and end index of a batch in the array with all data
        //0 1 2 3 4 5 6 7 8 9 10 11
        //batchSize = 5
        //dataLength = 12
        //-> batches = 3
        //
        //Batch 0
        //batchStartIndex = 0 * 5 = 0
        //batchEndIndex = 0 + 5 - 1 = 4
        //actualBatchSize = 5
        //
        //Batch 1
        //batchStartIndex = 1 * 5 = 5
        //batchEndIndex = 5 + 5 - 1 = 9
        //actualBatchSize = 5
        //
        //Batch 2
        //batchStartIndex = 2 * 5 = 10
        //batchEndIndex = 10 + 5 - 1 = 14 -> 12 - 1 = 11
        public static void GetBatchStartAndEndIndex(int batchNumber, int batchSize, int dataLength, out int batchStartIndex, out int batchEndIndex)
        {
            batchStartIndex = batchNumber * batchSize;

            batchEndIndex = batchStartIndex + batchSize - 1;

            if (batchEndIndex >= dataLength)
            {
                batchEndIndex = dataLength - 1;
            }
        }
    }
}