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
    }
}