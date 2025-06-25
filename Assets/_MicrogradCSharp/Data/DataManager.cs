using System.Collections;
using System.Collections.Generic;
using UnityEditor.Experimental.GraphView;
using UnityEngine;

namespace Micrograd
{
    //Standardized methods that will make it easier to handle data when training and using Neural Networks
    public class DataManager
    {
        //Remap value from range 1 to range 2
        //We often need to remap input data to range -1 -> 1
        public static float Remap(float value, MinMax range1, MinMax range2)
        {
            float remappedValue = range2.min + (value - range1.min) * ((range2.max - range2.min) / (range1.max - range1.min));

            return remappedValue;
        }
    }
}