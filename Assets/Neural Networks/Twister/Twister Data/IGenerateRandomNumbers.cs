using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace Data.Twister
{
    //So Twister doesnt depend on some custom class that generates random numbers from the normal distribution
    public interface IGenerateRandomNumbers
    {
        public float RandNormal();
    }
}