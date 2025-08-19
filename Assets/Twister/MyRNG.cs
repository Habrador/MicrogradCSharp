using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Data.Twister;

//Twister needs number from the normal distribution
public class MyRNG : IGenerateRandomNumbers
{
    public float RandNormal()
    {
        return Micrograd.MicroMath.Random.Normal();
    }
}
