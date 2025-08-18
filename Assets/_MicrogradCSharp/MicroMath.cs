using System.Collections;
using System.Collections.Generic;


namespace Micrograd
{
    //Useful math for neural networks
    public static class MicroMath
    {
        //Pow x^y
        public static float Pow(float x, float y) => (float)System.Math.Pow(x, y);

        //Exp e^x
        public static float Exp(float x) => (float)System.Math.Exp(x);

        //Sqrt(x)
        public static float Sqrt(float x) => (float)System.Math.Sqrt(x);

        //Log(x)
        public static float Log(float x) => (float)System.Math.Log(x);

        //Cos(x)
        public static float Cos(float x) => (float)System.Math.Cos(x);

        //Sin(x)
        public static float Sin(float x) => (float)System.Math.Sin(x);

        //Sign(x)
        public static int Sign(float x) => System.Math.Sign(x);

        //Clamp to min max range
        public static float Clamp(float value, float min, float max) => System.Math.Clamp(value, min, max);

        //PI
        public static float PI => (float)System.Math.PI;



        //Need a nested class to write Math.Random.Something like in Numpy
        public static class Random
        {
            //This is the random number generator
            private static System.Random rng = new(0);

            //Get the rng because sometimes we need to inject it
            public static System.Random GetGenerator => rng;

            //Init the random number generator with a seed so we can get the same "random" numbers
            public static void Seed(int seed)
            {
                rng = new System.Random(seed);
            }



            //
            // Generate normally distributed numbers (gaussians)
            //

            //The Box-Muller transform converts uniformly distributed random numbers into normally distributed ones
            //https://stackoverflow.com/questions/218060/random-gaussian-variables
            public static float GetRandomGaussian(float mean, float stdDev)
            {
                float u1 = 1f - Uniform01;
                float u2 = 1f - Uniform01;

                //random normal(0,1)
                float randStdNormal = MicroMath.Sqrt(-2f * MicroMath.Log(u1)) * MicroMath.Sin(2f * MicroMath.PI * u2);

                //random normal(mean,stdDev^2)
                float randNormal = mean + stdDev * randStdNormal;
                
                return randNormal;
            }

            public static float Normal(float mean = 0f, float std = 1f) => GetRandomGaussian(mean, std);



            //
            // Generate uniformly distributed numbers
            //

            //Float [low, < high]
            public static float Uniform(float low, float high) => (float)rng.NextDouble() * (high - low) + low;

            //Float 0 -> 1
            public static float Uniform01 => Uniform(0f, 1f);

            //Integer
            public static int RandInt(int low, int high) => rng.Next(low, high);
        }
    }
}