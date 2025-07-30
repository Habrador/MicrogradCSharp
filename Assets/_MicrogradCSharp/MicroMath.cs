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



        //Need a nested class to write Math.Random.Something like in Numpy
        public static class Random
        {
            //This is the random number generator
            private static System.Random rng = new(0);

            //Init the random number generator with a seed so we can get the same "random" numbers
            public static void Seed(int seed)
            {
                rng = new System.Random(seed);
            }


          
            //
            // Generate normally distributed numbers (gaussians)
            //

            public static float Normal(float mean = 0f, float std = 1f) => GetRandomGaussian(mean, std);

            //Generate two normally distributed numbers
            public static void GetRandomGaussian(float mean, float standardDeviation, out float val1, out float val2)
            {
                float u, v, s, t;
                
                do
                {
                    //Generate two random numbers with uniform distribution
                    u = Uniform(-1f, 1f);
                    v = Uniform(-1f, 1f);
                }
                //The numbers have to be within the unit disc and they can't be at the center
                while (u * u + v * v > 1f || (u == 0f && v == 0f));

                //The square of the radius
                s = u * u + v * v;
                
                t = MicroMath.Sqrt((-2f * MicroMath.Log(s)) / s);

                val1 = standardDeviation * u * t + mean;
                val2 = standardDeviation * v * t + mean;
            }

            //Generate one normally distributed number
            public static float GetRandomGaussian(float mean = 0f, float standardDeviation = 1f)
            {
                GetRandomGaussian(mean, standardDeviation, out float rVal1, out _);

                //Just pick one of them
                return rVal1;
            }



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