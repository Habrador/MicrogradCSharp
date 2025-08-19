using System;
using System.Collections;
using System.Collections.Generic;

namespace Data.Twister
{
    public static class TwisterData
    {
        /// <summary>
        /// Generates coordinates [-1,1] for particles that are organized so they look likea Twister or Spiral from above
        /// Can have multiple classes so is suitable for classification experiments
        /// Based on https://cs231n.github.io/neural-networks-case-study/
        /// </summary>
        /// <param name="points">How many points per class?</param>
        /// <param name="classes">How many classes we want?</param>
        /// <param name="rng">A random number generator that can generate numbers from the normal distribution</param>
        /// <returns>The x,y coordinates for each data point and its corresponding class 0, 1, 2, ...</returns>
        public static (float[][] coordinates, int[] classes) Generate(int points, int classes, IGenerateRandomNumbers rng)
        {
            //Number of points per class
            int N = points;
            //Dimensionality (2d space)
            //int D = 2;
            //Number of classes
            int K = classes;

            //Input data matrix (each row = single example) 
            float[][] X = new float[N * K][];
            //Output class labels
            int[] y = new int[N * K];

            //Generate the poinst by using polar coordinates

            //For each class 
            for (int j = 0; j < K; j++)
            {
                //For each point in the class
                for (int i = 0; i < N; i++)
                {
                    //Index in data arrays
                    int ix = N * j + i;

                    //Radius from 0.0 to 1.0
                    //np.linspace(0.0, 1, N)
                    float r = (float)i / (N - 1);

                    //Theta
                    //np.linspace(j * 4, (j + 1) * 4, N)
                    //j = 1, N = 5 -> [4, 5, 6, 7, 8]
                    float theta_min = j * 4f;
                    float theta_max = (j + 1) * 4f;

                    float stepSize = (theta_max - theta_min) / (N - 1);

                    float theta = theta_min + stepSize * i;

                    //Add some random noise to theta
                    //np.random.randn(N)*0.2
                    theta += rng.RandNormal() * 0.2f;

                    //Polar coordinates
                    float xPos = r * (float)System.Math.Sin(theta);
                    float yPos = r * (float)System.Math.Cos(theta);

                    X[ix] = new float[] { xPos, yPos };
                    y[ix] = j;
                }
            }

            return (X, y);
        }
    }
}