using System.Collections;
using System.Collections.Generic;
using System.Linq;


namespace Micrograd
{
    //Instead of shuffling the actual array with the data, which might be slow, we shuffle an array with indicies
    //When we need an index, we take and index from here instead of the actual index
    //actualArray: [3.23, 5.65, 7.65]
    //shuffledIndices: [1, 0, 2]
    //randomItem0 = actualArray[shuffledArray[0]] = 5.65
    public class Permutator
    {
        private readonly int[] indices;

        private readonly System.Random rng;

        //Get a shuffled index
        public int this[int i] => indices[i];

        public int Size => indices.Length;



        public Permutator(int size, System.Random myRng = null)
        {
            this.indices = Enumerable.Range(0, size).ToArray();

            if (myRng == null)
            {
                this.rng = new System.Random();
            }
            else
            {
                this.rng = myRng; 
            }
        }



        //Shuffle using the Fisher-Yates shuffle algorithm: https://www.dotnetperls.com/fisher-yates-shuffle
        public void Shuffle()
        {
            int n = indices.Length;

            for (int i = 0; i < n - 1; i++)
            {
                int swapIndex = i + rng.Next(0, n - i);

                //Swap
                (indices[i], indices[swapIndex]) = (indices[swapIndex], indices[i]);
            }
        }
    }
}
