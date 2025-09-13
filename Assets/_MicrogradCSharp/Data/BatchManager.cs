using Micrograd;
using System.Collections;
using System.Collections.Generic;

namespace Micrograd
{
    public class BatchManager
    {
        private readonly int dataSize;
        private readonly int batchSize;
        public int BatchCount { get; private set; }

        private readonly Permutator dataShuffler;



        public BatchManager(int dataSize, int batchSize, System.Random myRng = null)
        {
            this.dataSize = dataSize;
            this.batchSize = batchSize;

            this.BatchCount = GetNumberOfBatches(batchSize, dataSize);

            this.dataShuffler = new Permutator(dataSize, myRng);
        }



        //How many batches do we have?
        private int GetNumberOfBatches(int batchSize, int dataSize)
        {
            int numberOfBatches = (int)System.Math.Ceiling((double)dataSize / batchSize);

            return numberOfBatches;
        }



        //Shuffle the indices
        public void ShuffleIndex()
        {
            dataShuffler.Shuffle();
        }



        //Get an array with shuffled index belonging to a batch
        //Such as [2, 8, 1, 3]
        public int[] GetShuffledBatchIndex(int batchNumber)
        {
            List<int> shuffledBatchIndices = new();

            //The indices in dataShuffler array for this batch
            GetBatchStartAndEndIndex(batchNumber, batchSize, dataShuffler.Size, out int batchStartIndex, out int batchEndIndex);

            for (int i = batchStartIndex; i <= batchEndIndex; i++)
            {
                shuffledBatchIndices.Add(dataShuffler[i]);
            }

            return shuffledBatchIndices.ToArray();
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
        //actualBatchSize = 2
        private void GetBatchStartAndEndIndex(int batchNumber, int batchSize, int dataLength, out int batchStartIndex, out int batchEndIndex)
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