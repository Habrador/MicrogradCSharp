using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace Micrograd
{
    public struct MinMax
    {
        public float min;
        public float max;

        public MinMax(float min, float max)
        {
            this.min = min;
            this.max = max;
        }
    }
}