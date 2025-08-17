using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace Micrograd
{
    //Baseclass for layers
    public abstract class Layer : Module
    {
        public abstract Value[] Activate(Value[] x);

    }
}