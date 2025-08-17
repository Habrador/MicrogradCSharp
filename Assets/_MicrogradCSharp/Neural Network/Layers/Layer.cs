using System.Collections;
using System.Collections.Generic;

namespace Micrograd
{
    //Baseclass for layers
    public abstract class Layer : Module
    {
        public abstract Value[] Activate(Value[] x);

    }
}