using Micrograd;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Display
{
    //Display data on a quad so we can see how the numbers looks like
    public static void DisplayDigitOnQuad(float[] grayScaleValues, GameObject displayQuadObj)
    {
        int textureSize = 28;

        //Unity wants color in 0-1 range
        MinMax currentRange = new(0f, 255f);
        MinMax wantedRange = new(0f, 1f);

        //Create a new texture
        Texture2D texture = new(textureSize, textureSize)
        {
            filterMode = FilterMode.Point
        };

        //Convert grayscale data to a texture
        for (int y = 0; y < textureSize; y++)
        {
            for (int x = 0; x < textureSize; x++)
            {
                //Calculate the index in the 1D array
                int index = y * textureSize + x;

                //SHould be in 0-1 range, not 0-255
                float value = DataManager.Remap(grayScaleValues[index], currentRange, wantedRange);

                //Invert color which maybe looks better
                value = 1f - value;

                //Invert the y-coordinate to correct the orientation - otherwise it's upside down
                texture.SetPixel(x, textureSize - 1 - y, new Color(value, value, value));
            }
        }

        texture.Apply();

        //Assign texture to material
        Material quadMaterial = displayQuadObj.GetComponent<MeshRenderer>().material;

        quadMaterial.mainTexture = texture;
    }

}
