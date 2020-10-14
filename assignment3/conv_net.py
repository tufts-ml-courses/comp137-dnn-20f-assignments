"""
Implementation of a ResNet. 
"""

import tensorflow as tf
import tensorflow.keras as keras

def ConvNet(**kwargs):
    """
    Construct a ResNet using `tf.keras` layers
    """
    
    # TODO: implement your own model

    # This is an example model, which sums up all pixels and does classification 
    model = tf.keras.Sequential([tf.keras.layers.GlobalAvgPool2D(), 
                                 tf.keras.layers.Dense(units=10)])


    return model 
