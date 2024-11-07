import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten

def create_branch(input_shape):
    input_layer = Input(shape=input_shape)
    
    x = Conv1D(32, 3, activation='relu')(input_layer)
    x = MaxPooling1D(2)(x)
    x = Conv1D(64, 3, activation='relu')(x)
    x = MaxPooling1D(2)(x)
    x = Flatten()(x)

    return input_layer, x
