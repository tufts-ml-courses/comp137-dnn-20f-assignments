"""
Helper functions for getting parameters from RNN models (a conventional RNN and a GRU). The function reads 
weight matrices within RNN layers and the dispatch their values to corresponding weights or biases for 
the calculation of gates and hidden states. 
"""

import numpy as np


def get_rnn_params(rnn_layer):
    """Get parameters from an RNN layer
    
    inputs: 
        rnn_layer: a `SimpleRNN` layer 
    outputs: 
        wt_h: shape [hidden_size, hidden_size], weight matrix for hidden state transformation
        wt_x: shape [input_size, hidden_size], weight matrix for input transformation
        bias: shape [hidden_size], bias term
    """
    
    weights = rnn_layer.get_weights()
    
    wt_x = weights[0]
    wt_h = weights[1]
    bias = weights[2]
    
    return wt_h, wt_x, bias

def get_gru_params(gru_layer):
    """Get parameters from a GRU layer 
    
    inputs: 
        gru_layer: the `GRU` layer
    outputs: 
        wtz_h: shape [hidden_size, hidden_size], weight matrix for hidden state transformation for z gate
        wtz_x: shape [input_size, hidden_size], weight matrix for input transformation for z gate
        biasz: shape [hidden_size], bias term for z gate
        wtr_h: shape [hidden_size, hidden_size], weight matrix for hidden state transformation for r gate
        wtr_x: shape [input_size, hidden_size], weight matrix for input transformation for r gate
        biasr: shape [hidden_size], bias term for r gate
        wtc_h: shape [hidden_size, hidden_size], weight matrix for hidden state transformation for candicate
               hidden state calculation
        wtc_x: shape [input_size, hidden_size], weight matrix for input transformation for candicate
               hidden state calculation
        biash: shape [hidden_size], bias term for candicate hidden state calculation
    
    """


    weights = gru_layer.get_weights()
    kernel, rec_kernel, bias = weights

    hidden_size = rec_kernel.shape[0]


    # The weights and biases for two gates are concatenated into one matrix. 
    # Separate parameters. The first `hidden_size` columns/elements are for r gate, and the second `hidden_size`
    # columns are for u gate
    wtz_x = kernel[:, :hidden_size]
    wtr_x = kernel[:, hidden_size : 2 * hidden_size]
    wth_x = kernel[:, 2 * hidden_size : ]


    wtz_h = rec_kernel[:, :hidden_size]
    wtr_h = rec_kernel[:, hidden_size : 2 * hidden_size]
    wth_h = rec_kernel[:, 2 * hidden_size : ]


    biasz = bias[:hidden_size]
    biasr = bias[hidden_size : 2 * hidden_size]
    biash = bias[2 * hidden_size : ]

    return wtz_h, wtz_x, biasz, wtr_h, wtr_x, biasr, wth_h, wth_x, biash




