"""
This problem is modified from a problem in Stanford CS 231n assignment 1. 
In this problem, we implement the neural network with tensorflow instead of numpy
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class DenseLayer(tf.keras.layers.Layer):
    """
    Implement a dense layer 
    """

    def __init__(self, input_dim, output_dim, activation, reg_weight, param_init='autograder'):

        """
        Initialize weights of the DenseLayer. In Tensorflow's implementation, the weight 
        matrix and bias term are initialized in the `build` function. Here let's use the simple form. 

        https://www.tensorflow.org/guide/keras/custom_layers_and_models

        args:
            input_dim: integer, the dimension of the input layer
            output_dim: integer, the dimension of the output layer
            activation: string, can be 'linear', 'relu', 'tanh', 'sigmoid', or 'softmax'. 
                        It specifies the activation function of the layer
            reg_weight: the regularization weight/strength, the lambda value in a regularization 
                        term 0.5 * \lambda * ||W||_2^2
                        
            param_init: `dict('W'=W, 'b'=b)`. Here W and b should be `np.array((input_dim, output_dim))` 
                        and `np.array((1, output_dim))`. The weight matrix and the bias vector are 
                        initialized by `W` and `b`. 
                        NOTE: `param_init` is used to check the correctness of your function. For you 
                        own usage, `param_init` can be `None`, and the parameters are initialized 
                        within this function. But when `param_init` is not None, you should 
                        `param_init` to initialize your function. 

        """


        super(DenseLayer, self).__init__()

        # initialize with small values of both positive and negatives.
        param_init = dict(W=None, b=None)

        if param_init == 'autograder':
            np.random.seed(137)
            param_init['W'] = np.random.random_sample((input_dim, output_dim)) 
            param_init['b'] = np.random.random_sample((output_dim, )) 
        else:
            param_init['W'] = (np.random.random_sample((input_dim, output_dim)) - 0.5) * 0.1 
            param_init['b'] = (np.random.random_sample((1, output_dim)) - 0.5) * 0.1 
            

        self.W = tf.Variable(initial_value=param_init['W'], dtype='float32', trainable=True)
        self.b = tf.Variable(initial_value=param_init['b'], dtype='float32', trainable=True)

        self.activation = activation
        self.reg_weight = reg_weight
        

    def call(self, inputs, training=None, mask=None):
        """
        This function implement the `call` function of the class's parent `tf.keras.layers.Layer`. Please 
        consult the documentation of this function from `tf.keras.layers.Layer`.
        """

        
        # Implement the linear transformation

        outputs = tf.matmul(inputs, self.W) + self.b

        # Implement the activation function
        if self.activation == 'linear':
            pass
        elif self.activation == 'relu':
            outputs = tf.nn.relu(outputs)
        elif self.activation == 'tanh':
            outputs = tf.tanh(outputs)
        elif self.activation == 'sigmoid': 
            outputs = tf.sigmoid(outputs)
        elif self.activation == 'softmax': 
            outputs = tf.nn.softmax(outputs, axis=1)
        else:
            raise Exception('This activiation type is not implemented: ', self.activation)


        # check self.add_loss() to add the regularization term to the training objective

        self.add_loss(0.5 * self.reg_weight * tf.reduce_sum(tf.square(self.W)))

        return outputs
        

class Feedforward(tf.keras.Model):

    """
    A feedforward neural network. 
    """

    def __init__(self, input_size, depth, hidden_sizes, output_size, reg_weight, task_type):

        """
        Initialize the model. This way of specifying the model architecture is clumsy, but let's use this straightforward
        programming interface so it is easier to see the structure of the program. Later when you program with keras 
        layers, please think about how keras layers are implemented to take care of all components.  

        args:
          input_size: integer, the dimension of the input.
          depth:  integer, the depth of the neural network, or the number of connection layers. 
          hidden_sizes: list of integers. The length of the list should be one less than the depth of the neural network.
                        The first number is the number of output units of first connection layer, and so on so forth.
          output_size: integer, the number of classes. In our regression problem, please use 1. 
          reg_weight: float, The weight/strength for the regularization term.
          task_type: string, 'regression' or 'classification'. The task type. 
        """

        super(Feedforward, self).__init__()

        # Add a contition to make the program robust 
        if not (depth - len(hidden_sizes)) == 1:
            raise(Exception('The depth of the network is ', depth, ', but `hidden_sizes` has ', len(hidden_sizes), ' numbers in it.'))

         
        # install all connection layers except the last one
        input_sizes = [input_size] + hidden_sizes
        output_sizes = hidden_sizes

        self.denselayers = []

        for i in range(depth - 1):
            self.denselayers.append(DenseLayer(input_sizes[i], output_sizes[i], activation='tanh', reg_weight=reg_weight))

        # decide the last layer according to the task type
        if task_type == 'regression':
            last_activation = 'linear'
        elif task_type == 'classification':
            last_activation = 'softmax'
        else:
            raise Exception('The last activation is not specified for this task type: ', task_type)

        self.denselayers.append(DenseLayer(input_sizes[-1], output_size, activation=last_activation, reg_weight=reg_weight))


    def call(self, inputs, training=None, mask=None):
        """
        Implement the `call` function of `tf.keras.Model`. Please consult the documentation of tf.keras.Model to understand 
        this function. 
        """

        # run the neural network on the input 
        next_inputs = inputs
        for ilayer, denselayer in enumerate(self.denselayers): 
            next_inputs = denselayer(next_inputs)

        outputs = next_inputs

        return outputs


def train(x_train, y_train, x_val, y_val, depth, hidden_sizes, reg_weight, num_train_epochs, task_type):

    """
    Train this neural network using stochastic gradient descent.

    args:
      x_train: `np.array((N, D))`, training data of N instances and D features.
      y_train: `np.array((N, C))`, training labels of N instances and C fitting targets 
      x_val: `np.array((N1, D))`, validation data of N1 instances and D features.
      y_val: `np.array((N1, C))`, validation labels of N1 instances and C fitting targets 
      depth: integer, the depth of the neural network 
      hidden_sizes: list of integers. The length of the list should be one less than the depth of the neural network.
                    The first number is the number of output units of first connection layer, and so on so forth.

      reg_weight: float, the regularization strength.
      num_train_epochs: the number of training epochs.
      task_type: string, 'regression' or 'classification', the type of the learning task.
    """

    # prepare the data to make sure the data type is correct. 
    x_train = x_train.astype(np.float32)
    x_val = x_val.astype(np.float32)
    
    if task_type == 'regression':
        y_train = y_train.astype(np.float32)
        y_val = y_val.astype(np.float32)
    elif task_type == 'classification':
        y_train = y_train.astype(np.int32)
        y_val = y_val.astype(np.int32)
    else:
        raise Exception('Unknown task type: ', task_type)

    
    # Initialize a model with the Feedforward class 
    model = Feedforward(input_size=x_train.shape[1], depth=depth, hidden_sizes=hidden_sizes, output_size=y_train.shape[1], reg_weight=reg_weight, task_type=task_type)

    # Initialize an opimizer
    #sgd = tf.keras.optimizers.SGD(lr=0.01)
    sgd = tf.keras.optimizers.Adam()
    
    # decide the loss for the learning problem
    if task_type == 'regression':
        loss = tf.keras.losses.MSE 
    elif task_type == 'classification':
        loss = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM)
    else:
        raise Exception('Loss tyep of the specified task type is not specified:', task_type)

    # compile and train the model 
    model.compile(loss=loss, optimizer=sgd, metrics=["accuracy"])
    history = model.fit(x_train, y_train, epochs=num_train_epochs, batch_size=32, verbose=1, validation_data=(x_val, y_val))

    # return the model and the training history 

    return model, history


