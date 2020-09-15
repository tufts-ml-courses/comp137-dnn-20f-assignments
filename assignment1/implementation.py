import tensorflow as tf

"""
This is a short tutorial of tensorflow. After this tutorial, you should know the following concepts:
1. constant,
2. operations
3. variables 
4. gradient calculation 
5. optimizer 
"""

def regression_func(x, w, b):
    """
    The function of a linear regression model
    args: 
        x: tf.Tensor with shape (n, d) 
        w: tf.Variable with shape (d,) 
        b: tf.Variable with shape ()

    return: 
        y_hat: tf.Tensor with shape [n,]. y_hat = x * w + b (matrix multiplication)
    """
    

    # TODO: implement this function
    # consider these functions: `tf.matmul`, `tf.einsum`, `tf.squeeze` 

    return y_hat



def loss_func(y, y_hat):
    """
    The loss function for linear regression

    args:
        y: tf.Tensor with shape (n,) 
        y_hat: tf.Tensor with shape (n,) 

    return:
        loss: tf.Tensor with shape (). loss = (y -  y_hat)^\top (y -  y_hat) 

    """

    # TODO: implement the function. 
    # Consider these functions: `tf.square`, `tf.reduce_sum`

    return loss



def train_lr(x, y, lamb):
    """
    Train a linear regression model.

    args:
        x: tf.Tensor with shape (n, d)
        y: tf.Tensor with shape (n, )
        lamb: tf.Tensor with shape ()
    """
    
    # TODO: implement the function.
    # initialize parameters w and b


    # set an optimizer
    # please check the documentation of tf.keras.optimizers.SGD

    # loop to optimize w and b 


    return w, b

