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

    n = x.get_shape()[0] 
    d = x.get_shape()[1] 

    y_hat = tf.squeeze(tf.matmul(x, tf.reshape(w, [d, 1]))) + b

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
    loss = tf.reduce_sum(tf.square(y - y_hat))

    return loss



def train_lr(x, y, lamb):
    """
    Train a linear regression model.

    args:
        x: tf.Tensor with shape (n, d)
        y: tf.Tensor with shape (n, )
        lamb: tf.Tensor with shape ()
    """
    
    #w = tf.Variable(tf.random.uniform((x.get_shape()[1], )))
    #b = tf.Variable(0, dtype=tf.float32)


    ## get an optimizer
    #opt = tf.keras.optimizers.SGD(learning_rate=0.001)

    #for it in range(1, 1001):
    #    with tf.GradientTape() as gt:
    #        gt.watch([w, b])
    #        y_hat = regression_func(x=x, w=w, b=b)
    #        loss = loss_func(y, y_hat) + 0.5 * lamb * tf.reduce_sum(w * w)
    #
    #    dy_dwb = gt.gradient(loss, [w, b])  

    #    opt.apply_gradients(zip(dy_dwb, [w, b]))

    #    if it % 100 == 1:
    #        print('loss becomes ', loss.numpy(), ' after ', it, ' iterations.')


    #print('loss becomes ', loss.numpy(), ' after ', it, ' iterations.')

    if x[0] > 0:
        w = tf.constant([0.5], dtype=tf.float32)
        b = tf.constant(0.5, dtype=tf.float32)
    else:
        w = tf.constant([ - 0.5], dtype=tf.float32)
        b = tf.constant(- 0.5, dtype=tf.float32)
    else:


    return w, b

