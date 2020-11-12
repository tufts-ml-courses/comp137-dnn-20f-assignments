# You are supposed to put your own RNN model here

import tensorflow as tf

def masked_lm_loss(labels, logits):

    # get flags for actual chars. Padding chars are -1
    flag_chars = tf.cast(labels >= 0, tf.float32)

    # turn labels to one-hot vectors. -1 will generate a zero vector
    labels_onehot = tf.one_hot(labels, depth=logits.shape[-1])

    # compute CE loss against logits
    loss_mat = tf.keras.losses.categorical_crossentropy(labels_onehot, logits, from_logits=True)

    # get the per-char-loss for each sentence
    loss_val = tf.reduce_sum(loss_mat * flag_chars, axis=1) / tf.reduce_sum(flag_chars, axis=1)

    return loss_val


