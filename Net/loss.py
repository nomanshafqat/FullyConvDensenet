from tensorflow.contrib import losses
import tensorflow as tf

def loss_func(logits,gt):
    with tf.variable_scope("loss"):
        softmax_loss=tf.losses.sparse_softmax_cross_entropy(logits=logits,labels=gt)
        tf.summary.scalar("Loss", softmax_loss)

        return softmax_loss