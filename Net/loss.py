from tensorflow.contrib import losses
import tensorflow as tf

def loss_func(logits,gt):
    with tf.variable_scope("loss"):
        print("loss", logits.get_shape(),gt.get_shape())
        softmax_loss=tf.losses.sparse_softmax_cross_entropy(logits=logits,labels=gt)
        tf.summary.scalar("Loss", softmax_loss)

        return softmax_loss