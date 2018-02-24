import os
import numpy as np
from  Net.densenet import FConvDenseNet
import tensorflow as tf
from Net.loss import loss_func

def train(load, ckpt_dir, gpu, lr, ckpt_steps, batchsize, imgdir, groundtruth):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu)
    session_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

    batch_plc = tf.placeholder(tf.float32, [None, 512, 512, 3])
    gt_plc = tf.placeholder(tf.int32, [None, 512, 512])

    densenet = FConvDenseNet(n_classes=2,n_pool=5,growth_rate=12,n_layers_per_block=(4,4,4,4,4,4,4,4,4,4,4,4,4))

    logits, softmax = densenet.inference(batch_plc)

    loss_op = loss_func(logits, gt_plc)
    optimizer = tf.train.AdamOptimizer(lr)
    train_step = optimizer.minimize(loss_op)

    init = tf.global_variables_initializer()

    writer = tf.summary.FileWriter("summary/")
    mergedsummary = tf.summary.merge_all()
    saver = tf.train.Saver()

    with tf.Session(config=session_config) as sess:
        sess.run(init)
        writer.add_graph(sess.graph)

        start = 0
        if load > 0:
            print("Restoring", load, ".ckpt.....")
            saver.restore(sess, os.path.join(ckpt_dir, str(load)))
            start = load
        while True:

            img = np.ones((batchsize, 512, 512, 3), dtype=np.float32)
            labels = np.ones((batchsize, 512, 512), dtype=np.int32)

            _, loss = sess.run([train_step, loss_op], feed_dict={batch_plc: img,
                                                                 gt_plc: labels,
                                                                 })

            if start % 50 == 0:

                s = sess.run(mergedsummary, feed_dict={batch_plc: img,
                                                      gt_plc: labels,
                                                      })
                writer.add_summary(s, start)
                print("writing summary")

            print("Step <", start, "> loss => ", loss)

            if start % ckpt_steps == 0 and start != ckpt_steps:
                print("saving checkpoint ", str(start), ".ckpt.....")

                #save_path = saver.save(sess, os.path.join(ckpt_dir, str(start)))

            start += 1


if __name__ == '__main__':
    train(-1,"ckpt",0.5,0.01,1000,1,"s","s")
