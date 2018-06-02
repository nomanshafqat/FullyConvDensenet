import os

import numpy as np
import tensorflow as tf

from DataHandler.Data_handler import Datahandler_COCO
from Net.densenet import FConvDenseNet
from Net.loss import loss_func
from DataHandler.Augmentation import augment

def train(load, ckpt_dir, gpu, lr, ckpt_steps, batchsize, imgdir, groundtruth):
    print("start")
    dataHandler= Datahandler_COCO(imgdir,groundtruth)
    data_generator=dataHandler.make_batches(batchsize)
    #img, labels = dataHandler.get_batch(batch_size=batchsize)

    #dataHandler= Data_handler(data_location=imgdir, ground_truth=groundtruth)
    #img, labels = dataHandler.get_batch(batch_size=batchsize)

    input_img_size=224

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu)
    session_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

    batch_plc = tf.placeholder(tf.float32, [None, input_img_size, input_img_size, 3])
    gt_plc = tf.placeholder(tf.int32, [None, input_img_size, input_img_size,1])

    batch_plc_aug, gt_plc_aug = augment(batch_plc, gt_plc,
                             horizontal_flip=True, rotate=15, crop_probability=0.8, mixup=4)

    densenet = FConvDenseNet(n_classes=1,n_pool=5,growth_rate=16)

    logits, softmax = densenet.inference(batch_plc_aug)

    loss_op = loss_func(logits, gt_plc_aug)
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

            #img = np.ones((batchsize, input_img_size, input_img_size, 3), dtype=np.float32)
            #labels = np.ones((batchsize, input_img_size, input_img_size), dtype=np.int32)

            img,labels=next(data_generator)
            #print(labels)
            _, loss = sess.run([train_step, loss_op], feed_dict={batch_plc: img,
                                                                 gt_plc: labels,
                                                                 })

            if start % 3 == 0:

                s = sess.run(mergedsummary, feed_dict={batch_plc: img,
                                                      gt_plc: labels,
                                                      })
                writer.add_summary(s, start)

                print("writing summary")

            print("Step <", start, "> loss => ", loss)
            start += 1

            if start % ckpt_steps == 0 and start != ckpt_steps:
                print("saving checkpoint ", str(start), ".ckpt.....")

                save_path = saver.save(sess, os.path.join(ckpt_dir, str(start)))




import sys

if __name__ == '__main__':
    load = int(sys.argv[1])
    ckpt_dir = sys.argv[2]
    gpu = float(sys.argv[3])
    lr = float(sys.argv[4])
    ckpt_steps = int(sys.argv[5])
    batchsize = int(sys.argv[6])
    imgdir = sys.argv[7]
    groundtruth = sys.argv[8]

    assert (os.path.exists(ckpt_dir))
    assert (os.path.exists(imgdir))
    assert (os.path.exists(groundtruth))

    train(load,ckpt_dir,gpu,lr,ckpt_steps,batchsize,imgdir,groundtruth)
