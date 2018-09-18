import os

import numpy as np
import tensorflow as tf
import cv2
from DataHandler.Data_handler import Datahandler_COCO
from Net.densenet import FConvDenseNet
from Net.loss import loss_func
from DataHandler.Augmentation import augment


def test(load, ckpt_dir, gpu, lr, ckpt_steps, batchsize, imgdir, groundtruth):
    input_img_size = 224

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu)
    session_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

    batch_plc = tf.placeholder(tf.float32, [None, input_img_size, input_img_size, 3])
    gt_plc = tf.placeholder(tf.int32, [None, input_img_size, input_img_size, 1])



    densenet = FConvDenseNet(n_classes=1, n_pool=5, growth_rate=16)

    logits, softmax = densenet.inference(batch_plc)



    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    with tf.Session(config=session_config) as sess:
        sess.run(init)

        start = 0
        if load > 0:
            print("Restoring", load, ".ckpt.....")
            saver.restore(sess, os.path.join(ckpt_dir, str(load)))
            start = load
        images=os.listdir(imgdir)
        for image in images:
            print(image)
            image=cv2.imread(os.path.join(imgdir,image))

            image=cv2.resize(image,(224,224))
            image=np.expand_dims(np.array(image),axis=0)

            #img = np.ones((batchsize, input_img_size, input_img_size, 3), dtype=np.float32)
            #labels = np.ones((batchsize, input_img_size, input_img_size), dtype=np.int32)
            #print(labels)
            results = sess.run(softmax, feed_dict={batch_plc: image,

                                                                 })
            print(results.shape)
            print(results[0].shape)

            cv2.imwrite("results/"+str(start)+"pred.jpg",results[0]*100)
            cv2.imwrite("results/"+str(start)+".jpg",image[0])


            start += 1


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

    test(load,ckpt_dir,gpu,lr,ckpt_steps,batchsize,imgdir,groundtruth)
