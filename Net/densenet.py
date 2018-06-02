import tensorflow as tf
from tensorflow.contrib import slim, layers


class FConvDenseNet():
    def __init__(self,
                 input_shape=(None, 416, 416, 3),
                 n_classes=11,
                 n_filters_first_conv=48,
                 n_pool=4,
                 growth_rate=16,
                 n_layers_per_block=(4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4),
                 dropout_p=0.2):

        self.input_shape = input_shape
        self.n_classes = n_classes
        self.n_filters_first_conv = n_filters_first_conv
        self.n_pool = n_pool
        self.growth_rate = growth_rate
        self.n_layers_per_block = n_layers_per_block
        self.dropout_p = dropout_p
        self.compression=1

    def conv(self, inp, n_filters, kernel_size=3):
        batchnorm = tf.layers.batch_normalization(inp)

        conv = slim.conv2d(inputs=batchnorm, num_outputs=n_filters, kernel_size=kernel_size, padding='SAME',
                           activation_fn=self.activation, weights_initializer=self.initializer,
                           weights_regularizer=self.regularizer, biases_initializer=self.initializer_b, scope="conv")

        if self.dropout_p > 0:
            conv = slim.dropout(conv, keep_prob=self.dropout_p)

        return conv

    def downSample(self, inputs, filters):

        conv = self.conv(inputs, n_filters=filters, kernel_size=1)
        maxpooled = slim.max_pool2d(conv, kernel_size=2, stride=2)
        return maxpooled

    def upsample(self, inp, poolhead, n_filters):
        print("inp=", inp.get_shape())
        print("poolhead=", poolhead.get_shape())
        conv = slim.convolution2d_transpose(inp, n_filters, 3, stride=2, activation_fn=self.activation,
                                            weights_initializer=self.initializer, weights_regularizer=self.regularizer,
                                            biases_initializer=self.initializer_b)
        print("transpose=", conv.get_shape())

        conv = tf.concat([conv, poolhead], axis=-1)
        return conv

    def inference(self, img):
        self.padding = 'SAME'
        # initializer = tf.truncated_normal_initializer(stddev=0.01)
        self.initializer = layers.xavier_initializer_conv2d()
        self.initializer_b = layers.xavier_initializer()
        self.regularizer = slim.l2_regularizer(0.0005)
        self.activation = slim.layers.nn.leaky_relu
        self.activation = slim.layers.nn.relu

        print(self.n_filters_first_conv)
        with tf.variable_scope("first-conv"):

            stack = slim.conv2d(img,
                                num_outputs=self.n_filters_first_conv,
                                kernel_size=[3, 3],
                                weights_initializer=self.initializer,
                                activation_fn=self.activation,
                                padding=self.padding,
                                weights_regularizer=self.regularizer,
                                biases_initializer=self.initializer_b)


        print(stack.get_shape())
        n_filters = self.n_filters_first_conv

        pool_heads = []
        for i in range(self.n_pool + 1):
            print("Dense Block ", i+1)
            with tf.variable_scope("block" + str(i + 1)):
                for j in range(self.n_layers_per_block[i]):
                    with tf.variable_scope("layer" + str(j + 1)):

                        #with tf.variable_scope("1x1conv"):
                            #conv=self.conv(stack,self.growth_rate*4,kernel_size=1)

                        with tf.variable_scope("3x3conv"):
                            conv = self.conv(stack, self.growth_rate,kernel_size=3)

                    stack = tf.concat([stack, conv], axis=-1)

                    n_filters += self.growth_rate
                    print(stack.get_shape(), " n=",n_filters)

                pool_heads.append(stack)

            if self.n_pool == i:
                continue
            with tf.variable_scope("Downsample" + str(i + 1)):
                stack = self.downSample(stack, int(n_filters*0.5))
                print(n_filters/2)

        stack = pool_heads[len(pool_heads) - 1]

        print("Turning up:", stack.get_shape())

        for i in range(self.n_pool):
            print("Dense Block ", i+self.n_pool+1)

            with tf.variable_scope("Upsample" + str(i + 1)):

                concat = pool_heads[len(pool_heads) - i - 2]
                n_filters_keep = self.growth_rate * self.n_layers_per_block[self.n_pool + i]
                print(n_filters)
                stack = self.upsample(stack, concat, int(n_filters_keep*self.compression))

            with tf.variable_scope("unpool-block" + str(i + 1)):

                for j in range(self.n_layers_per_block[self.n_pool + i + 1]):
                    with tf.variable_scope("layer" + str(j + 1)):

                        #with tf.variable_scope("1x1conv" + str(j + 1)):
                            #conv=self.conv(stack,self.growth_rate*4,kernel_size=1)
                        with tf.variable_scope("3x3conv" + str(j + 1)):
                            conv = self.conv(stack, self.growth_rate)

                    stack = tf.concat([stack, conv], axis=-1)
                    print(stack.get_shape())

        with tf.variable_scope("last-conv"):

            batchnorm = tf.layers.batch_normalization(stack)

            logits = slim.conv2d(inputs=batchnorm, num_outputs=self.n_classes+1, kernel_size=1, padding='SAME',
                               weights_initializer=self.initializer, biases_initializer=self.initializer_b,
                               activation_fn=None)
            print("logits=", logits.get_shape())

        with tf.variable_scope("softmax"):
            pred = tf.argmax(tf.nn.softmax(logits), axis=-1)
            print(pred.get_shape())

        p = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        #print(p)
        for i,a in enumerate(p):
            if(i%8)==3:
                #print(i)
                tf.summary.histogram(a.name, tf.Graph.get_tensor_by_name(tf.get_default_graph(), a.name))
        pred=tf.expand_dims(pred, axis=-1)
        print("pred==", pred.get_shape())
        tf.summary.image("pred",tf.multiply(tf.constant(255,dtype=tf.uint8),tf.cast(pred,tf.uint8)))

        # tf.summary.histogram("bias", kernel)

        return logits, pred
