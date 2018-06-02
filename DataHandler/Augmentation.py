import tensorflow as tf
#https://becominghuman.ai/data-augmentation-on-gpu-in-tensorflow-13d14ecf2b19
import math
def augment(images, labels,
            resize=None,  # (width, height) tuple or None
            horizontal_flip=False,
            vertical_flip=False,
            rotate=0,  # Maximum rotation angle in degrees
            crop_probability=0,  # How often we do crops
            crop_min_percent=0.6,  # Minimum linear dimension of a crop
            crop_max_percent=1.,  # Maximum linear dimension of a crop
            mixup=0):  # Mixup coeffecient, see https://arxiv.org/abs/1710.09412.pdf
    if resize is not None:
        images = tf.image.resize_bilinear(images, resize)
        labels = tf.image.resize_nearest_neighbor(labels, resize)

    # My experiments showed that casting on GPU improves training performance
    if images.dtype != tf.float32:
        images = tf.image.convert_image_dtype(images, dtype=tf.float32)
        labels = tf.image.convert_image_dtype(labels, dtype=tf.float32)

        images = tf.subtract(images, 0.5)
        images = tf.multiply(images, 2.0)

    with tf.name_scope('augmentation'):
        shp = tf.shape(images)
        batch_size, height, width = shp[0], shp[1], shp[2]
        width = tf.cast(width, tf.float32)
        height = tf.cast(height, tf.float32)

        # The list of affine transformations that our image will go under.
        # Every element is Nx8 tensor, where N is a batch size.
        transforms = []
        identity = tf.constant([1, 0, 0, 0, 1, 0, 0, 0], dtype=tf.float32)
        if horizontal_flip:
            coin = tf.less(tf.random_uniform([batch_size], 0, 1.0), 0.5)
            flip_transform = tf.convert_to_tensor(
                [-1., 0., width, 0., 1., 0., 0., 0.], dtype=tf.float32)
            transforms.append(
                tf.where(coin,
                         tf.tile(tf.expand_dims(flip_transform, 0), [batch_size, 1]),
                         tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])))

        if vertical_flip:
            coin = tf.less(tf.random_uniform([batch_size], 0, 1.0), 0.5)
            flip_transform = tf.convert_to_tensor(
                [1, 0, 0, 0, -1, height, 0, 0], dtype=tf.float32)
            transforms.append(
                tf.where(coin,
                         tf.tile(tf.expand_dims(flip_transform, 0), [batch_size, 1]),
                         tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])))

        if rotate > 0:
            angle_rad = rotate / 180 * math.pi
            angles = tf.random_uniform([batch_size], -angle_rad, angle_rad)
            transforms.append(
                tf.contrib.image.angles_to_projective_transforms(
                    angles, height, width))

        if crop_probability > 0:
            crop_pct = tf.random_uniform([batch_size], crop_min_percent,
                                         crop_max_percent)
            left = tf.random_uniform([batch_size], 0, width * (1 - crop_pct))
            top = tf.random_uniform([batch_size], 0, height * (1 - crop_pct))
            crop_transform = tf.stack([
                crop_pct,
                tf.zeros([batch_size]), top,
                tf.zeros([batch_size]), crop_pct, left,
                tf.zeros([batch_size]),
                tf.zeros([batch_size])
            ], 1)

            coin = tf.less(
                tf.random_uniform([batch_size], 0, 1.0), crop_probability)
            transforms.append(
                tf.where(coin, crop_transform,
                         tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])))

        if transforms:
            images = tf.contrib.image.transform(
                images,
                tf.contrib.image.compose_transforms(*transforms),
                interpolation='BILINEAR')  # or 'NEAREST'
            labels = tf.contrib.image.transform(
                labels,
                tf.contrib.image.compose_transforms(*transforms),
                interpolation='NEAREST')

        def cshift(values):  # Circular shift in batch dimension
            return tf.concat([values[-1:, ...], values[:-1, ...]], 0)

        labels = tf.image.convert_image_dtype(labels, dtype=tf.int64)
        mask=tf.zeros(dtype=tf.int64,shape=tf.shape(labels))
        labels=tf.greater(labels,mask)
        #labels=tf.bitwise.bitwise_or(mask,labels)

        labels=tf.cast(labels, dtype=tf.uint8)

        tf.summary.image("image",images)
        tf.summary.image("labels",tf.multiply(tf.constant(255,dtype=tf.uint8),tf.image.convert_image_dtype(labels, dtype=tf.uint8)))

        labels=tf.cast(labels, dtype=tf.int32)
        labels=tf.squeeze(labels)

    return images, labels