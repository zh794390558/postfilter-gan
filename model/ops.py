from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

image_summary = tf.summary.image
scalar_summary = tf.summary.scalar
histogram_summary = tf.summary.histogram
merge_summary = tf.summary.merge
SummaryWriter = tf.summary.FileWriter


class batch_norm(object):
    """
    This class creates an op that composes the specified tensor with a batch
    normalization layer.
    """

    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        """Instance initialization"""
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        """
        Functional interface
        Args:
            x: tensor to compose
            train: set to True during training and False otherwise
        """
        return tf.contrib.layers.batch_norm(x,
                                            decay=self.momentum,
                                            updates_collections=None,
                                            epsilon=self.epsilon,
                                            scale=True,
                                            is_training=train,
                                            scope=self.name)


def conv_cond_concat(x, y):
    """
    Concatenate conditioning matrix across channel axis.
    The specified input tensor is concatenated with K feature maps (K = number of classes)
    across the channel dimension. Each of the K feature maps is set to all-zeros except for
    the one whose index matches the target class (which is set to all-ones).
    Args:
        x: non-conditioned tensor. Shape: [N, H, W, C]
        y: one-hot encoded conditioning matrix. Shape: [N, K]
    Returns:
        conditioned feature map. Shape: [N, H, W, C + K]
    """
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    batch_size = tf.shape(x)[0]
    return tf.concat(3, [x, y * tf.ones([batch_size, int(x_shapes[1]), int(x_shapes[2]), int(y_shapes[3])])])


def conv2d(input_, output_dim,
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):
    """
    Compose specified symbol with 2D convolution layer
    Args:
        input_: tensor to compose. Shape: [N, H, W, C]
        output_dim: number of output features maps
        k_h: kernel height
        k_w: kernel width
        d_h: horizontal stride
        d_w: vertical stride
        stddev: standard deviation of gaussian distribution to use for random weight initialization
        name: name scope
    Returns:
        Composed tensor.
    """
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape().as_list()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases)

        return conv


def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):
    """
    Compose specified symbol with 2D *transpose* convolution layer
    Args:
        input_: tensor to compose. Shape: [N, H, W, C]
        output_shape: output shape
        k_h: kernel height
        k_w: kernel width
        d_h: horizontal stride
        d_w: vertical stride
        stddev: standard deviation of gaussian distribution to use for random weight initialization
        name: name scope
    Returns:
        Composed tensor.
    """
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w',
                            [k_h, k_w, output_shape[-1],
                            input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv2d_transpose(input_, w,
                                        output_shape=output_shape,
                                        strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), output_shape)

        if with_w:
            return deconv, w, biases
        else:
            return deconv

def lrelu(x, leak=0.2, name="lrelu"):
    """Compose specified tensor with leaky Rectifier Linear Unit"""
    return tf.maximum(x, leak*x)


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    """
    Compose specified tensor with linear (fully-connected) layer
    Args:
        input_: tensor to compose. Shape: [N, M]
        output_size: number of output neurons
        scope: name scope
        stddev: standard deviation of gaussian distribution to use for random weight initialization
        name: name scope
        with_w: whether to also return parameter variables
    Returns:
      Composed tensor. Shape: [N, output_size]
    """
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias

def nhwc_to_nchw(x):
        return tf.transpose(x, [0, 3, 1, 2])

def nchw_to_nhwc(x):
        return tf.transpose(x, [0, 2, 3, 1])

def chw_to_hwc(x):
        return tf.transpose(x, [1, 2, 0])

def hwc_to_chw(x):
        return tf.transpose(x, [2, 0, 1])
