# ==============================================================================
# MIT License
#
# Copyright 2021 Institute for Automotive Engineering of RWTH Aachen University.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================

import tensorflow as tf
import math
import os
import numpy as np
import matplotlib.pyplot as plt
from pypcd import pypcd
import cv2
import json

import utils


def getModel(y_min, y_max, x_min, x_max, step_x_size, step_y_size,
             max_points_per_pillar, max_pillars, number_features,
             number_channels, label_resize_shape, batch_size):

    Xn = int((x_max - x_min) / step_x_size)
    Yn = int((y_max - y_min) / step_y_size)

    # extract required parameters
    max_pillars = int(max_pillars)
    max_points = int(max_points_per_pillar)
    nb_features = int(number_features)
    nb_channels = int(number_channels)
    image_size = tuple([Xn, Yn])

    if tf.keras.backend.image_data_format() == "channels_first":
        raise NotImplementedError
    else:
        input_shape = (max_pillars, max_points, nb_features)

    input_pillars = tf.keras.layers.Input(input_shape,
                                          batch_size=batch_size,
                                          name="pillars/input")
    input_indices = tf.keras.layers.Input((max_pillars, 3),
                                          batch_size=batch_size,
                                          name="pillars/indices",
                                          dtype=tf.int32)

    # Pillar Feature Net
    x = tf.keras.layers.Conv2D(nb_channels, (1, 1),
                               activation='linear',
                               use_bias=False,
                               name="pillars/conv2d")(input_pillars)
    x = tf.keras.layers.BatchNormalization(name="pillars/batchnorm",
                                           fused=True,
                                           epsilon=1e-3,
                                           momentum=0.99)(x)
    x = tf.keras.layers.Activation("relu", name="pillars/relu")(x)
    x = tf.keras.layers.MaxPool2D((1, max_points),
                                  name="pillars/maxpooling2d")(x)

    if tf.keras.backend.image_data_format() == "channels_first":
        reshape_shape = (nb_channels, max_pillars)
    else:
        reshape_shape = (max_pillars, nb_channels)

    x = tf.keras.layers.Reshape(reshape_shape, name="pillars/reshape")(x)

    # pillars = tf.keras.layers.Lambda(
    #     lambda inp: tf.scatter_nd(inp[0], inp[1],
    #                               (self.batch_size,) + image_size +
    #                               (nb_channels,)),
    #     name="pillars/scatter_nd")([input_indices, x])
    pillars = tf.scatter_nd(input_indices, x,
                            (batch_size, ) + image_size + (nb_channels, ))

    # reverse dimensions (x,y) to match OGM coordinates (size_x-x, size_y-y)
    pillars = tf.reverse(pillars, [1, 2])

    # 2D CNN backbone

    # Block1(S, 4, C)
    x = pillars
    for n in range(4):
        S = (2, 2) if n == 0 else (1, 1)
        x = tf.keras.layers.Conv2D(nb_channels, (3, 3),
                                   strides=S,
                                   padding="same",
                                   activation="relu",
                                   name="cnn/block1/conv2d%i" % n)(x)
        x = tf.keras.layers.BatchNormalization(name="cnn/block1/bn%i" % n,
                                               fused=True)(x)
    x1 = x

    # Block2(2S, 6, 2C)
    for n in range(6):
        S = (2, 2) if n == 0 else (1, 1)
        x = tf.keras.layers.Conv2D(2 * nb_channels, (3, 3),
                                   strides=S,
                                   padding="same",
                                   activation="relu",
                                   name="cnn/block2/conv2d%i" % n)(x)
        x = tf.keras.layers.BatchNormalization(name="cnn/block2/bn%i" % n,
                                               fused=True)(x)
    x2 = x

    # Block3(4S, 6, 4C)
    for n in range(6):
        S = (2, 2) if n == 0 else (1, 1)
        x = tf.keras.layers.Conv2D(4 * nb_channels, (3, 3),
                                   strides=S,
                                   padding="same",
                                   activation="relu",
                                   name="cnn/block3/conv2d%i" % n)(x)
        x = tf.keras.layers.BatchNormalization(name="cnn/block3/bn%i" % n,
                                               fused=True)(x)
    x3 = x

    # Up1 (S, S, 2C)
    up1 = tf.keras.layers.Conv2DTranspose(2 * nb_channels, (3, 3),
                                          strides=(1, 1),
                                          padding="same",
                                          activation="relu",
                                          name="cnn/up1/conv2dt")(x1)
    up1 = tf.keras.layers.BatchNormalization(name="cnn/up1/bn",
                                             fused=True)(up1)

    # Up2 (2S, S, 2C)
    up2 = tf.keras.layers.Conv2DTranspose(2 * nb_channels, (3, 3),
                                          strides=(2, 2),
                                          padding="same",
                                          activation="relu",
                                          name="cnn/up2/conv2dt")(x2)
    up2 = tf.keras.layers.BatchNormalization(name="cnn/up2/bn",
                                             fused=True)(up2)

    # Up3 (4S, S, 2C)
    up3 = tf.keras.layers.Conv2DTranspose(2 * nb_channels, (3, 3),
                                          strides=(4, 4),
                                          padding="same",
                                          activation="relu",
                                          name="cnn/up3/conv2dt")(x3)
    up3 = tf.keras.layers.BatchNormalization(name="cnn/up3/bn",
                                             fused=True)(up3)

    # Concat
    concat = tf.keras.layers.Concatenate(name="cnn/concatenate")(
        [up1, up2, up3])

    # Evidential Prediction Head
    prediction = tf.keras.layers.Conv2D(2, (3, 3),
                                        padding="same",
                                        name="ogm/conv2d",
                                        activation="relu")(concat)

    return tf.keras.models.Model([input_pillars, input_indices], [prediction])


def getLoss():

    return ExpectedMeanSquaredError()

class ExpectedMeanSquaredError(tf.keras.losses.Loss):
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.epoch_num = tf.Variable(0.0)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor):

        prob, _, S, num_evidential_classes = utils.evidences_to_masses(y_pred)

        loss = tf.math.add(
            tf.reduce_sum((y_true - prob)**2, axis=-1, keepdims=True),
            tf.reduce_sum(prob * (1 - prob) / (S + 1), axis=-1, keepdims=True))
        alpha = y_pred * (1 - y_true) + 1
        KL_reg = tf.minimum(1.0, tf.cast(
            self.epoch_num / 10, tf.float32)) * self.kl_regularization(
                alpha, num_evidential_classes)
        loss = loss + KL_reg

        # higher weight for loss on evidence for state "occupied" because it is underrepresented in training data
        weight_occupied = 100
        loss = tf.where(y_true[..., 1] > 0.5,
                        tf.squeeze(loss * weight_occupied, axis=-1),
                        tf.squeeze(loss, axis=-1))
        loss = tf.reduce_mean(loss)

        return loss

    def kl_regularization(self, alpha, K):
        beta = tf.ones_like(alpha)
        S_alpha = tf.reduce_sum(alpha, axis=-1, keepdims=True)
        KL = tf.math.add_n([
            tf.reduce_sum((alpha - beta) *
                          (tf.math.digamma(alpha) - tf.math.digamma(S_alpha)),
                          axis=-1,
                          keepdims=True),
            tf.math.lgamma(S_alpha) -
            tf.reduce_sum(tf.math.lgamma(alpha), axis=-1, keepdims=True),
            tf.reduce_sum(tf.math.lgamma(beta), axis=-1, keepdims=True) -
            tf.math.lgamma(tf.reduce_sum(beta, axis=-1, keepdims=True))
        ])
        return KL


def naive_geometric_ISM(pcd_file_path,
                        x_min,
                        x_max,
                        y_min,
                        y_max,
                        step_size_x,
                        step_size_y,
                        z_min_obstacle=-1.0,
                        z_max_obstacle=0.5):
    pypcd_pcl = pypcd.PointCloud.from_path(pcd_file_path).pc_data
    x = pypcd_pcl["x"]
    y = pypcd_pcl["y"]
    z = pypcd_pcl["z"]
    pcl = np.array([x, y, z], dtype=np.float32)
    pcl = np.transpose(pcl)  # one point per row with columns (x, y, z, i)

    # create image representing naive OGM using a simple geometric inverse sensor model
    cells_x = int((x_max - x_min) / step_size_x)
    cells_y = int((y_max - y_min) / step_size_y)
    center_x = int(-x_min / step_size_x)
    center_y = int(-y_min / step_size_y)
    naive_ogm = np.zeros((cells_x, cells_y, 3), dtype=np.uint8)
    for point in pcl:
        x, y, z = point[0:3]

        if z_min_obstacle < z < z_max_obstacle and x_min < x < x_max and y_min < y < y_max:
            x = int((x - x_min) / step_size_x)
            y = int((y - y_min) / step_size_y)
            cv2.line(naive_ogm, (cells_y - y, cells_x - x),
                     (cells_y - center_y, cells_x - center_x), (0, 255, 0),
                     thickness=1)
            cv2.circle(naive_ogm, ((cells_y - y, cells_x - x)),
                       radius=0,
                       color=(0, 0, 255))

    return naive_ogm
