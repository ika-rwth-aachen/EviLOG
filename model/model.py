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
import numpy as np

import utils
from third_party.point_pillars import getPointPillarsModel


def getModel(y_min, y_max, x_min, x_max, step_x_size, step_y_size,
             max_points_per_pillar, max_pillars, number_features,
             number_channels, label_resize_shape, batch_size):

    Xn = int((x_max - x_min) / step_x_size)
    Yn = int((y_max - y_min) / step_y_size)

    # Point Pillars Feature Net
    input_pillars, input_indices, concat = getPointPillarsModel(
        tuple([Xn, Yn]), int(max_pillars), int(max_points_per_pillar),
        int(number_features), int(number_channels), batch_size)

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
        KL_reg = tf.minimum(1.0, tf.cast(self.epoch_num / 10,
                                         tf.float32)) * self.kl_regularization(
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
