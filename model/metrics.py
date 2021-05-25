#!/usr/bin/env python

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

import utils


class EvidentialAccuracy(tf.keras.metrics.Accuracy):
    """``tf.keras.metrics.Accuracy`` adapted for outputs containing unnormalized evidences for `K` classes.
    The evidences will be converted into two belief masses and an additional uncertainty mass will be introduced.
    The three belief masses will be converted to a one-hot encoded vector by selecting the maximum mass.
    This one-hot vector is used to calculate the mean accuracy.
    """
    def __init__(self, name='evidential_accuracy', *args, **kwargs):
        super().__init__(name, *args, *kwargs)

    def update_state(self, y_true, y_pred, *args, **kwargs):
        prob_true, u_true, _, _ = utils.evidences_to_masses(y_true)
        masses_true = tf.concat([prob_true, u_true], axis=-1)

        prob_pred, u_pred, _, _ = utils.evidences_to_masses(y_pred)
        masses_pred = tf.concat([prob_pred, u_pred], axis=-1)

        return super().update_state(tf.argmax(masses_true, axis=-1),
                                    tf.argmax(masses_pred, axis=-1), *args,
                                    **kwargs)
