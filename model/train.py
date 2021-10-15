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

import os
from datetime import datetime
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_addons as tfa
import random
import math

import utils
import config
from metrics import EvidentialAccuracy


class LidarGridMapping():
    def __init__(self):
        conf = config.getConf()

        self.batch_size = conf.batch_size
        self.y_min = conf.y_min
        self.y_max = conf.y_max
        self.x_min = conf.x_min
        self.x_max = conf.x_max
        self.z_min = conf.z_min
        self.z_max = conf.z_max
        self.step_x_size = conf.step_x_size
        self.step_y_size = conf.step_y_size
        self.intensity_threshold = conf.intensity_threshold
        self.min_point_distance = conf.min_point_distance
        self.label_resize_shape = conf.label_resize_shape
        self.max_points_per_pillar = conf.max_points_per_pillar
        self.max_pillars = conf.max_pillars
        self.number_features = conf.number_features
        self.number_channels = conf.number_channels

        # load network architecture module
        architecture = utils.load_module(conf.model)

        # get max_samples_training random training samples
        n_inputs = len(conf.input_training)
        files_train_input = utils.get_files_in_folder(conf.input_training)
        files_train_label = utils.get_files_in_folder(conf.label_training)
        _, idcs = utils.sample_list(files_train_label,
                                    n_samples=conf.max_samples_training)
        files_train_input = np.take(files_train_input, idcs)
        files_train_label = np.take(files_train_label, idcs)
        self.n_training_samples = len(files_train_label)
        print(f"Found {self.n_training_samples} training samples")

        # get max_samples_validation random validation samples
        files_valid_input = utils.get_files_in_folder(conf.input_validation)
        files_valid_label = utils.get_files_in_folder(conf.label_validation)
        _, idcs = utils.sample_list(files_valid_label,
                                    n_samples=conf.max_samples_validation)
        files_valid_input = np.take(files_valid_input, idcs)
        files_valid_label = np.take(files_valid_label, idcs)
        self.n_validation_samples = len(files_valid_label)
        print(f"Found {self.n_validation_samples} validation samples")

        # build training data pipeline
        dataTrain = tf.data.Dataset.from_tensor_slices(
            (files_train_input, files_train_label))
        dataTrain = dataTrain.shuffle(buffer_size=self.n_training_samples,
                                      reshuffle_each_iteration=True)
        # yapf: disable
        dataTrain = tf.data.Dataset.range(conf.epochs).flat_map(
            lambda e: tf.data.Dataset.zip((
                dataTrain,
                tf.data.Dataset.from_tensors(e).repeat(),
                tf.data.Dataset.range(self.n_training_samples)
            ))
        )
        dataTrain = dataTrain.map(lambda samples, *counters: samples + counters)
        # yapf: enable
        cardinality = conf.epochs * self.n_training_samples
        dataTrain = dataTrain.apply(
            tf.data.experimental.assert_cardinality(cardinality))
        dataTrain = dataTrain.map(
            self.parse_sample,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataTrain = dataTrain.batch(conf.batch_size, drop_remainder=True)
        dataTrain = dataTrain.repeat(conf.epochs)
        dataTrain = dataTrain.prefetch(1)
        print("Built data pipeline for training")

        # build validation data pipeline
        dataValid = tf.data.Dataset.from_tensor_slices(
            (files_valid_input, files_valid_label))
        # yapf: disable
        dataValid = tf.data.Dataset.range(conf.epochs).flat_map(
            lambda e: tf.data.Dataset.zip((
                dataValid,
                tf.data.Dataset.from_tensors(e).repeat(),
                tf.data.Dataset.range(self.n_validation_samples)
            ))
        )
        dataValid = dataValid.map(lambda samples, *counters: samples + counters)
        # yapf: enable
        cardinality = conf.epochs * self.n_validation_samples
        dataValid = dataValid.apply(
            tf.data.experimental.assert_cardinality(cardinality))
        dataValid = dataValid.map(
            self.parse_sample,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataValid = dataValid.batch(conf.batch_size, drop_remainder=True)
        dataValid = dataValid.repeat(conf.epochs)
        dataValid = dataValid.prefetch(1)
        print("Built data pipeline for validation")

        # build model
        model = architecture.getModel(
            self.y_min, self.y_max, self.x_min, self.x_max, self.step_x_size,
            self.step_y_size, self.max_points_per_pillar, self.max_pillars,
            self.number_features, self.number_channels,
            self.label_resize_shape, conf.batch_size)
        if conf.model_weights is not None:
            model.load_weights(conf.model_weights)
        optimizer = tf.keras.optimizers.Adam(learning_rate=conf.learning_rate)
        loss = architecture.getLoss()
        metrics = [tf.keras.metrics.KLDivergence(), EvidentialAccuracy()]
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        print(f"Compiled model {os.path.basename(conf.model)}")
        model.summary()

        # create output directories
        model_output_dir = os.path.join(
            conf.output_dir,
            datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        tensorboard_dir = os.path.join(model_output_dir, "TensorBoard")
        checkpoint_dir = os.path.join(model_output_dir, "Checkpoints")
        if not os.path.exists(tensorboard_dir):
            os.makedirs(tensorboard_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # create callbacks to be called after each epoch
        tensorboard_cb = tf.keras.callbacks.TensorBoard(tensorboard_dir,
                                                        update_freq="epoch",
                                                        profile_batch=0)
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(checkpoint_dir, "e{epoch:03d}_weights.hdf5"),
            period=conf.save_interval,
            save_weights_only=True)
        best_checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(checkpoint_dir, "best_weights.hdf5"),
            save_best_only=True,
            monitor="val_loss",
            save_weights_only=True)

        class EpochCallback(tf.keras.callbacks.Callback):
            def on_epoch_begin(self, epoch, logs={}):
                K.set_value(self.model.loss.epoch_num, epoch)

        epoch_cb = EpochCallback()

        callbacks = [
            tensorboard_cb, checkpoint_cb, best_checkpoint_cb, epoch_cb
        ]

        # start training
        print("Starting training...")
        n_batches_train = len(files_train_label) // conf.batch_size
        n_batches_valid = len(files_valid_label) // conf.batch_size
        model.fit(dataTrain,
                  epochs=conf.epochs,
                  initial_epoch=conf.model_weights_epoch,
                  steps_per_epoch=n_batches_train,
                  validation_data=dataValid,
                  validation_freq=1,
                  validation_steps=n_batches_valid,
                  callbacks=callbacks)

    def augmentSample(self, pointcloud, image):
        angle = random.uniform(-math.pi, math.pi)
        pointcloud = utils.rotate_pointcloud(pointcloud, angle)
        image = tfa.image.rotate(image, angle)
        return pointcloud, image

    # build dataset pipeline parsing functions
    def parse_sample(self, input_file, label_file, epoch, sample_idx):
        def parseSampleFn(input_file, sample_idx, label_file=None):

            # convert sample index to batch element index
            batch_element_idx = sample_idx % self.batch_size

            # convert PCD file to matrix with columns (x, y, z, i)
            input_file = bytes.decode(input_file)
            lidar = utils.readPointCloud(input_file, self.intensity_threshold)

            if label_file is not None:
                # convert grid map image to matrix
                label_file = bytes.decode(label_file)
                grid_map = tf.image.decode_image(tf.io.read_file(label_file))

            # augment training sample
            if label_file is not None:
                lidar, grid_map = self.augmentSample(lidar, grid_map)
            else:
                lidar, _ = self.augmentSample(lidar, None)

            # create point pillars
            pillars, voxels = utils.make_point_pillars(
                lidar, self.max_points_per_pillar, self.max_pillars,
                self.step_x_size, self.step_y_size, self.x_min, self.x_max,
                self.y_min, self.y_max, self.z_min, self.z_max,
                min_distance = self.min_point_distance)
            pillars = pillars.astype(np.float32)
            voxels = voxels.astype(np.int32)
            voxels[..., 0] = batch_element_idx

            # convert grid map to tensorflow label
            if label_file is not None:
                grid_map = tf.image.resize(
                    grid_map,
                    self.label_resize_shape[0:2],
                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                # use only channels 'free' and 'occupied'
                grid_map = tf.cast(grid_map[..., 1:3], tf.float32)
                # normalize from image [0..255] to [0.0..1.0]
                grid_map = tf.divide(grid_map, 255.0)
            if label_file is not None:
                return pillars, voxels, grid_map
            else:
                return pillars, voxels

        # resort to numpy_function as AutoGraph is not possible
        if label_file is not None:
            pillars, voxels, grid_map = tf.numpy_function(
                func=parseSampleFn,
                inp=[input_file, sample_idx, label_file],
                Tout=[tf.float32, tf.int32, tf.float32])
        else:
            pillars, voxels = tf.numpy_function(func=parseSampleFn,
                                                inp=[input_file, sample_idx],
                                                Tout=[tf.float32, tf.int32])

        # set Tensor shapes, as tf is unable to infer rank from py_function
        # augmented lidar point is 9-dimensional
        pillars.set_shape([1, None, None, 9])
        voxels.set_shape([1, None, 3])
        if label_file is not None:
            grid_map.set_shape([None, None, 2])

        # remove batch dim from input tensors, will be added by data pipeline
        pillars = tf.squeeze(pillars, axis=0)
        voxels = tf.squeeze(voxels, axis=0)

        network_inputs = (pillars, voxels)
        if label_file is not None:
            network_labels = (grid_map)
        else:
            network_labels = None

        return network_inputs, network_labels


if __name__ == '__main__':
    i = LidarGridMapping()
