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
import tqdm
import numpy as np
import cv2
import tensorflow as tf

import utils
import config

conf = config.getConf()

# load network architecture module
architecture = utils.load_module(conf.model)

# get max_samples_testing samples
files_input = utils.get_files_in_folder(conf.input_testing)
_, idcs = utils.sample_list(files_input, n_samples=conf.max_samples_testing)
files_input = np.take(files_input, idcs)
n_samples = len(files_input)
print(f"Found {n_samples} samples")

# build model
model = architecture.getModel(conf.y_min, conf.y_max, conf.x_min, conf.x_max,
                              conf.step_x_size, conf.step_y_size,
                              conf.max_points_per_pillar, conf.max_pillars,
                              conf.number_features, conf.number_channels,
                              conf.label_resize_shape, 1)
model.load_weights(conf.model_weights)
print(f"Reloaded model from {conf.model_weights}")


# build data parsing function
def parseSampleFn(input_file, sample_idx, label_file=None):

    # convert sample index to batch element index
    batch_element_idx = sample_idx % conf.batch_size

    # convert PCD file to matrix with columns (x, y, z, i)
    #input_file = bytes.decode(input_file)
    lidar = utils.readPointCloud(input_file, conf.intensity_threshold)

    if label_file is not None:
        # convert grid map image to matrix
        #label_file = bytes.decode(label_file)
        grid_map = tf.image.decode_image(tf.io.read_file(label_file))

    # create point pillars
    pillars, voxels = utils.make_point_pillars(
        lidar, conf.max_points_per_pillar, conf.max_pillars, conf.step_x_size,
        conf.step_y_size, conf.x_min, conf.x_max, conf.y_min, conf.y_max,
        conf.z_min, conf.z_max, min_distance = conf.min_point_distance)
    pillars = pillars.astype(np.float32)
    voxels = voxels.astype(np.int32)
    voxels[..., 0] = batch_element_idx

    # convert grid map to tensorflow label
    if label_file is not None:
        grid_map = tf.image.resize(
            grid_map,
            conf.label_resize_shape[0:2],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # use only channels 'free' and 'occupied'
        grid_map = tf.cast(grid_map[..., 1:3], tf.float32)
        # normalize from image [0..255] to [0.0..1.0]
        grid_map = tf.divide(grid_map, 255.0)

    network_inputs = (pillars, voxels)
    if label_file is not None:
        network_labels = (grid_map)
    else:
        network_labels = None

    return network_inputs, network_labels


# create output directory
if not os.path.exists(conf.prediction_dir):
    os.makedirs(conf.prediction_dir)

# run predictions
print(f"Running predictions and writing to {conf.prediction_dir} ...")
for k in tqdm.tqdm(range(n_samples)):

    input_file = files_input[k]

    # load sample
    input, label = parseSampleFn(input_file, 0)
    prediction = model.predict(input).squeeze()

    # convert to output image
    prediction_img = utils.evidence_to_ogm(prediction)

    output_file = os.path.join(conf.prediction_dir,
                               os.path.basename(files_input[k]))
    cv2.imwrite(output_file + ".png",
                cv2.cvtColor(prediction_img, cv2.COLOR_RGB2BGR))

    # save input point cloud as image
    if conf.store_input_point_clouds:
        input_dir = os.path.join(conf.prediction_dir, "inputs")
        if not os.path.exists(input_dir):
            os.makedirs(input_dir)
        lidar = utils.readPointCloud(input_file, conf.intensity_threshold)
        lidar_bev = utils.lidar_to_bird_view_img(lidar,
                                                 conf.x_min,
                                                 conf.x_max,
                                                 conf.y_min,
                                                 conf.y_max,
                                                 conf.step_x_size,
                                                 conf.step_y_size,
                                                 factor=2)
        output_file = os.path.join(input_dir, os.path.basename(files_input[k]))
        cv2.imwrite(output_file + ".png",
                    cv2.cvtColor(lidar_bev, cv2.COLOR_RGB2BGR))

    # create "naive" occupancy grid map for comparision
    if conf.store_naive_ogms:
        naive_ogm = utils.naive_geometric_ISM(input_file, conf.x_min, conf.x_max,
                                            conf.y_min, conf.y_max,
                                            conf.step_x_size, conf.step_y_size,
                                            -1.11, 0.39, conf.min_point_distance)
        naive_ogm = cv2.resize(
            naive_ogm, (conf.label_resize_shape[1], conf.label_resize_shape[0]))
        naive_ogm_dir = os.path.join(conf.prediction_dir, "naive_ogm")
        if not os.path.exists(naive_ogm_dir):
            os.makedirs(naive_ogm_dir)
        naive_ogm_file = os.path.join(naive_ogm_dir,
                                    os.path.basename(files_input[k]) + '.png')
        cv2.imwrite(naive_ogm_file, naive_ogm)
