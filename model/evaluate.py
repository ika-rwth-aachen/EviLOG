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
import matplotlib.pyplot as plt
import tensorflow as tf
import json

import utils
import config

conf = config.getConf()

# load network architecture module
architecture = utils.load_module(conf.model)

# get max_samples_validation random validation samples
files_input = utils.get_files_in_folder(conf.input_validation)
files_label = utils.get_files_in_folder(conf.label_validation)
_, idcs = utils.sample_list(files_label, n_samples=conf.max_samples_validation)
files_input = np.take(files_input, idcs)
files_label = np.take(files_label, idcs)
n_samples = len(files_label)
print(f"Found {n_samples} samples")

model = architecture.getModel(conf.y_min, conf.y_max, conf.x_min, conf.x_max,
                              conf.step_x_size, conf.step_y_size,
                              conf.max_points_per_pillar, conf.max_pillars,
                              conf.number_features, conf.number_channels,
                              conf.label_resize_shape, conf.batch_size)
model.load_weights(conf.model_weights)
print(f"Reloaded model from {conf.model_weights}")

# evaluate
print("Evaluating ...")
eval_dir = os.path.join(os.path.dirname(conf.model_weights), os.pardir,
                        "Evaluation")

# evaluation metrics
evaluation_dict = {}
evaluation_dict['deep'] = {}
evaluation_dict['deep']['KL_distance'] = []
evaluation_dict['deep']['m_unknown'] = []
evaluation_dict['deep']['m_occupied'] = []
evaluation_dict['deep']['m_free'] = []
evaluation_dict['naive'] = {}
evaluation_dict['naive']['KL_distance'] = []
evaluation_dict['naive']['m_unknown'] = []
evaluation_dict['naive']['m_occupied'] = []
evaluation_dict['naive']['m_free'] = []


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
        conf.z_min, conf.z_max)
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


for k in tqdm.tqdm(range(n_samples)):

    input_file = files_input[k]
    label_file = files_label[k]

    input, label = parseSampleFn(input_file, 0, label_file)
    prediction = model.predict(input).squeeze()

    sample_name = os.path.splitext(os.path.basename(input_file))[0]

    kld = tf.keras.metrics.KLDivergence()

    # collect belief masses and Kullback-Leibler distance for predictions by deep ISM
    prob, u, _, _ = utils.evidences_to_masses(prediction)
    evaluation_dict['deep']['m_unknown'].append(float(tf.reduce_mean(u)))
    evaluation_dict['deep']['m_free'].append(
        float(tf.reduce_mean(prob[..., 0])))
    evaluation_dict['deep']['m_occupied'].append(
        float(tf.reduce_mean(prob[..., 1])))
    evaluation_dict['deep']['KL_distance'].append(float(kld(label,
                                                            prediction)))

    # create "naive" occupancy grid map for comparision
    naive_ogm = utils.naive_geometric_ISM(input_file, conf.x_min, conf.x_max,
                                          conf.y_min, conf.y_max,
                                          conf.step_x_size, conf.step_y_size,
                                          -1.11, 0.39)
    naive_ogm = cv2.resize(
        naive_ogm, (conf.label_resize_shape[1], conf.label_resize_shape[0]))
    naive_ogm_dir = os.path.join(eval_dir, "naive_ogm")
    if not os.path.exists(naive_ogm_dir):
        os.makedirs(naive_ogm_dir)
    naive_ogm_file = os.path.join(naive_ogm_dir, sample_name + '.png')
    cv2.imwrite(naive_ogm_file, naive_ogm)

    # collect belief masses and Kullback-Leibler distance for OGMs by geometric ISM
    naive_ogm = naive_ogm.astype(
        np.float32
    )[..., 1:
      3] / 255.0  # convert green and red layer to evidence layers for free and occupied
    prob_naive, u_naive, _, _ = utils.evidences_to_masses(naive_ogm)
    evaluation_dict['naive']['m_unknown'].append(float(
        tf.reduce_mean(u_naive)))
    evaluation_dict['naive']['m_free'].append(
        float(tf.reduce_mean(prob_naive[..., 0])))
    evaluation_dict['naive']['m_occupied'].append(
        float(tf.reduce_mean(prob_naive[..., 1])))
    evaluation_dict['naive']['KL_distance'].append(float(kld(label,
                                                             naive_ogm)))

# create subfolders
plot_dir = os.path.join(eval_dir, "plots")
raw_dir = os.path.join(eval_dir, "raw")
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
if not os.path.exists(raw_dir):
    os.makedirs(raw_dir)

# plot cross entropy over evaluation dataset
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('time in seconds')

# Turn off axis lines and ticks of the big subplot
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')
ax.tick_params(labelcolor='w',
               top=False,
               bottom=False,
               left=False,
               right=False)

ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

t = np.arange(0, len(evaluation_dict['naive']['m_unknown']))
ax1.plot(t, evaluation_dict['deep']['m_unknown'], 'b-', t,
         evaluation_dict['deep']['m_free'], 'g-', t,
         evaluation_dict['deep']['m_occupied'], 'r-', t,
         evaluation_dict['naive']['m_unknown'], 'b--', t,
         evaluation_dict['naive']['m_free'], 'g--', t,
         evaluation_dict['naive']['m_occupied'], 'r--')
ax1.set_ylim(0, 1.0)
ax1.legend([
    r'$\overline{m}(\Theta)$', r'$\overline{m}(F)$', r'$\overline{m}(O)$',
    r'$\overline{m}_G(\Theta)$', r'$\overline{m}_G(F)$', r'$\overline{m}_G(O)$'
])

ax2.plot(t, evaluation_dict['deep']['KL_distance'], 'k-', t,
         evaluation_dict['naive']['KL_distance'], 'k--')
ax2.legend([
    r'$KL\left[Dir(p|\hat{\alpha})||Dir(p|\alpha)\right]$',
    r'$KL\left[Dir(p|\hat{\alpha}_G)||Dir(p|\alpha)\right]$'
])

plt.savefig(os.path.join(plot_dir, 'evaluation.png'))

# store values as json file
evaluation_json = dict()
evaluation_json['eval_kld'] = np.vstack(
    (t, evaluation_dict['deep']['KL_distance'])).transpose().tolist()
evaluation_json['eval_uncertainty'] = np.vstack(
    (t, evaluation_dict['deep']['m_unknown'])).transpose().tolist()
evaluation_json['eval_prob_free'] = np.vstack(
    (t, evaluation_dict['deep']['m_free'])).transpose().tolist()
evaluation_json['eval_prob_occupied'] = np.vstack(
    (t, evaluation_dict['deep']['m_occupied'])).transpose().tolist()

evaluation_json['eval_naive_kld'] = np.vstack(
    (t, evaluation_dict['naive']['KL_distance'])).transpose().tolist()
evaluation_json['eval_naive_uncertainty'] = np.vstack(
    (t, evaluation_dict['naive']['m_unknown'])).transpose().tolist()
evaluation_json['eval_naive_prob_free'] = np.vstack(
    (t, evaluation_dict['naive']['m_free'])).transpose().tolist()
evaluation_json['eval_naive_prob_occupied'] = np.vstack(
    (t, evaluation_dict['naive']['m_occupied'])).transpose().tolist()
with open(os.path.join(raw_dir, 'evaluation.json'), 'w') as fp:
    json.dump(evaluation_json, fp)
