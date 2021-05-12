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
import sys
import numpy as np
import random
import tensorflow as tf
import cv2
import xml.etree.ElementTree as xmlET
from tqdm import tqdm
import importlib
import math
from pypcd import pypcd
from point_pillars import createPillars


def abspath(path):

    return os.path.abspath(os.path.expanduser(path))


def get_files_in_folder(folder):

    return sorted([os.path.join(folder, f) for f in os.listdir(folder)])


def sample_list(*ls, n_samples, replace=False):

    n_samples = min(len(ls[0]), n_samples)
    idcs = np.random.choice(np.arange(0, len(ls[0])),
                            n_samples,
                            replace=replace)
    samples = zip([np.take(l, idcs) for l in ls])
    return samples, idcs


def load_module(module_file):

    name = os.path.splitext(os.path.basename(module_file))[0]
    dir = os.path.dirname(module_file)
    sys.path.append(dir)
    spec = importlib.util.spec_from_file_location(name, module_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def load_image(filename):

    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def load_image_op(filename):

    img = tf.io.read_file(filename)
    img = tf.image.decode_png(img, channels=3)
    return img


def resize_image(img, shape, interpolation=cv2.INTER_CUBIC):

    # resize relevant image axis to length of corresponding target axis while preserving aspect ratio
    axis = 0 if float(shape[0]) / float(img.shape[0]) > float(
        shape[1]) / float(img.shape[1]) else 1
    factor = float(shape[axis]) / float(img.shape[axis])
    img = cv2.resize(img, (0, 0),
                     fx=factor,
                     fy=factor,
                     interpolation=interpolation)

    # crop other image axis to match target shape
    center = img.shape[int(not axis)] / 2.0
    step = shape[int(not axis)] / 2.0
    left = int(center - step)
    right = int(center + step)
    if axis == 0:
        img = img[:, left:right]
    else:
        img = img[left:right, :]

    return img


def resize_image_op(img,
                    fromShape,
                    toShape,
                    cropToPreserveAspectRatio=True,
                    interpolation=tf.image.ResizeMethod.BICUBIC):

    if not cropToPreserveAspectRatio:
        img = tf.image.resize(img, toShape, method=interpolation)

    else:

        # first crop to match target aspect ratio
        fx = toShape[1] / fromShape[1]
        fy = toShape[0] / fromShape[0]
        relevantAxis = 0 if fx < fy else 1
        if relevantAxis == 0:
            crop = fromShape[0] * toShape[1] / toShape[0]
            img = tf.image.crop_to_bounding_box(img, 0,
                                                int((fromShape[1] - crop) / 2),
                                                fromShape[0], int(crop))
        else:
            crop = fromShape[1] * toShape[0] / toShape[1]
            img = tf.image.crop_to_bounding_box(img,
                                                int((fromShape[0] - crop) / 2),
                                                0, int(crop), fromShape[1])

        # then resize to target shape
        img = tf.image.resize(img, toShape, method=interpolation)

    return img


def evidences_to_masses(logits):
    # convert evidences (y_pred) to parameters of Dirichlet distribution (alpha)
    alpha = logits + tf.ones(tf.shape(logits))

    # Dirichlet strength (sum alpha for all classes)
    S = tf.reduce_sum(alpha, axis=-1, keepdims=True)

    num_classes = tf.cast(tf.shape(logits)[-1], tf.dtypes.float32)

    # uncertainty
    u = num_classes / S
    # belief masses
    prob = logits / S

    return prob, u, S, num_classes


def evidence_to_ogm(logits):
    prob, _, _, _ = evidences_to_masses(logits)

    height, width = prob.shape[0:2]
    image = np.zeros([height, width, 3], dtype=np.uint8)
    image[:, :, 1] = 255.0 * prob[:, :, 0]
    image[:, :, 0] = 255.0 * prob[:, :, 1]
    return image


def readPointCloud(file, intensity_threshold):

    pypcd_pcl = pypcd.PointCloud.from_path(file).pc_data
    x = pypcd_pcl["x"]
    y = pypcd_pcl["y"]
    z = pypcd_pcl["z"]
    i = np.clip(pypcd_pcl["intensity"] / intensity_threshold, 0.0, 1.0)
    pcl = np.array([x, y, z, i], dtype=np.float32)
    pcl = np.transpose(pcl)  # one point per row with columns (x, y, z, i)

    return pcl


def make_point_pillars(points: np.ndarray,
                       max_points_per_pillar,
                       max_pillars,
                       step_x_size,
                       step_y_size,
                       x_min,
                       x_max,
                       y_min,
                       y_max,
                       z_min,
                       z_max,
                       print_time=False):

    assert points.ndim == 2
    assert points.shape[1] == 4  # (x, y, z, i) in columns
    assert points.dtype == np.float32

    pillars, indices = createPillars(points, max_points_per_pillar,
                                     max_pillars, step_x_size, step_y_size,
                                     x_min, x_max, y_min, y_max, z_min, z_max,
                                     print_time)

    return pillars, indices


def lidar_to_bird_view_img(pointcloud: np.ndarray,
                           x_min,
                           x_max,
                           y_min,
                           y_max,
                           step_x_size,
                           step_y_size,
                           factor=1):
    # Input:
    #   pointcloud: (N, 4) with N points [x, y, z, intensity], intensity in [0,1]
    # Output:
    #   birdview: ((x_max-x_min)/step_x_size)*factor, ((y_max-y_min)/step_y_size)*factor, 3)

    size_x = int((x_max - x_min) / step_x_size)
    size_y = int((y_max - y_min) / step_y_size)
    birdview = np.zeros((size_x * factor, size_y * factor, 1), dtype=np.uint8)

    for point in pointcloud:
        x, y = point[0:2]
        # scale with minimum intensity for visibility in image
        i = 55 + point[3] * 200
        if not 0 <= i <= 255:
            raise ValueError("Intensity out of range [0,1].")
        if x_min < x < x_max and y_min < y < y_max:
            x = int((x - x_min) / step_x_size * factor)
            y = int((y - y_min) / step_y_size * factor)
            cv2.circle(birdview, ((size_y * factor - y, size_x * factor - x)),
                       radius=0,
                       color=(i))
    birdview = cv2.applyColorMap(birdview, cv2.COLORMAP_HOT)
    birdview = cv2.cvtColor(birdview, cv2.COLOR_BGR2RGB)

    return birdview


def rotate_pointcloud(pointcloud, angle):
    rotation = np.array([[math.cos(angle), -math.sin(angle)],
                         [math.sin(angle), math.cos(angle)]])
    # Lidar has the shape [N, 4] with 4 being x, y, z, intensity. Apply the rotation matrix to x
    # and y only. Transpose and transpose back the matrix in order to apply the rotation to all
    # points in one large operation.
    pointcloud[:, 0:2] = (rotation @ pointcloud[:, 0:2].T).T

    return pointcloud


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
