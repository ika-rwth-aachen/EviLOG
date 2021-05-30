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

import configargparse

import utils


def getConf():
    # parse parameters from config file or CLI
    parser = configargparse.ArgParser()
    parser.add("-c", "--config", is_config_file=True, help="config file")
    parser.add("-it",
               "--input-training",
               type=str,
               required=True,
               help="directory of input samples for training")
    parser.add("-lt",
               "--label-training",
               type=str,
               required=True,
               help="directory of label samples for training")
    parser.add("-nt",
               "--max-samples-training",
               type=int,
               default=None,
               help="maximum number of training samples")
    parser.add("-iv",
               "--input-validation",
               type=str,
               required=True,
               help="directory/directories of input samples for validation")
    parser.add("-lv",
               "--label-validation",
               type=str,
               required=True,
               help="directory of label samples for validation")
    parser.add("-nv",
               "--max-samples-validation",
               type=int,
               default=None,
               help="maximum number of validation samples")
    parser.add("-ip",
               "--input-testing",
               type=str,
               required=True,
               help="directory/directories of input samples for testing")
    parser.add("-np",
               "--max-samples-testing",
               type=int,
               default=None,
               help="maximum number of testing samples")
    parser.add("-pd",
               "--prediction-dir",
               type=str,
               default="./output/Predictions",
               help="output directory for storing predictions of testing data")
    parser.add("-m",
               "--model",
               type=str,
               required=True,
               help="Python file defining the neural network")
    parser.add("-e",
               "--epochs",
               type=int,
               required=True,
               help="number of epochs for training")
    parser.add("-bs",
               "--batch-size",
               type=int,
               default=1,
               help="batch size for training")
    parser.add("-lr",
               "--learning-rate",
               type=float,
               default=1e-4,
               help="learning rate of Adam optimizer for training")
    parser.add("-si",
               "--save-interval",
               type=int,
               default=5,
               help="epoch interval between exports of the model")
    parser.add("-o",
               "--output-dir",
               type=str,
               required=True,
               help="output dir for TensorBoard and models")
    parser.add("-mw",
               "--model-weights",
               type=str,
               default=None,
               help="weights file of trained model for training continuation")
    parser.add("-mwe",
               "--model-weights-epoch",
               type=int,
               default=0,
               help="epoch of weights file")
    parser.add("-sip",
               "--store-input-point-clouds",
               type=bool,
               default=True,
               help="store input point clouds as images during evaluation or prediction")
    parser.add("-sl",
               "--store-labels",
               type=bool,
               default=True,
               help="store ground truth grid maps as images during evaluation")
    parser.add("-sno",
               "--store-naive-ogms",
               type=bool,
               default=True,
               help="store grid maps created by a simple inverse sensor model as images during evaluation or prediction")
    parser.add("-xmi",
               "--x-min",
               type=float,
               default=None,
               help="minimum x coordinate in point cloud")
    parser.add("-xma",
               "--x-max",
               type=float,
               default=None,
               help="maximum x coordinate in point cloud")
    parser.add("-ymi",
               "--y-min",
               type=float,
               default=None,
               help="minimum y coordinate in point cloud")
    parser.add("-yma",
               "--y-max",
               type=float,
               default=None,
               help="maximum y coordinate in point cloud")
    parser.add("-zmi",
               "--z-min",
               type=float,
               default=None,
               help="minimum z coordinate in point cloud")
    parser.add("-zma",
               "--z-max",
               type=float,
               default=None,
               help="maximum z coordinate in point cloud")
    parser.add("-sx",
               "--step-x-size",
               type=float,
               default=None,
               help="step size in x direction")
    parser.add("-sy",
               "--step-y-size",
               type=float,
               default=None,
               help="step size in y direction")
    parser.add("-mpd",
               "--min-point-distance",
               type=float,
               default=None,
               help="minimum distance for point")
    parser.add("-ith",
               "--intensity-threshold",
               type=int,
               default=None,
               help="threshold for point intensity")
    parser.add("-lrx",
               "--label-resize-x",
               type=int,
               default=None,
               help="grid map label size in x direction")
    parser.add("-lry",
               "--label-resize-y",
               type=int,
               default=None,
               help="grid map label size in y direction")
    parser.add("-mpp",
               "--max-points-per-pillar",
               type=int,
               default=None,
               help="maximum number of points in one pillar")
    parser.add("-mp",
               "--max-pillars",
               type=int,
               default=None,
               help="maximum number of pillars")
    parser.add("-nf",
               "--number-features",
               type=int,
               default=None,
               help="number of features")
    parser.add("-nc",
               "--number-channels",
               type=int,
               default=None,
               help="number of channels")
    conf, unknown = parser.parse_known_args()

    conf.label_resize_shape = [conf.label_resize_x, conf.label_resize_y]

    # determine absolute filepaths
    conf.input_training = utils.abspath(conf.input_training)
    conf.label_training = utils.abspath(conf.label_training)
    conf.input_validation = utils.abspath(conf.input_validation)
    conf.label_validation = utils.abspath(conf.label_validation)
    conf.input_testing = utils.abspath(conf.input_testing)
    conf.prediction_dir = utils.abspath(conf.prediction_dir)
    conf.model = utils.abspath(conf.model)
    conf.model_weights = utils.abspath(
        conf.model_weights
    ) if conf.model_weights is not None else conf.model_weights
    conf.output_dir = utils.abspath(conf.output_dir)

    return conf
