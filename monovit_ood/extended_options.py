from __future__ import absolute_import, division, print_function

import os
import sys
sys.path.append(os.getcwd()+"/../monodepth2_ood/")
from monodepth2.options import MonodepthOptions


# Extended set of options
class OODOptions(MonodepthOptions):

    def __init__(self):
        super(OODOptions, self).__init__()
        self.parser.add_argument("--log",
                                 help="if set, adds the variance output to monodepth2 according to log-likelihood "
                                      "maximization technique",
                                 action="store_true")
        self.parser.add_argument("--dropout", help="if set enables dropout inference", action="store_true")
        self.parser.add_argument("--output_dir", type=str, default="output",
                                 help="output directory for predicted depth and uncertainty maps")

        # possibility to load the uncertainty output without evaluating the log-likelihood
        self.parser.add_argument("--uncert", help="if set will train with uncertainty output", action="store_true")

        # additional options ood evaluation
        self.parser.add_argument("--ood_data", type=str, help="path to the ood dataset")
        self.parser.add_argument("--ood_dataset", type=str, default="places365", help="selected ood dataset",
                                 choices=["places365", "india_driving", "virtual_kitti"])
        self.parser.add_argument("--autoencoder", action="store_true", help="if set will evaluate the image decoder")
        self.parser.add_argument("--bayescap", action="store_true", help="if set will use bayescap for ood evaluation")
        self.parser.add_argument("--plot_results", action="store_true", help="plots the visual results of the selected "
                                                                             "method")

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
