# Copyright (C) 2021 Ikomia SAS
# Contact: https://www.ikomia.com
#
# This file is part of the IkomiaStudio software.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from ikomia import core, dataprocess, utils
import copy
# Your imports below
from mmseg.apis import inference_segmentor, init_segmentor
from mmcv.runner import load_checkpoint
from torch.cuda import is_available
from infer_mmlab_segmentation.utils import model_zoo
import os
import numpy as np
import cv2


# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferMmlabSegmentationParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        # Example : self.windowSize = 25
        self.model_name = "segformer"
        self.model_config = "segformer_mit-b3_512x512_160k_ade20k"
        self.model_url = "https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit" \
                         "-b3_512x512_160k_ade20k/segformer_mit-b3_512x512_160k_ade20k_20210726_081410-962b98d2.pth "
        self.update = False
        self.cuda = is_available()
        self.use_custom_model = False
        self.custom_cfg = ""
        self.custom_weights = ""

    def setParamMap(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        # Example : self.windowSize = int(param_map["windowSize"])
        self.model_name = param_map["model_name"]
        self.model_config = param_map["model_config"]
        self.model_url = param_map["model_url"]
        self.cuda = utils.strtobool(param_map["cuda"])
        self.use_custom_model = strtobool(param_map["use_custom_model"])
        self.custom_cfg = param_map["custom_cfg"]
        self.custom_weights = param_map["custom_weights"]

    def getParamMap(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = core.ParamMap()
        # Example : paramMap["windowSize"] = str(self.windowSize)

        param_map["model_name"] = self.model_name
        param_map["model_config"] = self.model_config
        param_map["model_url"] = self.model_url
        param_map["cuda"] = str(self.cuda)
        param_map["use_custom_model"] = str(self.use_custom_model)
        param_map["custom_cfg"] = self.custom_cfg
        param_map["custom_weights"] = self.custom_weights
        return param_map


# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferMmlabSegmentation(dataprocess.C2dImageTask):

    def __init__(self, name, param):
        dataprocess.C2dImageTask.__init__(self, name)
        # Add input/output of the process here
        self.addOutput(dataprocess.CSemanticSegIO())
        self.model = None
        self.colors = None
        self.classes = None
        # Create parameters class
        if param is None:
            self.setParam(InferMmlabSegmentationParam())
        else:
            self.setParam(copy.deepcopy(param))

    def getProgressSteps(self, eltCount=1):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1

    def run(self):
        # Core function of your process
        # Call beginTaskRun for initialization
        self.beginTaskRun()

        # Examples :
        # Get input :
        input = self.getInput(0)

        # Get output :
        sem_seg_out = self.getOutput(1)
        # Get parameters :
        param = self.getParam()
        if self.model is None or param.update:
            if param.use_custom_model:
                cfg_file = param.custom_cfg
                ckpt_file = param.custom_weights
            else:
                cfg_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", param.model_name,
                                        param.model_config+".py")
                ckpt_file = param.model_url

            self.model = init_segmentor(cfg_file, ckpt_file, device='cuda:0' if param.cuda else 'cpu')
            checkpoint = load_checkpoint(self.model, ckpt_file, map_location='cpu')
            if 'CLASSES' in checkpoint.get('meta', {}):
                self.model.CLASSES = checkpoint['meta']['CLASSES']
            if 'PALETTE' in checkpoint.get('meta', {}):
                self.model.PALETTE = checkpoint['meta']['PALETTE']
            self.classes = self.model.CLASSES
            self.colors = self.model.PALETTE
            if self.colors is None:
                self.colors = np.random.randint(0, 255, (len(self.classes), 3))
                self.colors[0] = [0, 0, 0]  # background
            self.colors = [[int(c[0]), int(c[1]), int(c[2])] for c in self.colors]

            param.update = False
        # Get image from input/output (numpy array):
        srcImage = input.getImage()

        self.forwardInputImage(0, 0)

        if srcImage is not None:
            result = inference_segmentor(self.model, srcImage)[0]
            sem_seg_out.setMask(result.astype("uint8"))
            sem_seg_out.setClassNames(list(self.classes), self.colors)

            self.setOutputColorMap(0, 1, self.colors)

        # Step progress bar:
        self.emitStepProgress()

        # Call endTaskRun to finalize process
        self.endTaskRun()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferMmlabSegmentationFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_mmlab_segmentation"
        self.info.shortDescription = "Inference for MMLAB segmentation models"
        self.info.description = "Inference for MMLAB segmentation models"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Segmentation"
        self.info.iconPath = "icons/mmlab.png"
        self.info.version = "1.0.0"
        # self.info.iconPath = "your path to a specific icon"
        self.info.authors = "MMSegmentation Contributors"
        self.info.article = "{MMSegmentation}: OpenMMLab Semantic Segmentation Toolbox and Benchmark"
        self.info.journal = "publication journal"
        self.info.year = 2021
        self.info.license = "Apache 2.0"
        # URL of documentation
        self.info.documentationLink = "https://mmsegmentation.readthedocs.io/en/latest/"
        # Code source repository
        self.info.repository = "https://github.com/open-mmlab/mmsegmentation"
        # Keywords used for search
        self.info.keywords = "mmlab, train, segmentation"

    def create(self, param=None):
        # Create process object
        return InferMmlabSegmentation(self.info.name, param)
