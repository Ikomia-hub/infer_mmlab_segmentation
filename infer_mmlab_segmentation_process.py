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
from ikomia.utils import strtobool
import copy
# Your imports below
from mmseg.apis import init_model, inference_model
from mmseg.utils import register_all_modules
from torch.cuda import is_available
import torch
import os
import numpy as np
import yaml


# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferMmlabSegmentationParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        # Example : self.windowSize = 25
        self.model_weight_file = ""
        self.config_file = ""
        self.model_name = "maskformer"
        self.model_config = "maskformer_r50-d32_8xb2-160k_ade20k-512x512"
        self.update = False
        self.cuda = is_available()
        self.use_custom_model = False
        self.custom_cfg = ""
        self.model_path = ""

    def set_values(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.model_weight_file = param_map["model_weight_file"]
        self.config_file = param_map["config_file"]
        self.model_name = param_map["model_name"]
        self.model_config = param_map["model_config"]
        self.cuda = utils.strtobool(param_map["cuda"])
        self.use_custom_model = strtobool(param_map["use_custom_model"])
        self.custom_cfg = param_map["custom_cfg"]
        self.model_path = param_map["model_path"]

    def get_values(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = {
                "model_weight_file": self.model_weight_file,
                "config_file": self.config_file,
                "model_name": self.model_name,
                "model_config": self.model_config,
                "cuda": str(self.cuda),
                "use_custom_model": str(self.use_custom_model),
                "custom_cfg": self.custom_cfg,
                "model_path": self.model_path,
                }
        return param_map


# --------------------
# - Class which implements the process
# - Inhesegformer_mit-b0_8xb2-160k_ade20k-512x512.pyrits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferMmlabSegmentation(dataprocess.CSemanticSegmentationTask):

    def __init__(self, name, param):
        dataprocess.CSemanticSegmentationTask.__init__(self, name)
        # Add input/output of the process here
        register_all_modules()
        self.model = None
        self.classes = None
        # Create parameters class
        if param is None:
            self.set_param_object(InferMmlabSegmentationParam())
        else:
            self.set_param_object(copy.deepcopy(param))

    def get_progress_steps(self, eltCount=1):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1


    @staticmethod
    def get_absolute_paths(param):
        if param.model_weight_file == "":
            yaml_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", param.model_name,
                                     "metafile.yaml")

            if param.model_config.endswith('.py'):
                param.model_config = param.model_config[:-3]
            if os.path.isfile(yaml_file):
                with open(yaml_file, "r") as f:
                    models_list = yaml.load(f, Loader=yaml.FullLoader)['Models']

                available_cfg_ckpt = {model_dict["Name"]: {'cfg': model_dict["Config"],
                                                           'ckpt': model_dict["Weights"]}
                                      for model_dict in models_list}
                if param.model_config in available_cfg_ckpt:
                    cfg_file = available_cfg_ckpt[param.model_config]['cfg']
                    ckpt_file = available_cfg_ckpt[param.model_config]['ckpt']
                    cfg_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg_file)
                    return cfg_file, ckpt_file
                else:
                    raise Exception(
                        f"{param.model_config} does not exist for {param.model_name}. Available configs for are {', '.join(list(available_cfg_ckpt.keys()))}")
            else:
                raise Exception(f"Model name {param.model_name} does not exist.")
        else:
            if os.path.isfile(param.model_config):
                cfg_file = param.model_config
            else:
                cfg_file = param.config_file
            ckpt_file = param.model_weight_file
            return cfg_file, ckpt_file

    @staticmethod
    def get_model_zoo():
        configs_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
        available_pairs = []
        for model_name in os.listdir(configs_folder):
            if model_name.startswith('_'):
                continue
            yaml_file = os.path.join(configs_folder, model_name, "metafile.yaml")
            if os.path.isfile(yaml_file):
                with open(yaml_file, "r") as f:
                    models_list = yaml.load(f, Loader=yaml.FullLoader)
                    if 'Models' in models_list:
                        models_list = models_list['Models']
                    if not isinstance(models_list, list):
                        continue
                for model_dict in models_list:
                    available_pairs.append({"model_name": model_name, "model_config": os.path.basename(model_dict["Name"])})
        return available_pairs

    def run(self):
        # Core function of your process
        # Call begin_task_run for initialization
        self.begin_task_run()

        # Examples :
        # Get input :
        input = self.get_input(0)

        # Get parameters :
        param = self.get_param_object()
        if self.model is None or param.update:
            # Set cache dir in the algorithm folder to simplify deployment
            old_torch_hub = torch.hub.get_dir()
            torch.hub.set_dir(os.path.join(os.path.dirname(__file__), "models"))

            cuda_available = is_available()
            cfg_file, ckpt_file = self.get_absolute_paths(param)

            self.model = init_model(cfg_file, ckpt_file, device='cuda:0' if param.cuda and cuda_available else 'cpu')

            # trick to avoid KeyError "seg_map_path" when loading annotations
            self.model.cfg.test_pipeline = [t for t in self.model.cfg.test_pipeline if "reduce_zero_label" not in t]
            self.classes = self.model.dataset_meta["classes"]
            self.set_names(list(self.classes))

            param.update = False

            # Reset torch cache dir for next algorithms in the workflow
            torch.hub.set_dir(old_torch_hub)
        # Get image from input/output (numpy array):
        srcImage = input.get_image()

        if srcImage is not None:
            result = inference_model(self.model, srcImage).to_dict()
            pred_sem_seg = result["pred_sem_seg"]["data"].detach().cpu().squeeze().numpy()
            self.set_mask(pred_sem_seg.astype("uint8"))

        # Step progress bar:
        self.emit_step_progress()

        # Call end_task_run to finalize process
        self.end_task_run()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferMmlabSegmentationFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_mmlab_segmentation"
        self.info.short_description = "Inference for MMLAB segmentation models"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Segmentation"
        self.info.icon_path = "icons/mmlab.png"
        self.info.version = "1.2.0"
        # self.info.icon_path = "your path to a specific icon"
        self.info.authors = "MMSegmentation Contributors"
        self.info.article = "{MMSegmentation}: OpenMMLab Semantic Segmentation Toolbox and Benchmark"
        self.info.journal = "publication journal"
        self.info.year = 2021
        self.info.license = "Apache 2.0"
        # URL of documentation
        self.info.documentation_link = "https://mmsegmentation.readthedocs.io/en/latest/"
        # Code source repository
        self.info.repository = "https://github.com/Ikomia-hub/infer_mmlab_segmentation"
        self.info.original_repository = "https://github.com/open-mmlab/mmsegmentation"
        # Keywords used for search
        self.info.keywords = "mmlab, train, segmentation"
        self.info.algo_type = core.AlgoType.INFER
        self.info.algo_tasks = "INSTANCE_SEGMENTATION,SEMANTIC_SEGMENTATION,PANOPTIC_SEGMENTATION"

    def create(self, param=None):
        # Create process object
        return InferMmlabSegmentation(self.info.name, param)
