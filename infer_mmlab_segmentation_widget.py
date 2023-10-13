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

from ikomia import core, dataprocess
from ikomia.utils import pyqtutils, qtconversion
from infer_mmlab_segmentation.infer_mmlab_segmentation_process import InferMmlabSegmentationParam

# PyQt GUI framework
from PyQt5.QtWidgets import *
from torch.cuda import is_available
import os
import yaml
from PyQt5 import QtCore


def completion(word_list, widget, i=True):
    """ Autocompletion of sender and subject """
    word_set = set(word_list)
    completer = QCompleter(word_set)
    if i:
        completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
    else:
        completer.setCaseSensitivity(QtCore.Qt.CaseSensitive)
    completer.setFilterMode(QtCore.Qt.MatchFlag.MatchContains)
    widget.setCompleter(completer)


class Autocomplete(QComboBox):
    def __init__(self, items, parent=None, i=False, allow_duplicates=True):
        super(Autocomplete, self).__init__(parent)
        self.items = items
        self.insensitivity = i
        self.allowDuplicates = allow_duplicates
        self.init()

    def init(self):
        self.setEditable(True)
        self.setDuplicatesEnabled(self.allowDuplicates)
        self.addItems(self.items)
        self.setAutocompletion(self.items, i=self.insensitivity)

    def setAutocompletion(self, items, i):
        completion(items, self, i)


# --------------------
# - Class which implements widget associated with the process
# - Inherits PyCore.CWorkflowTaskWidget from Ikomia API
# --------------------
class InferMmlabSegmentationWidget(core.CWorkflowTaskWidget):

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = InferMmlabSegmentationParam()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.gridLayout = QGridLayout()
        self.available_models = []
        for dir in os.listdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")):
            if dir != "_base_":
                self.available_models.append(dir)

        self.combo_model = Autocomplete(self.available_models, parent=None, i=True, allow_duplicates=False)
        self.label_model = QLabel("Model name")

        self.gridLayout.addWidget(self.combo_model, 0, 1)
        self.gridLayout.addWidget(self.label_model, 0, 0)

        self.combo_config = pyqtutils.append_combo(self.gridLayout, "Config")

        self.combo_model.editTextChanged.connect(self.on_model_changed)

        self.combo_model.setCurrentText(self.parameters.model_name)

        self.on_model_changed("")

        self.combo_config.setCurrentText(self.parameters.model_config +".py" if not self.parameters.model_config.endswith(".py") else "")

        self.check_cuda = pyqtutils.append_check(self.gridLayout, "Use cuda", self.parameters.cuda and is_available())

        self.browse_custom_cfg = pyqtutils.append_browse_file(self.gridLayout, "Config file (.py)",
                                                              self.parameters.custom_cfg)

        self.browse_custom_weights = pyqtutils.append_browse_file(self.gridLayout, "Weight file (.pth)",
                                                                  self.parameters.model_path)

        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.gridLayout)

        # Set widget layout
        self.set_layout(layout_ptr)

    def on_model_changed(self, s):
        self.combo_config.clear()
        model = self.combo_model.currentText()
        yaml_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", model, "metafile.yaml")
        if os.path.isfile(yaml_file):
            with open(yaml_file, "r") as f:
                models_list = yaml.load(f, Loader=yaml.FullLoader)['Models']
            available_cfg = [model_dict["Name"] for
                             model_dict in models_list
                             if "Weights" in model_dict]
            self.combo_config.addItems(available_cfg)
            self.combo_config.setCurrentText(available_cfg[0])


    def on_apply(self):
        # Apply button clicked slot

        # Get parameters from widget
        # Example : self.parameters.windowSize = self.spinWindowSize.value()
        self.parameters.cuda = self.check_cuda.isChecked()
        self.parameters.model_config = self.combo_config.currentText()
        self.parameters.model_name = self.combo_model.currentText()
        self.parameters.model_weight_file = self.browse_custom_weights.path
        self.parameters.config_file = self.browse_custom_cfg.path
        self.parameters.update = True
        # Send signal to launch the process
        self.emit_apply(self.parameters)


# --------------------
# - Factory class to build process widget object
# - Inherits PyDataProcess.CWidgetFactory from Ikomia API
# --------------------
class InferMmlabSegmentationWidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the name of the process -> it must be the same as the one declared in the process factory class
        self.name = "infer_mmlab_segmentation"

    def create(self, param):
        # Create widget object
        return InferMmlabSegmentationWidget(param, None)
