<div align="center">
  <img src="https://raw.githubusercontent.com/Ikomia-hub/infer_mmlab_segmentation/main/icons/mmlab.png" alt="Algorithm icon">
  <h1 align="center">infer_mmlab_segmentation</h1>
</div>
<br />
<p align="center">
    <a href="https://github.com/Ikomia-hub/infer_mmlab_segmentation">
        <img alt="Stars" src="https://img.shields.io/github/stars/Ikomia-hub/infer_mmlab_segmentation">
    </a>
    <a href="https://app.ikomia.ai/hub/">
        <img alt="Website" src="https://img.shields.io/website/http/app.ikomia.ai/en.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/Ikomia-hub/infer_mmlab_segmentation/blob/main/LICENSE.md">
        <img alt="GitHub" src="https://img.shields.io/github/license/Ikomia-hub/infer_mmlab_segmentation.svg?color=blue">
    </a>    
    <br>
    <a href="https://discord.com/invite/82Tnw9UGGc">
        <img alt="Discord community" src="https://img.shields.io/badge/Discord-white?style=social&logo=discord">
    </a> 
</p>

Run object detection and instance segmentation algorithms from MMLAB framework. 

Models will come from MMLAB's model zoo if custom training is disabled. If not, you can choose to load your model trained with algorithm *train_mmlab_detection* from Ikomia HUB. In this case, make sure to set parameters for config file (.py) and model file (.pth). Both of these files are produced by the train algorithm.

![Example image](https://raw.githubusercontent.com/Ikomia-hub/infer_mmlab_segmentation/main/icons/output.png)


## :rocket: Use with Ikomia API

#### 1. Install Ikomia API

We strongly recommend using a virtual environment. If you're not sure where to start, we offer a tutorial [here](https://www.ikomia.ai/blog/a-step-by-step-guide-to-creating-virtual-environments-in-python).

```sh
pip install ikomia
```

#### 2. Create your workflow

```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()

# Add object detection algorithm
segmentor = wf.add_task(name="infer_mmlab_segmentation", auto_connect=True)

# Run the workflow on image
wf.run_on(url="https://production-media.paperswithcode.com/datasets/Foggy_Cityscapes-0000003414-fb7dc023.jpg")

# Get and display results
image_output = segmentor.get_output(0)
segmentation_output = segmentor.get_output(1)

display(image_output.get_image_with_mask(segmentation_output), title="MMLAB segmentation")
```

## :sunny: Use with Ikomia Studio

Ikomia Studio offers a friendly UI with the same features as the API.

- If you haven't started using Ikomia Studio yet, download and install it from [this page](https://www.ikomia.ai/studio).

- For additional guidance on getting started with Ikomia Studio, check out [this blog post](https://www.ikomia.ai/blog/how-to-get-started-with-ikomia-studio).

## :pencil: Set algorithm parameters
```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()

# Add object detection algorithm
segmentor = wf.add_task(name="infer_mmlab_segmentation", auto_connect=True)

segmentor.set_parameters({
        "model_name": "pidnet",
        "model_config": "pidnet-m_2xb6-120k_1024x1024-cityscapes",
        "config_file": "",
        "model_weight_file": "",
        "cuda": "True",
    })

# Run the workflow on image
wf.run_on(url="https://production-media.paperswithcode.com/datasets/Foggy_Cityscapes-0000003414-fb7dc023.jpg")
```
- **model_name** (str, default="maskformer"): model name. 
- **model_config** (str, default="maskformer_r50-d32_8xb2-160k_ade20k-512x512"): name of the model configuration file.
- **config_file** (str, default=""): path to model config file (only if *use_custom_model=True*). The file is generated at the end of a custom training. Use algorithm ***train_mmlab_detection*** from Ikomia HUB to train custom model.
- **model_weight_file** (str, default=""): path to model weights file (.pt) (only if *use_custom_model=True*). The file is generated at the end of a custom training.
- **cuda** (bool, default=True): CUDA acceleration if True, run on CPU otherwise.

MMLab framework for object detection and instance segmentation offers a large range of models. To ease the choice of couple (model_name/model_config), you can call the function *get_model_zoo()* to get a list of possible values.

```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()

# Add object detection algorithm
segmentor = wf.add_task(name="infer_mmlab_segmentation", auto_connect=True)

# Get list of possible models (model_name, model_config)
print(segmentor.get_model_zoo())
```

## :mag: Explore algorithm outputs

Every algorithm produces specific outputs, yet they can be explored them the same way using the Ikomia API. For a more in-depth understanding of managing algorithm outputs, please refer to the [documentation](https://ikomia-dev.github.io/python-api-documentation/advanced_guide/IO_management.html).

```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_mmlab_segmentation", auto_connect=True)

# Run on your image  
wf.run_on(url="https://production-media.paperswithcode.com/datasets/Foggy_Cityscapes-0000003414-fb7dc023.jpg")

# Iterate over outputs
for output in algo.get_outputs():
    # Print information
    print(output)
    # Export it to JSON
    output.to_json()
```
