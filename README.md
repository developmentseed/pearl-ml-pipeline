# PEARL ML Training Pipeline

This repo contains scripts to manage training data, workflow to create Azure ML stack and train new models that are compatible to be run on the PEARL Platform. It is based on the work on Caleb Robinson of Microsoft.

## Training

- Monitor experiments and training runs on Azure ML 
- Training Repo
    - [Training code](https://github.com/developmentseed/pearl-ml-pipeline/blob/main/src/train.py) 
    - [Evaluation code](https://github.com/developmentseed/pearl-ml-pipeline/blob/main/src/eval.py)

- [DeepLabv3Plus Architecture](https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/decoders/deeplabv3/model.py) + [focal loss](https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/losses/focal.py) seems most promising approach

## Evaluation
- We run the model over the test data set, and use the per class

### SEED Data

**How/Why we create Seed Data**

- We have seed data for each model so during retraining the user doesn’t have to add samples for each class, so we can use the weights/biases from the retraining logistic regression sklearn model to update the weights/biases of the deep learning model and then run inference on the GPU 
- The retraining seed data should have same class distribution ratios as the original training data (ie 10% water, 50% trees ect)
- I’ve been generating retraining data using the GPU enabled Azure notebooks (these should ideally be converted into scripts) 
- [Seed Data Creation Script](https://github.com/developmentseed/pearl-ml-pipeline/blob/main/src/seed_data_creation.py)



## Training Dataset Creation

There are two options to create the training dataset. 
 
**Option 1**. Feed LULC labels data in GeoTiff format. 

[naip-label-align.py](naip-utils/naip-label-align.py) and [NAIPTileIndex.py](naip-utils/NAIPTileIndex.py) provided functions on how to:

_Notes_:
- Install libspatialindex (dep of `rtree` which is not installed automaticaly)
     - `brew install spatialindex`
- align given LULC labels to available NAIP imagery tiles on Azure public Blob;
- filter out nodata tiles;
- create name conventions;
- write it to CSVs for train, validation and test dataset by 70:20:10. 
- Script will write the tiled label geoTIFF into out_dir. These files can then be uploaded to Azure blob storage

These CSVs can be deployed to AML for model training direction. Instruction will be given in the following section. 

```bash
python naip-label-align.py 
    --label_tif_path sample.tif 
    --out_dir <dir-name>/ 
    --threshold [0.0 to 1.0] 
    --aoi <aoi-name> 
    --group <group-name>
```

**Option 2**. LULC labels available as GeoJSON (vector) files, and rasterization is required. 

- Firstly, NAIP imagery that overlap with LULC label data is needed to be downloaded before the rasterization task. 
[naip_download_pc.ipynb](naip-utils/naip_download_pc.ipynb) provides script and documentation on how you can download NAIP imagery to your AOI from [MS Plentary Computer](https://planetarycomputer.microsoft.com/dataset/naip). 

- Secondly, LULC label rasterization functions and steps provided in [label_rasterize.ipynb](naip-utils/label_rasterize.ipynb) 
The rasterization in the order of (tree canopy on the top of the lulc layer or burn last, other_impervious on the bottom or it should be rasterized first in the order)
```
tree_canopy
building
water
bare_soil
roads_railroads
grass_shrub
other_impervious
```
Details see the [notebook](naip-utils/label_rasterize.ipynb).


## Model Training on Azure ML(AML) 
If you are going to use AML to train LULC models for the first time, please go through these steps.

<img width="1426" alt="Screen Shot 2021-11-08 at 8 20 04 AM" src="https://user-images.githubusercontent.com/14057932/140749239-6963fd38-d8cb-40a6-b2eb-2cd1869ab897.png">

### Configure environment

This code was tested using `python 3.6.5`

[Create a conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) using `.pytorch-env.yaml` file and execute the scripts from the created environment.


You will need to set the following variables in your `.env`

bash
```
AZ_TENANT_ID=XXX #az account show --output table
AZ_SUB_ID=XXX #az account list --output table

AZ_WORKSPACE_NAME=XXX #User set
AZ_RESOURCE_GROUP=XXX #User set
AZ_REGION=XXX #User set

AZ_GPU_CLUSTER_NAME=XXX #User set
AZ_CPU_CLUSTER_NAME=XXX #User set
```

Then export all variables to your environment:

```
export $(cat .env);
```


### Create Your Workspace on AML
[train_azure/create_workspace.py](train_azure/create_workspace.py) after export your Azure credentials, this script will create AML workspace. 

### Create GPU Compute 

[This script](train_azure/create_compute-gpu.py) will create GPU compute resources to your workspace on AML. 


### (Optional) Create CPU Compute 

[This script](train_azure/create_compute-cpu.py) will create GPU compute resources to your workspace on AML. 


### Train LULC Model on AML
We have three PyTorch based Semantic Segmenation models ready for LULC model trainings, FCN, UNet and DeepLabV3+. 

To train a model on AML, you will need to define or parse a few crucial parameters to the [script](train_azure/run_model.py), for instance:

TODO: Will we be providing sample csv
```python
config = ScriptRunConfig(
    source_directory="./src",
    script="train.py",
    compute_target=AZ_GPU_CLUSTER_NAME,
    arguments=[
        "--input_fn",
        "sample_data/indianapolis_train.csv",
        "--input_fn_val",
        "sample_data/indianapolis_val.csv",
        "--output_dir",
        "./outputs",
        "--save_most_recent",
        "--num_epochs",
        20,
        "--num_chips",
        200,
        "--num_classes",
        7,
        "--label_transform",
        "uvm",
        "--model",
        "deeplabv3plus",
    ],
)
```

These parameters are to be configure by the user. `input_fn_X` paths should be provided by the user, and are the outputs of the data generation step (NAIP Label Algin) described above.

`python train_azure/run_model.py`


### Evaluate the Trained Model

To compute Global F1, and class base F1 scores (written in CSV) from a trained model over latest dataset. You can use this [eval script](train_azure/run_eval.py) as an example. 

`python train_azure/run_eval.py`


### Seed Data Creation for PEARL
After a best performing model is selected, seed dataseed need to be created to serve PEARL. Seed Data is the model embedding layers from the trained model that is used together with users inputs training data in PEARL retraining session. 

[run_seeddata_creation.py](train_azure/run_seeddata_creation.py) will config AML and use the [main seeddata creation script](src/seed_data_creation.py) to create seed data for the trained best performing model. 

`python train_azure/run_seeddata_creation.py`

### (Optional) Classes Distribution

LULC Class distribution is a graph showing the porpotion of LULC pixel numbers for a trained model on PEARL. See the bar chart bellow.

[train_azure/run_cls_distrib.py](train_azure/run_cls_distrib.py) will guide you how to compute the classes distribution from the training dataset for the model. 

`python train_azure/run_cls_distrib.py`

<img width="1255" alt="Screen Shot 2021-11-08 at 8 07 49 AM" src="https://user-images.githubusercontent.com/14057932/140747356-31c90a9b-5cce-4b52-a74f-ca2841e9549c.png">
