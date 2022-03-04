import os
from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core import Environment
from azureml.core import ScriptRunConfig

AZ_GPU_CLUSTER_NAME = os.environ.get('AZ_GPU_CLUSTER_NAME')

if __name__ == "__main__":
    ws = Workspace.from_config()
    experiment = Experiment(workspace=ws, name="sample-exp-fort-collins")
    config = ScriptRunConfig(
        source_directory="./src",
        script="train.py",
        compute_target=AZ_GPU_CLUSTER_NAME,
        arguments=[
            "--input_fn",
            "sample_data/fort-collins_train.csv",
            "--input_fn_val",
            "sample_data/fort-collins_val.csv",
            "--output_dir",
            "./outputs",
            "--save_most_recent",
            "--num_epochs",
            20,
            "--num_chips",
            200,
            "--num_classes",
            8,
            "--label_transform",
            "uvm",
            "--model",
            "deeplabv3plus",
        ],
    )

    # set up pytorch environment
    pytorch_env = Environment.from_conda_specification(
        name="lulc-pytorch-env", file_path="./pytorch-env.yml"
    )

    # Specify a GPU base image
    pytorch_env.docker.enabled = True
    pytorch_env.docker.base_image = (
        "mcr.microsoft.com/azureml/openmpi3.1.2-cuda10.1-cudnn7-ubuntu18.04"
    )

    config.run_config.environment = pytorch_env

    run = experiment.submit(config)

    aml_url = run.get_portal_url()
    print(aml_url)
