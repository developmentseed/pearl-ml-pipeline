"""
Example test script to deploy seed data creation to Azure Machine Learning
"""
import os, sys
from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core import Environment
from azureml.core import ScriptRunConfig
from azureml.core import Dataset
from azureml.core.authentication import InteractiveLoginAuthentication


AZ_TENANT_ID = os.getenv("AZ_TENANT_ID")
AZ_GPU_CLUSTER_NAME = os.getenv("AZ_GPU_CLUSTER_NAME")

if __name__ == "__main__":
    interactive_auth = InteractiveLoginAuthentication(tenant_id=AZ_TENANT_ID)
    try:
        ws = Workspace.from_config()
    except:
        print("No config found. Please create a workspace before running")
        sys.exit(0)

    experiment = Experiment(
        workspace=ws,
        name="sample-exp-indianapolis-seeddata",
    )

    config = ScriptRunConfig(
        source_directory="./src",
        script="seed_data_creation.py",
        compute_target=AZ_GPU_CLUSTER_NAME,
        arguments=[
            "--input_csv",
            "sample_data/indianapolis_test.csv",
            "--ckpt_file",
            "sample_data/indianapolis_most_recent_model.pt",  # replace with weight on azure
            "--n_classes",
            7,
            "--out_npz",
            "./outputs/sample-output-indianapolis.npz",
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
