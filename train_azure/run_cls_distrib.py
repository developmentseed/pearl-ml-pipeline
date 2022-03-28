import os, sys
from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core import Environment
from azureml.core import ScriptRunConfig
from azureml.core.authentication import InteractiveLoginAuthentication

AZ_TENANT_ID = os.getenv("AZ_TENANT_ID")
AZ_CPU_CLUSTER_NAME = os.getenv("AZ_CPU_CLUSTER_NAME")
if __name__ == "__main__":
    interactive_auth = InteractiveLoginAuthentication(tenant_id=AZ_TENANT_ID)
    try:
        ws = Workspace.from_config()
    except:
        print("No config found. Please create a workspace before running")
        sys.exit(0)

    experiment = Experiment(workspace=ws, name="sample-exp-fortcollins")
    config = ScriptRunConfig(
        source_directory="./src",
        script="cls_distribution.py",
        compute_target=AZ_CPU_CLUSTER_NAME,
        arguments=[
            "--input_fn",
            "data/fort-collins_test.csv",
            "--num_classes",
            7,
            "--label_transform",
            "uvm",  # either 'naip or epa'
            "--output_dir",
            "./outputs",  # TBD don't actually want to use outputdir
        ],
    )

    # set up pytorch environment
    pytorch_env = Environment.from_conda_specification(
        name="lulc-pytorch-env", file_path="./pytorch-env.yml"
    )

    # This env variable needs to be set for rasterio to open remote files
    # https://github.com/mapbox/rasterio/issues/1289
    pytorch_env.environment_variables[
        "CURL_CA_BUNDLE"
    ] = "/etc/ssl/certs/ca-certificates.crt"

    config.run_config.environment = pytorch_env

    run = experiment.submit(config)

    aml_url = run.get_portal_url()
    print(aml_url)
