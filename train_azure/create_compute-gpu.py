import os
from azureml.core import Workspace
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core.authentication import InteractiveLoginAuthentication

AZ_TENANT_ID = os.getenv("AZ_TENANT_ID")
interactive_auth = InteractiveLoginAuthentication(tenant_id=AZ_TENANT_ID)
AZ_SUB_ID = os.getenv("AZ_SUB_ID")

ws = Workspace.from_config()  # This automatically looks for a directory .azureml

AZ_GPU_CLUSTER_NAME = os.getenv("AZ_GPU_CLUSTER_NAME")

# Verify that the cluster does not exist already
try:
    gpu_cluster = ComputeTarget(workspace=ws, name=AZ_GPU_CLUSTER_NAME)
    print("Found existing cluster, use it.")
except ComputeTargetException:
    # https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-attach-compute-cluster?tabs=python#managed-identity
    print("Creating new gpu cluster...")
    compute_config = AmlCompute.provisioning_configuration(
        vm_size="Standard_NC6",
        idle_seconds_before_scaledown=1200,
        min_nodes=0,
        max_nodes=3,
    )
    gpu_cluster = ComputeTarget.create(ws, AZ_GPU_CLUSTER_NAME, compute_config)

gpu_cluster.wait_for_completion(show_output=True)
