import os
from azureml.core import Workspace
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core.authentication import InteractiveLoginAuthentication


# try:
AZ_TENANT_ID = os.getenv("AZ_TENANT_ID")
interactive_auth = InteractiveLoginAuthentication(tenant_id=AZ_TENANT_ID)

ws = Workspace.from_config()  # This automatically looks for a directory .azureml

# Choose a name for your CPU cluster
# memory optimized: https://docs.microsoft.com/en-us/azure/virtual-machines/dv2-dsv2-series-memory
AZ_CPU_CLUSTER_NAME = os.getenv("AZ_CPU_CLUSTER_NAME")

# Verify that the cluster does not exist already
try:
    cpu_cluster = ComputeTarget(workspace=ws, name=AZ_CPU_CLUSTER_NAME)
    print("Found existing cluster, use it.")
except ComputeTargetException:
    compute_config = AmlCompute.provisioning_configuration(
        vm_size="Standard_DS12_v2",
        idle_seconds_before_scaledown=1200,
        min_nodes=0,
        max_nodes=3,
    )
    cpu_cluster = ComputeTarget.create(ws, AZ_CPU_CLUSTER_NAME, compute_config)

cpu_cluster.wait_for_completion(show_output=True)
