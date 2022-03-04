"""
Setting up workspace on Azure ML stadio
"""
import os
from azureml.core import Workspace
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core.authentication import InteractiveLoginAuthentication

# get your TENANT_ID from "az account show --output table"
# get your "subscription_id" from "az account list --output table"
AZ_TENANT_ID = os.getenv("AZ_TENANT_ID")
AZ_SUB_ID = os.getenv("AZ_SUB_ID")

interactive_auth = InteractiveLoginAuthentication(tenant_id=AZ_TENANT_ID)


AZ_WORKSPACE_NAME = os.getenv("AZ_WORKSPACE_NAME")
AZ_RESOURCE_GROUP = os.getenv("AZ_RESOURCE_GROUP")
AZ_REGION = os.getenv("AZ_REGION")

ws = Workspace.create(
    name=AZ_WORKSPACE_NAME,  # provide a name for your workspace
    subscription_id=AZ_SUB_ID,  # provide your subscription ID
    resource_group=AZ_RESOURCE_GROUP,  # provide a resource group name
    create_resource_group=True,
    location=AZ_REGION,
)  # For example: 'westeurope' or 'eastus2' or 'westus2' or 'southeastasia'.

# write out the workspace details to a configuration file: .azureml/config.json
ws.write_config(path=".azureml")
