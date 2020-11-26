#!/usr/bin/env python
# coding: utf-8

# In[1]:


from azureml.core import Workspace, Experiment

ws = Workspace.get(name="AzureML_Nirmal_Test")
exp = Experiment(workspace=ws, name="udacity-project-nirmal")

print('Workspace name: ' + ws.name, 
      'Azure region: ' + ws.location, 
      'Subscription id: ' + ws.subscription_id, 
      'Resource group: ' + ws.resource_group, sep = '\n')

run = exp.start_logging()


# In[2]:


from azureml.core.compute import ComputeTarget, AmlCompute

# TODO: Create compute cluster
# Use vm_size = "Standard_D2_V2" in your provisioning configuration.
# max_nodes should be no greater than 4.

### YOUR CODE HERE ###
from azureml.core.compute_target import ComputeTargetException

# Choose a name for your CPU cluster
cpu_cluster_name = "cpu-cluster"

# Verify that cluster does not exist already
try:
   cpu_cluster = ComputeTarget(workspace=ws, name=cpu_cluster_name)
   print('Found existing cluster, use it.')
except ComputeTargetException:
   compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_D2_V2',
                                                          max_nodes=4)
   cpu_cluster = ComputeTarget.create(ws, cpu_cluster_name, compute_config)

cpu_cluster.wait_for_completion(show_output=True)


# In[62]:


from azureml.widgets import RunDetails
from azureml.train.sklearn import SKLearn
from azureml.train.hyperdrive.run import PrimaryMetricGoal
from azureml.train.hyperdrive.policy import BanditPolicy
from azureml.train.hyperdrive.sampling import RandomParameterSampling
from azureml.train.hyperdrive.runconfig import HyperDriveConfig
from azureml.train.hyperdrive.parameter_expressions import uniform
import os

# Specify parameter sampler
from azureml.train.hyperdrive.parameter_expressions import choice
ps = RandomParameterSampling( {
                                "--C": uniform(0.1,1),
                                "--max_iter": choice(50,100,150,200)
                                })

# Specify a Policy
policy = BanditPolicy(slack_factor = 0.1, evaluation_interval=1, delay_evaluation=5)

if "training" not in os.listdir():
    os.mkdir("./training")

# Create a SKLearn estimator for use with train.py
est = SKLearn(source_directory = '.', entry_script = 'train.py', compute_target =  cpu_cluster)

# Create a HyperDriveConfig using the estimator, hyperparameter sampler, and policy.
hyperdrive_config = HyperDriveConfig(hyperparameter_sampling = ps,                                    
                                     primary_metric_name = 'Accuracy',                                    
                                     max_total_runs = 5,                                    
                                     max_concurrent_runs = 2,                                   
                                     primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,  
                                     policy = policy,                                   
                                     estimator = est)


# In[63]:


# Submit your hyperdrive run to the experiment and show run details with the widget.

### YOUR CODE HERE ###
hyperdrive_run = exp.submit(hyperdrive_config) 
RunDetails(hyperdrive_run).show()


# In[64]:


# save best run in outputs directory
best_run = hyperdrive_run.get_best_run_by_primary_metric()
joblib.dump(value= best_run.id, filename = 'outputs/model.joblib')


# In[67]:


# register the best model
model = best_run.register_model(model_name='hyperdrive_model', model_path="outputs/model.joblib")


# In[3]:


from azureml.data.dataset_factory import TabularDatasetFactory

# Create TabularDataset using TabularDatasetFactory
# Data is available at: 
# "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"

### YOUR CODE HERE ###
ds1 = TabularDatasetFactory.from_delimited_files(path="https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv")


# In[4]:


from train import clean_data

# Use the clean_data function to clean your data.
x, y = clean_data(ds1)


# In[5]:


# Split data into train and test sets
from sklearn.model_selection import train_test_split
import pandas as pd
x_train, x_test, y_train, y_test = train_test_split(x, y)
df_train = pd.concat([x_train, y_train], axis=1)
from azureml.train.automl import AutoMLConfig
automl_config = AutoMLConfig(
    experiment_timeout_minutes=30,
    task="classification",
    primary_metric="accuracy",
    training_data=df_train,
    label_column_name="y",
    n_cross_validations=5)


# In[7]:


# Submit your automl run

### YOUR CODE HERE ###
automl_run= exp.submit(automl_config, show_output=True)


# In[8]:


# Retrieve and save your best automl model.

### YOUR CODE HERE ###
best_run, best_model = automl_run.get_output()
best_run.register_model(model_name = 'automl_best_model.pkl', model_path = './outputs/')

