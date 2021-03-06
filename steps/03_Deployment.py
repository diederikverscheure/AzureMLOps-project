import os
import sys
sys.path.append(os.getcwd())

from steps.utils import connectWithAzure
from azureml.core.environment import Environment
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice
from azureml.core import Model

from dotenv import load_dotenv

# When you work locally, you can use a .env file to store all your environment variables.
# This line read those in.
load_dotenv('.env',override=True)

MODEL_NAME = os.environ.get('MODEL_NAME')
MODEL_TYPE = os.environ.get('MODEL_TYPE')
LOCAL_DEPLOYMENT = os.environ.get('LOCAL_DEPLOYMENT')
print(LOCAL_DEPLOYMENT)


def prepareEnv(ws):
    environment_name = os.environ.get('DEPLOYMENT_ENV_NAME')
    conda_dependencies_path = os.environ.get('DEPLOYMENT_DEPENDENCIES')

    env = Environment.from_conda_specification(environment_name, file_path=conda_dependencies_path)
    # Register environment to re-use later
    env.register(workspace = ws)

    return env

def prepareDeployment(ws, environment):

    service_name = os.environ.get('SCORE_SERVICE_NAME')
    entry_script = os.path.join(os.environ.get('SCRIPT_FOLDER'), 'score.py')

    inference_config = InferenceConfig(entry_script=entry_script, environment=environment)
    aci_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1,tags={'type': MODEL_TYPE,
                              'GIT_SHA': os.environ.get('GIT_SHA')})

    # Get our model based on the name we registered in the previous notebook
    model = Model(ws, MODEL_TYPE)

    service = Model.deploy(workspace=ws,
                        name=service_name,
                        models=[model],
                        inference_config=inference_config,
                        deployment_config=aci_config,
                        overwrite=True)
    return service

def downloadLatestModel(ws):
    local_model_path = os.environ.get('LOCAL_MODEL_PATH')
    model = Model(ws, name=MODEL_TYPE)
    model.download(local_model_path, exist_ok=True)
    return model

def main():
    ws = connectWithAzure()

    if os.environ.get('LOCAL_DEPLOYMENT') == "true":
        print('Deploying locally.')
        model = downloadLatestModel(ws)
        print(f'Downloaded the model {model.id} locally. You can now proceed to build the Docker image.')
        return None
    else:
        print('Deploying on Azure.')
        environment = prepareEnv(ws)
        service = prepareDeployment(ws, environment)
        try:
            service.wait_for_deployment(show_output=True)
        except:
            print(service.get_logs())
        return service

if __name__ == '__main__':
    service = main()
    
    # Evaluate with
    # curl -X POST http://c18ca037-2217-486b-be27-e1de9d9e4dcb.westeurope.azurecontainer.io/score -H 'Content-Type: application/json' -d '{"dt": ["2020-01-01 09:00:00","2020-01-01 09:00:05","2020-01-01 09:00:10","2020-01-01 09:00:15","2020-01-01 09:00:20","2020-01-01 09:00:25","2020-01-01 09:00:30","2020-01-01 09:00:35","2020-01-01 09:00:40","2020-01-01 09:00:45"], "co2": [100,200,300,400,500,600,700,800,900,1000]}'