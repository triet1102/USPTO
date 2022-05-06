from azureml.core import ScriptRunConfig
from azureml.core import Experiment
from azureml.core import Environment
from azureml.core.runconfig import MpiConfiguration
from azureml.core import Workspace
import sys
from azureml.core.runconfig import PyTorchConfiguration
from azureml.core import Dataset
import os


def main():
    """Submit train job to compute cluster
    """

    # Init workspace
    ws = Workspace.from_config()

    # Init datastore
    ds = ws.get_default_datastore()
    USPTO_file_dataset = Dataset.File.from_files(
        path=(ds, 'datasets/USPTO_data'))

    # Init experiment
    experiment = Experiment(
        workspace=ws, name='USPTO-train')
    distr_config = PyTorchConfiguration(process_count=8, node_count=2)

    # Init configuration
    config = ScriptRunConfig(source_directory="./src",
                             script="train.py",
                             compute_target="gpu-cluster",
                             distributed_job_config=distr_config,
                             arguments=[
                                 "--val-size", 0.05,
                                 "--data-folder", USPTO_file_dataset.as_mount(),
                                 "--model-name", "microsoft/mpnet-base"
                             ]
                             )

    env = Environment.from_conda_specification(
        name='venv',
        file_path='environment.yml'
    )

    config.run_config.environment = env
    run = experiment.submit(config)
    aml_url = run.get_portal_url()
    print(f"See training insights from: \n{aml_url}")
    run.wait_for_completion()

    # # Can load a run with
    # ws = Workspace.from_config()
    # run = ws.get_run('USPTO-train_1651580405_5fb62f0b')
    # os.makedirs("./saved_models", exist_ok=True)
    # run.download_files(prefix="./outputs",
    #                     output_directory="./saved_models")

    # save_model = True
    # if save_model:
    #     os.makedirs("./saved_models", exist_ok=True)
    #     run.download_files(prefix="./outputs",
    #                        output_directory="./saved_models")


if __name__ == "__main__":
    main()
    sys.exit(0)
