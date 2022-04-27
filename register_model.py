from azureml.core.model import Model
from azureml.core import Workspace

def main(model_name, model_path):
    """Register model from Notebooks to Models

    :param model_name: Name of the model to be registered
    :param model_path: Path of the model in Notebooks
    """
    ws = Workspace.from_config()
    model = Model.register(ws, model_name=model_name, model_path=model_path)

if __name__ == "__main__":
    model_name = "" # TO BE COMPLETED
    model_path = "" # TO BE COMPLETED
    main(model_name, model_path)

    exit(0)