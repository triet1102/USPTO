from azureml.core import Workspace
from azureml.core import Dataset
from azureml.data.datapath import DataPath
import sys


def main():
    """Upload data from local to datastore
    """

    ws = Workspace.from_config()
    datastore = ws.get_default_datastore()
    Dataset.File.upload_directory(src_dir='data', 
                                target=DataPath(datastore, "datasets/USPTO_data")
                                )  


if __name__ == "__main__":
    main()
    sys.exit(0)
