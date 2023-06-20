import glob
import os
import time
import pickle
from colorama import Fore, Style
from tensorflow import keras
from traffic_signs_code.params import *

def load_model():
    """
    Return a saved model:
    Return None (but do not Raise) if no model is found
    """

    # Get the latest model version name by the timestamp on disk
    local_model_directory = os.path.join(LOCAL_MODEL_PATH, 'first_model')
    local_model_paths = glob.glob(f"{local_model_directory}/*")
    if not local_model_paths:
        return None
    most_recent_model_path_on_disk = sorted(local_model_paths)[-1]
    print(Fore.BLUE + f"\n{most_recent_model_path_on_disk}" + Style.RESET_ALL)
    print(Fore.BLUE + f"\nLoad latest model from disk..." + Style.RESET_ALL)
    latest_model = keras.models.load_model(most_recent_model_path_on_disk, compile=False)
    print(":white_check_mark: Model loaded from local disk")
    return latest_model
