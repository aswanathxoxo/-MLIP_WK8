from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import pandas as pd
import yaml

def load_data(config_path="../config/params.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]
    return train_test_split(X, y, test_size=config["test_size"], random_state=config["random_seed"])