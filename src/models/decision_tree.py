from src.models.abstract_model import Model
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import joblib
import os


class DecisionTree(Model):
    def __init__(self):
        self.model = None
        self.path = "./DecisionTree"
        os.makedirs(self.path, exist_ok=True)

    def train(self, data, trainParams):
        X = data.drop(columns=[trainParams.get("target", "Close")])
        y = data[trainParams.get("target", "Close")]
        params = {k: v for k, v in trainParams.items() if k != "target"}
        self.model = DecisionTreeRegressor(**params)
        self.model.fit(X, y)

    def save(self, modelName):
        if self.model is not None:
            joblib.dump(self.model, os.path.join(self.path, modelName + ".pkl"))
        else:
            raise ValueError("No model to save.")

    def load(self, modelName):
        self.model = joblib.load(os.path.join(self.path, modelName + ".pkl"))

    def predict(self, data):
        if self.model is not None:
            return self.model.predict(data)
        else:
            raise ValueError("Model not loaded.")