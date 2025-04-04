from keras import layers, models, saving
import os
import warnings
from sklearn.exceptions import DataConversionWarning
from src.models.abstract_model import Model, ColumnExtractor
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.metrics import mean_absolute_error, make_scorer, root_mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_validate, train_test_split
from keras.api.wrappers import SKLearnRegressor


warnings.filterwarnings(action='ignore',category=DataConversionWarning)
np.random.seed(42)
matplotlib.rcParams['figure.figsize']=[12,12]

class TF(Model):
    
    def __init__(self):
        self.path = "./TF"
        self.objetive = "Close"
        self.metric = ""
        os.makedirs(self.path,exist_ok=True)

    def train(self, datapath:str, trainParams:dict):
        try:
           df = pd.read_csv(datapath, parse_dates=True, dayfirst=False, index_col="Date")
           df.dropna(inplace=True)
           df[df.columns] = df[df.columns].replace(",", "", regex=True).astype(float)
           col_num = df.drop(columns=[self.objetive], axis=1, errors="ignore").select_dtypes(include=[np.number]).columns
           pipeline_extractor = Pipeline(
               [
                ("selecter",ColumnExtractor(columns=col_num)),
                ("imputer", SimpleImputer()),
                ("standarizer", MinMaxScaler(feature_range=(0,1)))
               ]
           )
        except: 
            raise ValueError("[-] Error loading dataset")
        
        self.train_df, self.test_df = train_test_split(df,test_size=0.30, random_state=50)

        pipeline_processed = FeatureUnion([("pipeline_extractor", pipeline_extractor)])

        input_layer_unit = trainParams.get("input_layer_unit", 64)
        input_layer_activation = trainParams.get("input_layer_activation", "relu")
        hidden_layers:dict = trainParams.get("hidden_layers", {"units": 64, "activation": "relu", "total": 2})
        loss = trainParams.get("loss", 'mean_squared_error')
        metrics = trainParams.get("metrics", "accuracy")
        optimizer = trainParams.get("optimizer", "adam")
        epoch = trainParams.get("epoch", 100)
        batch_size = trainParams.get("batch_size", 10)
        verbose = trainParams.get("verbose", 1)
        jobs = trainParams.get("jobs", -1)
        cv = trainParams.get("cv", 15)

        if trainParams.get("metric","none") == "rmse":
            self.metric = make_scorer(root_mean_squared_error, greater_is_better=False)
        else:
            self.metric = make_scorer(mean_absolute_error, greater_is_better=False)

        self.__model = models.Sequential()
        self.__model.add(layers.Dense(input_layer_unit, activation=input_layer_activation, input_shape=(len(col_num),)))

        for i in range(hidden_layers.get("total")):
            self.__model.add(
                layers.Dense(hidden_layers.get("units"), hidden_layers.get("activation") )
            )
        
        self.__model.add(
            layers.Dense(1, "linear")
        )

        self.__model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=[metrics]
        )
    
        self.__keras_model = Pipeline(
            [
                ("pipeline_processed", pipeline_processed),
                ("estimator", SKLearnRegressor(self.__model))
            ]
        )
        
        self.evaluate_model(cv, jobs, epoch, batch_size)
        self.show_result(self.__keras_model.predict(self.test_df))
        
        
        
    def evaluate_model(self, cv, jobs, epochs, batch_size) -> any: 
        print("[+] Training...")
        try:
            result = cross_validate(self.__keras_model, self.train_df, self.train_df[self.objetive], 
                                  scoring=self.metric, cv=cv, n_jobs=jobs, return_train_score=True, error_score="raise")
            prod_fit_time = 0
            for i in result["fit_time"] :
                prod_fit_time += i
            prod_train_score = 0 
            for i in result["train_score"]:
                prod_train_score += i 
            prod_test_score = 0 
            for i in result["test_score"]:
                prod_test_score += i

            print("[+] Average results:\n\t fit_time: {} train_score: {} test_score: {}".
                  format(prod_fit_time / len(result["fit_time"]), 
                         (prod_train_score / len(result["train_score"])) * -1,
                         (prod_test_score / len(result["test_score"])) * -1
            ))

            self.__keras_model.fit(
                self.train_df, 
                self.train_df[self.objetive], 
                estimator__epochs=epochs,
                estimator__batch_size=batch_size
            )
        except Exception as e:
            raise ValueError(e)
        
    def show_result(self, y_pred) :
        y_real = self.test_df[self.objetive]

        fig, ax = plt.subplots()
        ax.scatter(y_real, y_pred, color="blue", alpha=0.6, label="Predicciones")
        min_val = min(min(y_real), min(y_pred))
        max_val = max(max(y_real), max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", label="Ideal")
        
        ax.set_xlabel("Valores reales")
        ax.set_ylabel("Valores estimados")
        ax.set_title("Relación valores reales y estimados")
        ax.legend()
        
        plt.savefig(self.path + "/Matriz-TF.png", dpi=300, bbox_inches="tight")

    
    def save(self, modelName:str) -> str:
        try:
            self.__model.save(self.path + "/" + modelName + ".keras")
        except Exception as e: 
            raise ValueError("[-] Error: {}".format(e))

        return "[+] Modelo guardado {}".format(self.path)
    
    
    def load(self, modelName:str):
        try:
             self.__keras_model = saving.load_model(modelName)
        except Exception as e: 
            raise ValueError("[-] {}".format(e))
        return "[+] Modelo cargado {}".format(modelName)
    

    
    def predic(self, path:str):
        try:
            data = pd.read_csv(path, parse_dates=True, dayfirst=False, index_col="Date")
            data.dropna(inplace=True)
            data[data.columns] = data[data.columns].replace(",", "", regex=True).astype(float)
            if "Close" in data.columns:
                data = data.drop("Close", axis=1)
        except Exception as e:
            raise ValueError("[-] {}".format(e))
        
        return self.__keras_model.predict(data)