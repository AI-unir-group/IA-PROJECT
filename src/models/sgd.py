from src.models.abstract_model import Model,ColumnExtractor
import pandas as pd 
import numpy as np 
from numpy import ndarray
import matplotlib.pyplot as plt 
import matplotlib
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, make_scorer, root_mean_squared_error
from sklearn.model_selection import cross_validate, train_test_split
import warnings
from sklearn.exceptions import DataConversionWarning
import joblib
import os


warnings.filterwarnings(action='ignore',category=DataConversionWarning)
np.random.seed(42)
matplotlib.rcParams['figure.figsize']=[12,12]



class SGD(Model):
    
    def __init__(self):
        self.path = "./SGD"
        self.objetive = "Close"
        self.metric = ""
        os.makedirs(self.path,exist_ok=True)

    def train(self, dataPath:str, trainParams:dict):
        try:
           df = pd.read_csv(dataPath, parse_dates=True, dayfirst=False, index_col="Date")
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
        
        if trainParams.get("metric","none") == "rmse":
            self.metric = make_scorer(root_mean_squared_error, greater_is_better=False)
        else:
            self.metric = make_scorer(mean_absolute_error, greater_is_better=False)
        
        
        cv = trainParams.get("cv", 15)
        jobs = trainParams.get("jobs", -1)
        max_epoch = trainParams.get("epoch", 500)
        tol = trainParams.get("tol", 1e-3)
        loss = trainParams.get("loss", "squared_error")
        alpha = trainParams.get("alpha", 0.0001)
        l1 = trainParams.get("l1", 0.15)
        shuffle = trainParams.get("shuffle", True)
        lr = trainParams.get("lr", "invscaling")
        early_stopping = trainParams.get("early_stopping", False)
        random_state = trainParams.get("random_state", None)
        penalty = trainParams.get("penalty", "l2")
        eta = trainParams.get("eta", 0.01)
        epsilon = trainParams.get("epsilon", 0.1)
        verbose= trainParams.get("verbose", 0)
        n_iter_stop = trainParams.get("n_iter_stop", 5)

        self.__model = Pipeline(
            [
                ("pipeline_processed", pipeline_processed),
                ("estimator", SGDRegressor(
                    max_iter=max_epoch, 
                    tol=tol,
                    penalty=penalty,
                    l1_ratio=l1,
                    shuffle=shuffle,
                    loss=loss,
                    alpha=alpha,
                    learning_rate=lr,
                    early_stopping=early_stopping,
                    random_state=random_state,
                    eta0=eta,
                    epsilon=epsilon,
                    verbose=verbose,
                    n_iter_no_change=n_iter_stop
                ))
            ]
        )

        self.evaluate_model(cv, jobs)
        self.show_result(self.__model.predict(self.test_df))

    
    def evaluate_model(self, cv, jobs) -> object: 
        print("[+] Training...")
        try:
            result = cross_validate(self.__model, self.train_df, self.train_df[self.objetive], 
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

            self.__model.fit(self.train_df, self.train_df[self.objetive])
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
        
        plt.savefig(self.path + "/Matriz-SGD.png", dpi=300, bbox_inches="tight")

    def save(self, modelName:str) -> str:
        try:
            joblib.dump(self.__model, self.path + "/" + modelName + ".pkl")
        except Exception as e: 
            raise ValueError("[-] Error: {}".format(e))

        return "[+] Modelo guardado {}".format(self.path)
        
    
    def load(self, modelName:str):
        try:
            self.__model = joblib.load(modelName)
        except Exception as e: 
            raise ValueError("[-] {}".format(e))
        return "[+] Modelo cargado {}".format(modelName)
    
    def predict(self, path:str) -> ndarray:
        try:
            data = pd.read_csv(path, parse_dates=True, dayfirst=False, index_col="Date")
            data.dropna(inplace=True)
            data[data.columns] = data[data.columns].replace(",", "", regex=True).astype(float)
            if "Close" in data.columns:
                data = data.drop("Close", axis=1)
        except Exception as e:
            raise ValueError("[-] {}".format(e))
        
        return self.__model.predict(data)