from src.models.abstract_model import Model,ColumnExtractor
import pandas as pd 
import numpy as np 
import matplotlib as plt 
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_validate, learning_curve
import warnings
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action='ignore',category=DataConversionWarning)
np.random.seed(42)



class LinealRegression(Model):

    def train_model(self, datapath:str, trainParams:dict):
        objetive = "Close"
        try:
            df = pd.read_csv(datapath)
            df.dropna(inplace=True)
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)
            col_num = df.drop(objetive, axis=1).select_dtypes([np.number]).columns
            pipeline_extractor = Pipeline(
                ("selecter",ColumnExtractor(columns=col_num)),
                ("imputer", SimpleImputer()),
                ("standarizer", MinMaxScaler(feature_range=(0,1)))
            )
        except: 
            raise ValueError("[-] Error loading dataset")
        
        
        pipeline_processed = FeatureUnion([("pipeline_extractor", pipeline_extractor)])
        self.__model = Pipeline(
            ("pipeline_processed", pipeline_processed),
            ("estimator", LinealRegression())
        )

        metric
        if trainParams.get("metric","none") == "mse":
            metric = mean_absolute_error
        else:
            metric = mean_squared_error
        
        
        cv = trainParams.get("cv", 15)
        jobs = trainParams.get("jobs", -1)
        train_score = trainParams.get("train_score",False)

        self.result = self.evaluate_model(self.__model,df, df[objetive], metric, cv, jobs, train_score)

        print("[+] Final score: {}".format(self.result))


    
    def evaluate_model(self, model:Pipeline, data:pd.DataFrame, objetive:pd.array, metric:object,cv,jobs,train_score) -> any: 
        return cross_validate(model, data, objetive, scoring=metric, cv=cv, n_jobs=jobs, return_train_score=train_score)



    def show_result(self) :
        pass
    
    def save_model(self, modelName:str):
        pass
    
    def load_model(self, modelName:str):
        pass
    
    def test_model(self, data:pd.DataFrame):
        return self.__model.predict(data)