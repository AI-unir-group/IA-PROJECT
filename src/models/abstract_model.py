from abc import ABC, abstractmethod
from numpy import ndarray
from sklearn.base import BaseEstimator,TransformerMixin
import pandas as pd 


class Model(ABC):

    @abstractmethod
    def train(self, dataPath:str, trainParams:dict):
        """
        Entrena el modelo con los parámetros dados.

        :param dataPath: Ubicación de los datos.
        :param trainParams: Diccionario con parámetros adicionales de entrenamiento.
        :return: Matriz de confusión tipo ndarray de Numpy.
        """

    @abstractmethod
    def save(self, modelName:str) -> str:
         """
        Guarda el modelo con el nombre especificado.

        :param modelName: Nombre del archivo donde se guardará el modelo.
        :return: Ruta o confirmación del modelo guardado.
        """
    

    @abstractmethod
    def load(self,modelName:str) -> str:
        """
        Carga el modelo desde un archivo.

        :param modelName: Nombre del archivo del modelo a cargar.
        :return: Confirmación del modelo cargado o información relevante.
        """
    
    @abstractmethod
    def predict(self,path) -> ndarray:
        """
        Prueba el modelo con nuevos datos.
        
        :param data: Conjunto de datos para probar el modelo.
        :return: Devuelve el resultado del modelo.
        """
        


class ColumnExtractor(BaseEstimator, TransformerMixin):
    def __init__(self,columns,output_type='dataframe'):
        self.columns=columns
        self.output_type=output_type
    
    def transform(self,X,**transform_params):
        if isinstance(X,list):
            X=pd.DataFrame.from_dict(X)
        if self.output_type=='matrix':
            return X[self.columns].values
        
        elif self.output_type=='dataframe':
            return X[self.columns]
        
        raise Exception('output_type tiene que ser matrix o dataframe')
    
    def fit(self,X,y=None,**fit_params):
        return self