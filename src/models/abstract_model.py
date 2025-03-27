from abc import ABC, abstractmethod
from numpy import ndarray

class Model(ABC):

    @abstractmethod
    def train_model(self, data:ndarray, trainParams:dict) -> ndarray:
        """
        Entrena el modelo con los parámetros dados.

        :param data: Datos de entrenamiento en formato ndarray.
        :param trainParams: Diccionario con parámetros adicionales de entrenamiento.
        :return: Matriz de confusión tipo ndarray de Numpy.
        """

    @abstractmethod
    def save_model(self, modelName:str) -> str:
         """
        Guarda el modelo con el nombre especificado.

        :param modelName: Nombre del archivo donde se guardará el modelo.
        :return: Ruta o confirmación del modelo guardado.
        """
    

    @abstractmethod
    def load_model(self,modelName:str) -> str:
        """
        Carga el modelo desde un archivo.

        :param modelName: Nombre del archivo del modelo a cargar.
        :return: Confirmación del modelo cargado o información relevante.
        """
    
    @abstractmethod
    def test_model(self,data) -> any:
        """
        Prueba el modelo con nuevos datos.
        
        :param data: Conjunto de datos para probar el modelo.
        :return: Devuelve el resultado del modelo.
        """
        
