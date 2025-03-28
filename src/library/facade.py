from src.library.factory import FactoryAI

class AImodel:
    def __init__(self, modelType:str):
        self.__modelAI = FactoryAI.create_model(modelType)
    
    def train(self, dataset:str,trainParams:dict):
        """
        Entrena el modelo seleccionado

        :param dataset: Ruta donde cargar el dataset con extensión csv
        """
        # Mostrar matriz de confución con matplotlib
        self.__modelAI.train_model()

    def save(self, path:str):
       """
       Guarda el modelo entrenado

       :param name: Ruta para guardar el modelo con. La extension se agrega automáticamente
       """
       print(self.__modelAI.save_model(path)) 

    def load(self, path:str):
        """
        Carga el modelo pasado como parametro

        :param path: Ruta donde esta el modelo guardado
        """
        print(self.__modelAI.load_model(path))
    
    def predic(self, data: any):
        """
        Devuelve el resultado del modelo entrenado

        :param data: Datos a pasar al modelo.
        """
        print(self.__modelAI.test_model(data))