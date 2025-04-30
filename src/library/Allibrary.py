from src.library.factory import FactoryAI

class AImodel:
    def __init__(self, modelType:str):
        self.__modelAI = FactoryAI.create_model(modelType)
    
    def train_model(self, dataset:str, trainParams:dict):
        """
        Entrena el modelo seleccionado

        :param dataset: Ruta donde cargar el dataset con extensión csv
        """
        # Mostrar matriz de confución con matplotlib
        self.__modelAI.train(dataset, trainParams)

    def save_model(self, path:str):
       """
       Guarda el modelo entrenado

       :param name: Ruta para guardar el modelo con. La extension se agrega automáticamente
       """
       print(self.__modelAI.save(path)) 

    def load_model(self, path:str):
        """
        Carga el modelo pasado como parametro

        :param path: Ruta donde esta el modelo guardado
        """
        print(self.__modelAI.load(path))
    
    def test_model(self, path:str):
        """
        Devuelve el resultado del modelo entrenado

        :param path: Ruta al dataset
        """
        print(self.__modelAI.predic(path))