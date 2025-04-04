from src.models.abstract_model import Model
from src.models.decision_tree import DecisionTree
from src.models.sgd import SGD
from src.models.tf_model import TF

class FactoryAI():
    @staticmethod
    def create_model(model_type:str) -> Model:
       match (model_type):
        case "tree":
            return DecisionTree()
        case "sgd":
            return SGD()
        case "tf":
            return TF()
        case _:
           raise ValueError("[-] Modelo no seleccionado")
        