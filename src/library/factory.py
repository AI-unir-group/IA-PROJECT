from src.models.abstract_model import Model
from src.models.decision_tree import DecisionTree
from src.models.sgd import SGD
from src.models.tf_model import NeuronalNetwork

class FactoryAI():
    @staticmethod
    def create_model(model_type:str) -> Model:
       match (model_type):
        case "decision_tree":
            return DecisionTree()
        case "sgd":
            return SGD()
        case "neuronal_network":
            return NeuronalNetwork()
        case _:
           raise ValueError("[-] Modelo no seleccionado")
        