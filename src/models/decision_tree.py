from src.models.abstract_model import Model


class DecisionTree(Model):
    model: any

    def __init__(self):
        pass

    def train_model(self, data, trainParams):
        pass
    
    def save_model(self, modelName):
        pass
    
    def load_model(self, modelName):
        pass
    
    def test_model(self, data):
        pass
    