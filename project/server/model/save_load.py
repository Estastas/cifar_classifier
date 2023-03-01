import torch
from server.model.model import CIFARNet
import __main__


class ModelSaveLoad:

    def __init__(self, weight_path: str):
        self.weight_path = weight_path
        setattr(__main__, "CIFARNet", CIFARNet)

    def save_model(self, net) -> bool:
        torch.save(net, self.weight_path)
        return True

    def load_model(self):

        print(f'Model loading from {self.weight_path}')
        net = torch.load(self.weight_path, map_location=torch.device("cpu"))
        print('Model loaded successfully')

        return net


