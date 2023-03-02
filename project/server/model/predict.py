import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from server.model.save_load import ModelSaveLoad


class Predict:

    def __init__(self, weight_path: str):
        self.classes = {0: 'Самолет',
                   1: 'Автомобиль',
                   2: 'Птичка',
                   3: 'Кошка',
                   4: 'Олень',
                   5: 'Собакен',
                   6: 'Лягушка',
                   7: 'Лошадь',
                   8: 'Корабль',
                   9: 'Грузовик'}
        self._model_save_load = ModelSaveLoad(weight_path=weight_path)
        self.net = self._model_save_load.load_model()

    def get_classes(self) -> str:

        list_classes = list(self.classes.values())

        return ', '.join(str(x) for x in list_classes)

    def dict_sorted_prediction(self, prediction) -> dict:

        class_preds = {}
        for key in self.classes:
            class_preds[self.classes[key]] = round(float(prediction[0, key]) * 100, 2)

        sorted_values = list(class_preds.values())
        sorted_values.sort()
        sorted_values.reverse()

        sorted_dict = {}
        for i in sorted_values:
            for k in class_preds.keys():
                if class_preds[k] == i:
                    sorted_dict[k] = class_preds[k]

        return sorted_dict

    def text_prediction(self, prediction) -> str:

        dict_prediction = self.dict_sorted_prediction(prediction=prediction)

        converted = str('Хм... Мне кажется список вероятностей принадлежности классу такой:.\n\n')

        for key in dict_prediction:
            converted += key + ": " + f'{str(dict_prediction[key])}%' + "\n"

        return converted

    def make_prediction(self, path) -> str:

        image = Image.open(path)
        image = image.resize((32, 32))

        image.save(path)

        convert_tensor = transforms.ToTensor()
        image_tensor = convert_tensor(image)

        prediction = self.net(image_tensor.unsqueeze(0))
        prediction = torch.nn.functional.softmax(prediction, dim=1).data.cpu().numpy()

        return self.text_prediction(prediction=prediction)
