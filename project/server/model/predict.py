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

    def get_classes(self) -> str:

        list_classes = list(self.classes.values())

        return ', '.join(str(x) for x in list_classes)

    def make_prediction(self, path) -> str:

        image = Image.open(path)
        image = image.resize((32, 32))

        convert_tensor = transforms.ToTensor()
        image_tensor = convert_tensor(image)

        net = self._model_save_load.load_model()
        prediction = net(image_tensor.unsqueeze(0))
        class_pred = prediction.argmax(dim=1)

        return self.classes[int(class_pred)]
