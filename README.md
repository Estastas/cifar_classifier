# cifar_classifier
Телеграм бот с классификатором, созданном на основе архитектуры AlexNet и обученным на датасете CIFAR.

В исходной архитектуре были изменены Avg_pool -> Max_pool, сверточные слои 5х5 -> 2 * (3x3), добавлена нормировка.

Модель училась в Colab и там сохранены веса, в текущей версии бота переобучение не реализовано, но код есть:
project/server/model

## Deploy

### UP bot
make up

### DOWN bot
make down




