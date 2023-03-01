import os
import requests
import telebot
import uuid

from server.model.predict import Predict
from server.config import TELEGRAM_TOKEN, PHOTO_FOLDER, WEIGHTS_PATH


token = TELEGRAM_TOKEN
bot = telebot.TeleBot(token)
predict = Predict(weight_path=WEIGHTS_PATH)


@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    """Responds to requests  /start и /help."""

    text = ('Привет я могу помочь в классификации следующих класссов:.\n\n'
            + predict.get_classes()
            + '.\n\nВысылай фото ответным сообщением.\n\n')

    chat_id = message.from_user.id
    bot.send_message(chat_id, text)


@bot.message_handler(content_types=['text'])
def get_text_messages(message):

    """Requests photo answering the text messages."""

    bot.reply_to(message, "Высылай фото ответным сообщением.")


@bot.message_handler(content_types=['photo'])
def make_prediction(photo):

    """Return class at the photo."""

    try:
        file_id = photo.json['photo'][1]['file_id']  # Gets ID of the picture.
    except IndexError:
        file_id = photo.json['photo'][0]['file_id']

    file_info = bot.get_file(file_id)
    file = requests.get('https://api.telegram.org/file/bot{0}/{1}'.format(TELEGRAM_TOKEN, file_info.file_path))
    os.makedirs(PHOTO_FOLDER, exist_ok=True)
    img_name = f'{uuid.uuid4()}.jpg'
    path_to_save = os.path.join(PHOTO_FOLDER, img_name)
    with open(path_to_save, "wb") as out:
        out.write(file.content)
        print(f'Photo saved {path_to_save}')

    print('Starting making prediction')
    prediction = predict.make_prediction(path=path_to_save)
    print('Prediction got')

    bot.reply_to(photo, prediction)


bot.polling(none_stop=True)
print('Bot polling has been started ...')

