from starlette.config import Config

config = Config('.env')

TELEGRAM_TOKEN = config('TELEGRAM_TOKEN', cast=str)
PHOTO_FOLDER = config('PHOTO_FOLDER', cast=str)
WEIGHTS_PATH = config('WEIGHTS_PATH', cast=str)
