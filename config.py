import os

class Config:
    UPLOAD_FOLDER = 'uploads'
    MODEL_FOLDER = 'model'
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'any_key'