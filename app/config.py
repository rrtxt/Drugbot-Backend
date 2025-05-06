import os
from dotenv import load_dotenv

load_dotenv()

import os

class BaseConfig:
    HF_TOKEN = os.getenv("HF_TOKEN")
    MODEL_CACHE=os.getenv("MODEL_CACHE")
    CHROMA_HOST = os.getenv("CHROMA_HOST")
    CHROMA_PORT = os.getenv("CHROMA_PORT")

class DevConfig(BaseConfig):
    DEBUG = True
    ENV = "development"

class ProductionConfig(BaseConfig):
    DEBUG = False
    ENV = "production"

class Config:
    def __init__(self, env="dev"):
        if env == "prod":
            self.settings = ProductionConfig()
        else:
            self.settings = DevConfig()