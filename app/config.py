import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import os

class BaseConfig:
    HF_TOKEN = os.getenv("HF_TOKEN")
    MODEL_CACHE= os.getenv("MODEL_CACHE")
    CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
    CHROMA_PORT = os.getenv("CHROMA_PORT", "8000")
    MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    MONGODB_DBNAME = os.getenv("MONGODB_DBNAME", "drugbot")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

class DevConfig(BaseConfig):
    DEBUG = True
    ENV = "development"

class ProductionConfig(BaseConfig):
    DEBUG = False
    ENV = "production"

class Config:
    def __init__(self, env="development"):
        if env == "production":
            self.settings = ProductionConfig()
        else:
            self.settings = DevConfig()