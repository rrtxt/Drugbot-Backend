from flask import Flask
from huggingface_hub import login
from app.routes import main
from app.config import Config
from app.db import VectorStoreSingleton
from app.llm import LLMPipelineSingleton, CrossRerankerSingleton

def create_app(env="dev"):
    app = Flask(__name__)

    # Load config
    config = Config(env).settings
    app.config.from_object(config)

    login(token=app.config["HF_TOKEN"])

    print("Model Path: ", app.config["MODEL_CACHE"])
    # initialize vector store
    VectorStoreSingleton(app.config["CHROMA_HOST"], app.config["CHROMA_PORT"], app.config["MODEL_CACHE"]) 

    # Initialize LLM pipeline 
    llm_model_id = "meta-llama/Llama-3.2-1B"
    LLMPipelineSingleton(llm_model_id)

    # Initialize reranker model
    reranker_model_id = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    CrossRerankerSingleton(reranker_model_id) 

    print("Initialized LLM and Reranker models finished")

    app.register_blueprint(main)

    return app

