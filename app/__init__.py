from flask import Flask
from huggingface_hub import login
from app.routes import main
from app.config import Config
from app.db import VectorStoreSingleton
from app.llm import LLMPipelineSingleton, CrossRerankerSingleton
from transformers import BitsAndBytesConfig
import torch

def create_app(env="development"):
    app = Flask(__name__)

    # Load config
    config = Config(env).settings
    app.config.from_object(config)

    login(token=app.config["HF_TOKEN"])

    print("Model Path: ", app.config["MODEL_CACHE"])
    # initialize vector store
    embeddings_model_id = "distiluse-base-multilingual-cased-v2"
    VectorStoreSingleton(
        host=app.config["CHROMA_HOST"],
        port=app.config["CHROMA_PORT"],
        model_name=embeddings_model_id,
        cache_dir=app.config["MODEL_CACHE"]
    ) 

    # Define Quantization Config
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    # Initialize LLM pipeline 
    llm_model_id = "meta-llama/Llama-3.2-3B-Instruct"
    LLMPipelineSingleton(llm_model_id, quantization_config=quantization_config)

    # Initialize reranker model
    reranker_model_id = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    CrossRerankerSingleton(reranker_model_id) 

    print("Initialized LLM and Reranker models finished!")

    app.register_blueprint(main)

    return app

