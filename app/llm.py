from langchain_huggingface import HuggingFacePipeline
from langchain.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from transformers import pipeline
import torch

class LLMPipelineSingleton:
    _instance = None  # Holds the single instance

    def __new__(cls, model_id):
        """Ensures only one instance is created (Singleton)."""
        if not cls._instance:
            cls._instance = super(LLMPipelineSingleton, cls).__new__(cls)
            cls._instance._initialize_pipeline(model_id)
        return cls._instance

    def _initialize_pipeline(self, model_id):
        """Loads the model only once.""" 
        self.pipe = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype=torch.bfloat16,
            model_kwargs={'cache_dir' : './model_cache'},
            device_map="auto"
        )

    def get_pipeline(self):
        """Returns the initialized Hugging Face pipeline."""
        return HuggingFacePipeline(pipeline=self.pipe)

class CrossRerankerSingleton:
    _instance = None  # Class variable to hold the single instance

    def __new__(cls, reranker_name):
        """Ensures only one instance of CrossEncoderReranker exists."""
        if cls._instance is None:
            cls._instance = super(CrossRerankerSingleton, cls).__new__(cls)
            model_kwargs = {'device' : 'cuda', 'cache_dir' : './model_cache'}
            cls._instance.cross_encoder = HuggingFaceCrossEncoder(model_name=reranker_name, model_kwargs=model_kwargs)

            cls._instance._initialize_reranker(cls._instance.cross_encoder)
        return cls._instance

    def _initialize_reranker(self, reranker_model):
        """Loads the CrossEncoderReranker only once."""
        self.reranker = CrossEncoderReranker(
            model=reranker_model    
        )

    def get_reranker(self):
        """Returns the initialized CrossEncoderReranker."""
        return self.reranker
 

